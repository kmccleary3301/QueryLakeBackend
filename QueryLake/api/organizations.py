import os
from ..database import sql_db_tables
from sqlmodel import Session, select, and_
import time
from ..database import encryption
from .user_auth import *
from .hashing import random_hash

server_dir = "/".join(os.path.dirname(os.path.realpath(__file__)).split("/")[:-1])
user_db_path = server_dir+"/user_db/files/"

def create_organization(database : Session, username : str, password_prehash : str, organization_name : str, organization_description : str = None):
    """
    Add an organization to the db. Verify the user first.
    """
    user = get_user(database, username, password_prehash)
    
    (public_key, private_key) = encryption.ecc_generate_public_private_key()

    new_organization = sql_db_tables.organization(
        hash_id=random_hash(),
        name=organization_name,
        creation_timestamp=time.time(),
        public_key=public_key
    )
    database.add(new_organization)
    database.commit()
    database.flush()
    
    private_key_secured = encryption.ecc_encrypt_string(user.public_key, private_key)

    new_membership = sql_db_tables.organization_membership(
        role="owner",
        organization_id=new_organization.id,
        user_name=user.name,
        organization_private_key_secure=private_key_secured,
        invite_still_open=False
    )

    database.add(new_membership)
    database.commit()
    return {"success": True, "organization_id": new_organization.id, "organization_dict": new_organization.__dict__, "membership_dict": new_membership.__dict__}

def invite_user_to_organization(database : Session, username : str, password_prehash : str, username_to_invite : str, member_class : str, organization_id : int):
    """
    Invite a user to organization. 
    Raise an error if they are already in the organization.
    """
    assert member_class in ["owner", "admin", "member", "viewer"], "Invalid member class"
    user = get_user(database, username, password_prehash)
    invitee = database.exec(select(sql_db_tables.user).where(sql_db_tables.user.name == username_to_invite)).first()

    org_get = database.exec(select(sql_db_tables.organization).where(sql_db_tables.organization.id == organization_id)).all()
    assert len(org_get) > 0, "Organization not found"
    membership_get = database.exec(select(sql_db_tables.organization_membership).where(
        sql_db_tables.organization_membership.organization_id == organization_id and \
        sql_db_tables.organization_membership.user_name == username)).all()
    
    assert len(membership_get) > 0, "Not a member of given organization"

    if (member_class == "owner"):
        assert membership_get[0].role == "owner", "Invalid Permissions"
    if (member_class == "admin"):
        assert membership_get[0].role in ["owner", "admin"], "Invalid Permissions"
    # if (member_class == "admin"):
    #     assert membership_get[0].role in ["owner", "admin"], "Invalid Permissions"

    private_key_encryption_salt = user.private_key_encryption_salt
    user_private_key_decryption_key = hash_function(password_prehash, private_key_encryption_salt, only_salt=True)

    user_private_key = encryption.ecc_decrypt_string(user_private_key_decryption_key, user.private_key_secured)

    organization_private_key = encryption.ecc_decrypt_string(user_private_key, membership_get[0].organization_private_key_secure)

    new_membership = sql_db_tables.organization_membership(
        role=member_class,
        organization_id=organization_id,
        organization_private_key_secure=encryption.ecc_encrypt_string(invitee.public_key, organization_private_key),
        invite_sender_user_name=username,
    )

    database.add(new_membership)
    database.commit()

    return {"success" : True}

def accept_organization_invitation(database : Session, username : str, password_prehash : str, organization_id : int):
    """
    Given the index of an organization, find the membership between user and org and accept the invite.
    """
    user = get_user(database, username, password_prehash)
    # invitee = database.exec(select(sql_db.user).where(sql_db.user.name == username_to_invite)).first()

    org_get = database.exec(select(sql_db_tables.organization).where(sql_db_tables.organization.id == organization_id)).all()
    assert len(org_get) > 0, "Organization not found"
    membership_get = database.exec(select(sql_db_tables.organization_membership).where(
        and_(sql_db_tables.organization_membership.organization_id == organization_id,
        sql_db_tables.organization_membership.user_name == username))).all()

    if membership_get[0].invite_still_open == False:
        raise ValueError("Invitation already accepted")
    
    membership_get[0].invite_still_open = False
    database.commit()

    return {"success": True}

def fetch_memberships(database : Session, username : str, password_prehash : str, return_subset : str = "accepted"):
    """
    Returns a list of dicts for organizations for which the user has a membership table connecting the two.
    return_subset is "accepted" | "open_invitations" | "all".
    dicts contain: org_name, org_id, role, accepted    
    """
    assert return_subset in ["accepted", "open_invitations", "all"], "Invalid return type specification."

    user = get_user(database, username, password_prehash)
    if return_subset == "all":
        membership_get = database.exec(select(sql_db_tables.organization_membership).where(sql_db_tables.organization_membership.user_name == username)).all()
    elif return_subset == "open_invitations":
        membership_get = database.exec(select(sql_db_tables.organization_membership).where(and_(
            sql_db_tables.organization_membership.user_name == username,
            sql_db_tables.organization_membership.invite_still_open == True
            ))).all()
    else:
         membership_get = database.exec(select(sql_db_tables.organization_membership).where(and_(
            sql_db_tables.organization_membership.user_name == username,
            sql_db_tables.organization_membership.invite_still_open == False
            ))).all()
    
    organizations = []
    for membership in membership_get:
        organization = database.exec(select(sql_db_tables.organization).where(sql_db_tables.organization.id == membership.organization_id)).first()
        organizations.append({
            "organization_id": organization.id,
            "organization_name": organization.name,
            "role": membership.role,
            "accepted": membership.invite_still_open,
        })
    
    return {"success": True, "memberships": organizations, "admin": user.is_admin}
