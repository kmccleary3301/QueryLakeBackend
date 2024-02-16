import os
from ..database import sql_db_tables
from sqlmodel import Session, select, and_
import time
from ..database import encryption
# from .user_auth import get_user_private_key
from .single_user_auth import get_user
from .hashing import random_hash, hash_function
from ray import get
from ray.actor import ActorHandle
from ..typing.config import Config, AuthType, getUserType

server_dir = "/".join(os.path.dirname(os.path.realpath(__file__)).split("/")[:-1])
# user_db_path = server_dir+"/user_db/files/"

membership_value_map = {
    "owner": 4,
    "admin": 3,
    "member": 2,
    "viewer": 1
}

def create_organization(database : Session, 
                        auth : AuthType, 
                        organization_name : str, 
                        organization_description : str = None):
    """
    Add an organization to the db. Verify the user first.
    """
    user_retrieved : getUserType  = get_user(database, auth)
    (user, user_auth) = user_retrieved
    
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
        user_name=user_auth.username,
        organization_private_key_secure=private_key_secured,
        invite_still_open=False
    )

    database.add(new_membership)
    database.commit()
    # return {"success": True, "organization_id": new_organization.id, "organization_dict": new_organization.__dict__, "membership_dict": new_membership.__dict__}
    return {"organization_id": new_organization.id, "organization_dict": new_organization.__dict__, "membership_dict": new_membership.__dict__}

def invite_user_to_organization(database : Session, 
                                auth : AuthType, 
                                username_to_invite : str, 
                                organization_id : int, 
                                member_class : str = "member"):
    """
    Invite a user to organization. 
    Raise an error if they are already in the organization.
    """
    assert member_class in ["owner", "admin", "member", "viewer"], "Invalid member class"
    user_retrieved : getUserType  = get_user(database, auth)
    (user, user_auth) = user_retrieved
    invitee = database.exec(select(sql_db_tables.user).where(sql_db_tables.user.name == username_to_invite)).first()

    org_get = database.exec(select(sql_db_tables.organization).where(sql_db_tables.organization.id == organization_id)).all()
    assert len(org_get) > 0, "Organization not found"
    membership_get = database.exec(select(sql_db_tables.organization_membership).where(
        sql_db_tables.organization_membership.organization_id == organization_id and \
        sql_db_tables.organization_membership.user_name == user_auth.username)).all()
    
    assert len(membership_get) > 0, "Not a member of given organization"

    # Must be above the offered role, or be owner.
    assert membership_value_map[membership_get[0].role] > membership_value_map[member_class] or membership_get[0].role == "owner", "Attempted Priviledge Escalation"

    if (member_class == "owner"):
        assert membership_get[0].role == "owner", "Invalid Permissions"
    if (member_class == "admin"):
        assert membership_get[0].role in ["owner", "admin"], "Invalid Permissions"
    # if (member_class == "admin"):
    #     assert membership_get[0].role in ["owner", "admin"], "Invalid Permissions"

    # private_key_encryption_salt = user.private_key_encryption_salt
    # user_private_key_decryption_key = hash_function(password_prehash, private_key_encryption_salt, only_salt=True)

    # user_private_key = encryption.ecc_decrypt_string(user_private_key_decryption_key, user.private_key_secured)
    private_key_encryption_salt = user.private_key_encryption_salt
    user_private_key_decryption_key = hash_function(user_auth.password_prehash, private_key_encryption_salt, only_salt=True)

    user_private_key = encryption.aes_decrypt_string(user_private_key_decryption_key, user.private_key_secured)

    # user_private_key = get_user_private_key(database, username, password_prehash)["private_key"]

    organization_private_key = encryption.ecc_decrypt_string(user_private_key, membership_get[0].organization_private_key_secure)

    new_membership = sql_db_tables.organization_membership(
        role=member_class,
        organization_id=organization_id,
        organization_private_key_secure=encryption.ecc_encrypt_string(invitee.public_key, organization_private_key),
        invite_sender_user_name=user_auth.username,
        user_name=username_to_invite
    )

    database.add(new_membership)
    database.commit()

    # return {"success" : True}
    return True

def resolve_organization_invitation(database : Session, 
                                    auth : AuthType, 
                                    organization_id : int, 
                                    accept : bool):
    """
    Given the index of an organization, find the membership between user and org and accept or decline the invite.
    """
    user_retrieved : getUserType  = get_user(database, auth)
    (user, user_auth) = user_retrieved
    # invitee = database.exec(select(sql_db.user).where(sql_db.user.name == username_to_invite)).first()

    org_get = database.exec(select(sql_db_tables.organization).where(sql_db_tables.organization.id == organization_id)).all()
    assert len(org_get) > 0, "Organization not found"
    membership_get = database.exec(select(sql_db_tables.organization_membership).where(
        and_(sql_db_tables.organization_membership.organization_id == organization_id,
        sql_db_tables.organization_membership.user_name == user_auth.username))).all()

    if membership_get[0].invite_still_open == False:
        raise ValueError("Invitation already accepted")
    
    if accept:
        membership_get[0].invite_still_open = False
        database.commit()
    else:
        database.delete(membership_get[0])
        database.commit()

    # return {"success": True}
    return True

def fetch_memberships(database : Session, 
                      auth : AuthType, 
                      return_subset : str = "accepted"):
    """
    Returns a list of dicts for organizations for which the user has a membership table connecting the two.
    return_subset is "accepted" | "open_invitations" | "all".
    dicts contain: org_name, org_id, role, accepted    
    """
    assert return_subset in ["accepted", "open_invitations", "all"], "Invalid return type specification."

    user_retrieved : getUserType  = get_user(database, auth)
    (user, user_auth) = user_retrieved
    
    
    if return_subset == "all":
        membership_get = database.exec(select(sql_db_tables.organization_membership).where(sql_db_tables.organization_membership.user_name == user_auth.username)).all()
    elif return_subset == "open_invitations":
        membership_get = database.exec(select(sql_db_tables.organization_membership).where(and_(
            sql_db_tables.organization_membership.user_name == user_auth.username,
            sql_db_tables.organization_membership.invite_still_open == True
            ))).all()
    else:
         membership_get = database.exec(select(sql_db_tables.organization_membership).where(and_(
            sql_db_tables.organization_membership.user_name == user_auth.username,
            sql_db_tables.organization_membership.invite_still_open == False
            ))).all()
    
    organizations = []
    for membership in membership_get:
        organization = database.exec(select(sql_db_tables.organization).where(sql_db_tables.organization.id == membership.organization_id)).first()
        
        org_result = {
            "organization_id": organization.id,
            "organization_name": organization.name,
            "role": membership.role,
            "invite_still_open": membership.invite_still_open,
        }
        if (org_result["invite_still_open"]):
            org_result.update({
                "sender": membership.invite_sender_user_name
            })
        organizations.append(org_result)
    
    # return {"success": True, "memberships": organizations, "admin": user.is_admin}
    return {"memberships": organizations, "admin": user.is_admin}

def fetch_memberships_of_organization(database : Session, 
                                      auth : AuthType, 
                                      organization_id : int):
    """
    Fetches all active memberships of an organization, first verifying that the user is in the given organization.
    """

    user_retrieved : getUserType  = get_user(database, auth)
    (user, user_auth) = user_retrieved

    organization = database.exec(select(sql_db_tables.organization).where(sql_db_tables.organization.id == organization_id)).first()

    membership_get = database.exec(select(sql_db_tables.organization_membership).where(
        sql_db_tables.organization_membership.organization_id == organization_id and \
        sql_db_tables.organization_membership.user_name == user_auth.username)).all()
    
    assert len(membership_get) > 0, "Not a member of given organization"

    memberships_get = database.exec(select(sql_db_tables.organization_membership).where(sql_db_tables.organization_membership.organization_id == organization_id)).all()
    
    memberships = []
    for membership in memberships_get:
        memberships.append({
            "organization_id": organization.id,
            "organization_name": organization.name,
            "role": membership.role,
            "username": membership.user_name,
            "invite_still_open": membership.invite_still_open,
        })
    
    # return {"success": True, "memberships": memberships}
    return {"memberships": memberships}