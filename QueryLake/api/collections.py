import os
from ..database import sql_db_tables
from sqlmodel import Session, select, delete, and_
import time
from .user_auth import *
from .hashing import random_hash
from ..typing.config import AuthType
from ..misc_functions.function_run_clean import file_size_as_string
from ..database import encryption
from io import BytesIO
from tqdm import tqdm

def fetch_document_collections_belonging_to(database : Session, 
                                            auth : AuthType, 
                                            organization_id : int = None, 
                                            global_collections : bool = False):
    """
    Gets a list of dicts for document collections. 
    If organization_id is none, return the user's personal collections. 
    If it is provided, check the user is an accepted member of the org, then retrieve the collections.
    """
    
    (user, user_auth) = get_user(database, auth)
    if global_collections:
        collections_get = database.exec(select(sql_db_tables.document_collection).where(sql_db_tables.document_collection.collection_type == "global")).all()
    if not organization_id is None:
        membership_get = database.exec(select(sql_db_tables.organization_membership).where(
            sql_db_tables.organization_membership.user_name == user_auth.username and \
            sql_db_tables.organization_membership.organization_id == organization_id)).all()
        assert len(membership_get) > 0, "User not in organization"
        collections_get = database.exec(select(sql_db_tables.document_collection).where(sql_db_tables.document_collection.author_organization == organization_id)).all()
    else:
        collections_get = database.exec(select(sql_db_tables.document_collection).where(sql_db_tables.document_collection.author_user_name == user_auth.username)).all()
    collections_return = []
    for collection in collections_get:
        collections_return.append({
            "name": collection.name,
            "id" : collection.id,
            "document_count": collection.document_count
        })
    # return {"success": True, "collections": collections_return}
    return {"collections": collections_return}

def create_document_collection(database : Session, 
                               auth : AuthType, 
                               name : str,
                               description : str = None,
                               public : bool = False,
                               organization_id : int = None):
    """
    Create a new document collection. 
    If no hash_id is given, one is created, then returned.
    """
    hash_id = random_hash()
    
    encryption_key = random_hash()
    
    (user, user_auth) = get_user(database, auth)
    
    
    
    if not organization_id is None:
        membership_get = database.exec(select(sql_db_tables.organization_membership).where(and_(
            sql_db_tables.organization_membership.user_name == user_auth.username,
            sql_db_tables.organization_membership.organization_id == organization_id))).all()
        
        assert len(membership_get) > 0, "User not in organization"
        assert membership_get[0].role != "viewer", "Invalid Permissions"
        
        organization = database.exec(select(sql_db_tables.organization).where(sql_db_tables.organization.id == organization_id)).first()
        
        new_doc_password = random_hash(base=16)
        encryption_key = random_hash(base=16)
        
        encryption_key_secure = encryption.ecc_encrypt_string(organization.public_key, encryption_key)
        doc_password_secure = encryption.aes_encrypt_string(encryption_key, new_doc_password)
        
        new_collection = sql_db_tables.document_collection(
            name=name,
            author_organization=organization_id,
            creation_timestamp=time.time(),
            public=public,
            description=description,
            encryption_key_secure=encryption_key_secure,
            document_unlock_key=doc_password_secure,
            collection_type="organization"
        )
    else:
        new_doc_password = random_hash(base=16)
        encryption_key = random_hash(base=16)
        
        encryption_key_secure = encryption.ecc_encrypt_string(user.public_key, encryption_key)
        doc_password_secure = encryption.aes_encrypt_string(encryption_key, new_doc_password)
        
        new_collection = sql_db_tables.document_collection(
            name=name,
            author_user_name=user_auth.username,
            creation_timestamp=time.time(),
            public=public,
            description=description,
            encryption_key_secure=encryption_key_secure,
            document_unlock_key=doc_password_secure,
            collection_type="user"
        )
    database.add(new_collection)
    database.commit()
    database.flush()
    # return {"success": True, "hash_id": new_collection.hash_id}
    return {"hash_id": new_collection.id}

def fetch_all_collections(database : Session, 
                          auth : AuthType):
    """
    Fetches all collections that a user has priviledge to read.
    """
    (user, user_auth) = get_user(database, auth)

    fetch_memberships = database.exec(
        select(sql_db_tables.organization_membership)
        .where(
            and_(
                sql_db_tables.organization_membership.user_name == user_auth.username,
                sql_db_tables.organization_membership.invite_still_open == False
            )
        )
    ).all()
    organizations = []
    for membership in fetch_memberships:
        if not membership.invite_still_open:
            organizations.append(database.exec(select(sql_db_tables.organization).where(sql_db_tables.organization.id == membership.organization_id)).first())
    
    return_value = {"global_collections": [], "user_collections": []}

    organization_collections = {}
    for organization in organizations:
        collections = database.exec(select(sql_db_tables.document_collection).where(sql_db_tables.document_collection.author_organization == organization.id)).all()
        organization_collections[organization.id] = {"name": organization.name, "collections": []}
        for collection in collections:
            organization_collections[organization.id]["collections"].append({
                "hash_id": collection.id,
                "name": collection.name,
                "document_count": collection.document_count,
                "type": "organization",
            })
    return_value["organization_collections"] = organization_collections
    global_collections = database.exec(select(sql_db_tables.document_collection).where(sql_db_tables.document_collection.collection_type == "global")).all()
    for collection in global_collections:
        return_value["global_collections"].append({
            "name": collection.name,
            "hash_id": collection.id,
            "document_count": collection.document_count,
            "type": "global"
        })
    user_collections = database.exec(select(sql_db_tables.document_collection).where(
        and_(
            sql_db_tables.document_collection.author_user_name == user_auth.username,
            sql_db_tables.document_collection.collection_type == "user"
        )
    )).all()
    for collection in user_collections:
        return_value["user_collections"].append({
            "name": collection.name,
            "hash_id": collection.id,
            "document_count": collection.document_count,
            "type" : "user"
        })
    # return {"success": True, "result": return_value}
    return {"collections": return_value}

def fetch_collection(database : Session,
                     auth : AuthType,
                     collection_hash_id : str):
    """
    Retrieves details of a collection for user, 
    including all documents in the collection.
    """

    (user, user_auth) = get_user(database, auth)
    collection = database.exec(select(sql_db_tables.document_collection).where(and_(sql_db_tables.document_collection.id == collection_hash_id))).first()
    assert not collection is None, "Collection not found"
    
    if collection.collection_type in ["user", "toolchain_session"]:
        if collection.public == False:
            assert collection.author_user_name == user_auth.username, "User not authorized to view collection"
        
        ret_args = {"owner": "personal"}
    elif collection.collection_type == "organization":
        user_membership = database.exec(select(sql_db_tables.organization_membership).where(and_(sql_db_tables.organization_membership.organization_id == collection.author_organization,
                                                                                          sql_db_tables.organization_membership.user_name == user_auth.username))).all()
        assert len(user_membership) > 0 or collection.public == True, "User not authorized to view collection"
        author_org = database.exec(select(sql_db_tables.organization).where(sql_db_tables.organization.id == collection.author_organization)).first()
        assert not author_org is None, f"Author organization `{collection.author_organization}` not found"
        ret_args = {"owner": author_org.id, "owner_org_name": author_org.name}
    elif collection.collection_type == "global":
        ret_args = {"owner": "global"}
    else:
        raise Exception(f"Invalid collection type `{collection.collection_type}`")

    data = {
        "title" : collection.name,
        "description": collection.description,
        "type": collection.collection_type,
        **ret_args,
        "public": collection.public,
        "document_count": collection.document_count
    }
    # for document in documents:
    #     data["document_list"].append({
    #         "title": document.file_name,
    #         "hash_id": document.id,
    #         "size": file_size_as_string(document.size_bytes),
    #         "finished_processing": document.finished_processing,
    #     })
    # return {"success": True, "result": data}
    return data

def fetch_collection_documents(database : Session,
                               auth : AuthType,
                               collection_hash_id : str,
                               limit : int = 100,
                               offset : int = 0):
    assert (limit > 0 and offset >= 0), "Invalid limit or offset, both must be ints >= 0"
    assert (limit <= 500), "Limit must be <= 500"
    fetch_collection(database, auth, collection_hash_id) # Doing this to authenticate user perms.
    documents = database.exec(
        select(sql_db_tables.document_raw)
        .where(sql_db_tables.document_raw.document_collection_id == collection_hash_id)
        .offset(offset)
        .limit(limit)
    ).all()
    
    results = list(map(lambda x: {
        "title": x.file_name,
        "hash_id": x.id,
        "size": file_size_as_string(x.size_bytes),
        "md": x.md,
        "finished_processing": x.finished_processing,
    }, list(documents)))
    
    return results

def modify_document_collection(database : Session,
                                auth : AuthType,
                                collection_hash_id : str,
                                title : str = None,
                                description : str = None):
    """
    Changes document collection properties for a user.
    
    TODO: Implement public/private switch.
    """
    
    (user, user_auth) = get_user(database, auth)

    collection = database.exec(select(sql_db_tables.document_collection).where(and_(sql_db_tables.document_collection.id == collection_hash_id))).first()
    assert not collection is None, "Collection not found"
    
    if collection.collection_type == "user":
        if collection.public == False:
            assert collection.author_user_name == user_auth.username, "User not authorized to modify collection"
    elif collection.collection_type == "organization":
        memberships = database.exec(select(sql_db_tables.organization_membership).where(and_(sql_db_tables.organization_membership.organization_id == collection.author_organization,
                                                                                          sql_db_tables.organization_membership.user_name == user_auth.username))).all()
        assert len(memberships) > 0 or collection.public == True, "User not authorized to modify collection"
        assert len(memberships) > 0 and memberships[0].role in ["owner", "admin", "member"], "User not authorized to modify collection"
    elif collection.collection_type == "global":
        assert user.is_admin == True, "User not authorized to modify collection"

    if not title is None:
        collection.name = title
    if not description is None:
        collection.description = description

    # return {"success": True}
    return True

def delete_document_collection(database : Session, 
                               auth : AuthType,
                               collection_id : str,):
    (user, user_auth) = get_user(database, auth)
    
    
    # Organization collection first
    collection = database.exec(select(sql_db_tables.document_collection).where(sql_db_tables.document_collection.id == collection_id)).first()
    assert not collection is None, "Collection not found"
    
    # Organization collection
    if collection.collection_type == "organization":
        organization = database.exec(select(sql_db_tables.organization).where(sql_db_tables.organization.id == collection.author_organization_id)).first()

        memberships = database.exec(select(sql_db_tables.organization_membership).where(and_(sql_db_tables.organization_membership.organization_id == organization.id,
                                                                                    sql_db_tables.organization_membership.user_name == user_auth.username))).all()

        if not user.is_admin:
            assert len(memberships) > 0, "User has no membership in organization."
            assert memberships[0].role in ["owner", "admin"], f"User of role `{memberships[0].role}` is not authorized to delete collections, must be owner or admin."
    
    # User collection
    elif collection.collection_type == "user" or collection.collection_type == "toolchain_session":
        assert (collection.author_user_name == user_auth.username) or user.is_admin, "User not authorized"
    elif collection.collection_type == "global":
        assert user.is_admin, "User not authorized"
    else:
        raise Exception(f"Invalid collection type `{collection.collection_type}`")
    
    # Wipe the chunks for all documents in the collection.
    database.exec(delete(sql_db_tables.DocumentChunk).where(sql_db_tables.DocumentChunk.collection_id == collection_id))
    # Wipe the documents
    database.exec(delete(sql_db_tables.document_raw).where(sql_db_tables.document_raw.document_collection_id == collection_id))
    # Wipe the zip blobs for the document bytes
    database.exec(delete(sql_db_tables.document_zip_blob).where(sql_db_tables.document_zip_blob.document_collection_id == collection_id))
    
    database.delete(collection)
    
    database.commit()
    
    return True
    

def assert_collections_priviledge(database : Session, 
                                  auth : AuthType,
                                  collection_ids : List[str],
                                  modifying : bool = False):
    """
    Ensure that the given user can view the collections.
    """
    (user, user_auth) = get_user(database, auth)
    
    assert len(collection_ids) > 0, "No collections provided"
    assert len(collection_ids) < 2000, "Too many collections provided (max is 2000). Please split up your call."
    
    collections = list(database.exec(
        select(sql_db_tables.document_collection)
        .where(sql_db_tables.document_collection.id.in_(collection_ids))
    ).all())
    
    organization_collections = [x for x in collections if x.collection_type == "organization"]
    user_collections = [x for x in collections if x.collection_type == "user"]
    
    
    if len(organization_collections) > 0:
        organization_ids = list(map(lambda x: x.author_organization, organization_collections))
        # organizations = database.exec(select(sql_db_tables.organization).where(sql_db_tables.organization.id == collection.author_organization_id)).first()
        # for collection in organization_collections:
        memberships : List[sql_db_tables.organization_membership] = list(database.exec(
            select(sql_db_tables.organization_membership)
            .where(and_(sql_db_tables.organization_membership.organization_id.in_(organization_ids),
                    sql_db_tables.organization_membership.user_name == user_auth.username))
        ).all())
        
        membership_lookup = {x.organization_id: x for x in memberships}
        viewables = {
            org_collection.id: org_collection.author_organization in membership_lookup 
            for org_collection in organization_collections
        }
        if modifying:
            viewables = {
                org_collection.id: membership_lookup[org_collection.author_organization].role in ["owner", "admin", "member"]
                for org_collection in organization_collections
            }
        
        assert all(list(viewables.values())), f"You are not authorized to view the following organization collections: {str([k for k, v in viewables.items() if not v])}"
    
    if len(user_collections) > 0:
        viewables = {
            user_collection.id: user_collection.author_user_name == user.name or user_collection.public
            for user_collection in user_collections
        }
        assert all(list(viewables.values())), f"You are not authorized to view the following user collections: {str([k for k, v in viewables.items() if not v])}"

        # assert not membership is None, "User not authorized to view collection"
    
    return organization_collections + user_collections
    
 
def get_collection_document_password(
    database : Session, 
    auth: AuthType,
    collection_id: str
):
    """
    Get the decryption password for all documents in a collection.
    [NOT MEANT TO BE EXPOSED TO API]
    """

    (user, user_auth) = get_user(database, auth)

    
    collection = database.exec(select(sql_db_tables.document_collection).where(sql_db_tables.document_collection.id == collection_id)).first()
    assert not collection is None, "Collection not found"
    
    if collection.collection_type == "global" or collection.public == True:
        
        # Encryption key stored plainly for global and public collections.
        encryption_key = collection.encryption_key_secure
        document_password = encryption.aes_decrypt_string(encryption_key, collection.document_unlock_key)
    
    elif collection.collection_type == "user":
        
        assert collection.author_user_name == user_auth.username, "User not authorized"
        private_key_encryption_salt = user.private_key_encryption_salt
        user_private_key_decryption_key = hash_function(user_auth.password_prehash, private_key_encryption_salt, only_salt=True)

        user_private_key = encryption.aes_decrypt_string(user_private_key_decryption_key, user.private_key_secured)

        encryption_key_secured = collection.encryption_key_secure
        encryption_key = encryption.ecc_decrypt_string(user_private_key, encryption_key_secured)
        document_password = encryption.aes_decrypt_string(encryption_key, collection.document_unlock_key)
        
    elif collection.collection_type == "organization":
        organization = database.exec(select(sql_db_tables.organization).where(sql_db_tables.organization.id == collection.author_organization)).first()

        memberships = database.exec(select(sql_db_tables.organization_membership).where(and_(sql_db_tables.organization_membership.organization_id == organization.id,
                                                                                    sql_db_tables.organization_membership.user_name == user_auth.username))).all()
        assert len(memberships) > 0, "User not authorized"
        
        private_key_encryption_salt = user.private_key_encryption_salt
        user_private_key_decryption_key = hash_function(user_auth.password_prehash, private_key_encryption_salt, only_salt=True)

        user_private_key = encryption.aes_decrypt_string(user_private_key_decryption_key, user.private_key_secured)

        organization_private_key = encryption.ecc_decrypt_string(user_private_key, memberships[0].organization_private_key_secure)
        
        encryption_key_secured = collection.encryption_key_secure
        encryption_key = encryption.ecc_decrypt_string(organization_private_key, encryption_key_secured)
        document_password = encryption.aes_decrypt_string(encryption_key, collection.document_unlock_key)
    
    elif collection.collection_type == "toolchain_session":
        # private_key_encryption_salt = user.private_key_encryption_salt
        # user_private_key_decryption_key = hash_function(user_auth.password_prehash, private_key_encryption_salt, only_salt=True)
        # user_private_key = encryption.aes_decrypt_string(user_private_key_decryption_key, user.private_key_secured)
        # document_password = encryption.ecc_decrypt_string(user_private_key, document.encryption_key_secure)
        encryption_key = collection.encryption_key_secure
        document_password = encryption.aes_decrypt_string(encryption_key, collection.document_unlock_key)
        
    else:
        raise ValueError(f"Collection type `{collection.collection_type}` not supported on this method yet.")
    
    return document_password
    

def change_collection_ownership(
    database : Session, 
    auth: AuthType,
    collection_id: str,
    organization_id: str = None,
    user_name: str = None,
    global_collection: bool = False,
    public: bool = None
):
    """
    Change the ownership of a collection.
    Could be organization, user, or global (only if user is an admin).
    """
    
    (user, user_auth) = get_user(database, auth)
    
    assert isinstance(global_collection, bool), "Invalid global collection value, must be boolean"
    
    data_val = [
        not organization_id is None,
        not user_name is None,
        global_collection
    ]
    
    assert sum([1 if e else 0 for e in data_val]) == 1, "Must provide exactly one of organization_id, user_name, or global_collection as True."
    
    collection = database.exec(
        select(sql_db_tables.document_collection)
        .where(sql_db_tables.document_collection.id == collection_id)
    ).first()
    
    assert not collection is None, "Collection not found"
    
    if public is None:
        public = collection.public
    
    assert isinstance(public, bool), "Invalid public value, must be boolean"
    
    if collection.collection_type == "organization":
        organization = database.exec(
            select(sql_db_tables.organization)
            .where(sql_db_tables.organization.id == collection.author_organization)
        ).first()
        membership = database.exec(
            select(sql_db_tables.organization_membership)
            .where(and_(
                sql_db_tables.organization_membership.organization_id == organization.id,
                sql_db_tables.organization_membership.user_name == user_auth.username,
                sql_db_tables.organization_membership.invite_still_open == False
            ))
        ).first()
        assert not membership is None, "User not authorized"
        assert membership.role in ["owner", "admin"], "User not authorized, must be owner or admin to change collection ownership."  
        print("First switch met condition `org`")
        
    elif collection.collection_type == "user":
        assert collection.author_user_name == user_auth.username, "User not authorized"
        print("First switch met condition `user`")
    elif collection.collection_type == "global":
        assert user.is_admin, "User not authorized"
        print("First switch met condition `global`")
    else:
        print("First switch met condition `exit`")
        raise ValueError(f"Collection type `{collection.collection_type}` not supported on this method.")
    
    document_password_unencrypted = get_collection_document_password(database, auth, collection_id)
    
    new_collection_type = "organization" if organization_id else "user" if user_name else "global"
    
    if new_collection_type == "global" or public:
        new_encryption_key = random_hash(base=16)
        doc_password_secured = encryption.aes_encrypt_string(new_encryption_key, document_password_unencrypted)
        
        collection.encryption_key_secure = new_encryption_key
        collection.document_unlock_key = doc_password_secured
        print("Second switch met condition `global or switch`")
    elif new_collection_type == "user":
        new_encryption_key = random_hash(base=16)
        new_encryption_key_secured = encryption.ecc_encrypt_string(user.public_key, new_encryption_key)
        doc_password_secured = encryption.aes_encrypt_string(new_encryption_key, document_password_unencrypted)
        
        collection.encryption_key_secure = new_encryption_key_secured
        collection.document_unlock_key = doc_password_secured
        print("Second switch met condition `user`")
        
    if new_collection_type in ["global", "user"]:
        collection.author_user_name = user.name
        collection.author_organization = None
        
        print("Third switch met condition `global or user`")
    elif new_collection_type == "organization":
        
        organization_new = database.exec(
            select(sql_db_tables.organization)
            .where(sql_db_tables.organization.id == organization_id)
        ).first()
        assert not organization_new is None, "Organization not found"
        
        membership_new = database.exec(
            select(sql_db_tables.organization_membership)
            .where(and_(
                sql_db_tables.organization_membership.organization_id == organization_id,
                sql_db_tables.organization_membership.user_name == user_auth.username,
                sql_db_tables.organization_membership.invite_still_open == False
            ))
        ).first()
        assert not membership_new is None, "User not authorized"
        assert membership_new.role in ["owner", "admin"], \
            "User not authorized, must be owner or admin to add an organization collection."  

        
        collection.author_organization = organization_new.id
        collection.author_user_name = None
        print("Third switch met condition `org`")
        
        if not public:
            new_encryption_key = random_hash(base=16)
            new_encryption_key_secured = encryption.ecc_encrypt_string(organization_new.public_key, new_encryption_key)
            doc_password_secured = encryption.aes_encrypt_string(new_encryption_key, document_password_unencrypted)
            
            collection.encryption_key_secure = new_encryption_key_secured
            collection.document_unlock_key = doc_password_secured
            print("Third switch met condition `org - private`")
    
    collection.collection_type = new_collection_type
    collection.public = public
    
    print("Collection ownership changed:", type(collection), collection)
    
    database.commit()
    
    return_dict = {k: v for k, v in collection.model_dump().items() if not k in ["document_unlock_key"]}
    
    return return_dict
        
    
