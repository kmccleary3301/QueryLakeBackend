import os
from ..database import sql_db_tables
from sqlmodel import Session, select, and_
import time
from .user_auth import *
from .hashing import random_hash
from ..typing.config import AuthType
from ..misc_functions.function_run_clean import file_size_as_string

server_dir = "/".join(os.path.dirname(os.path.realpath(__file__)).split("/")[:-2])
# user_db_path = server_dir+"/user_db/files/"

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
        collections_get = database.exec(select(sql_db_tables.global_document_collection)).all()
    if not organization_id is None:
        membership_get = database.exec(select(sql_db_tables.organization_membership).where(
            sql_db_tables.organization_membership.user_name == user_auth.username and \
            sql_db_tables.organization_membership.organization_id == organization_id)).all()
        assert len(membership_get) > 0, "User not in organization"
        collections_get = database.exec(select(sql_db_tables.organization_document_collection).where(sql_db_tables.organization_document_collection.author_organization_id == organization_id)).all()
    else:
        collections_get = database.exec(select(sql_db_tables.user_document_collection).where(sql_db_tables.user_document_collection.author_user_name == user_auth.username)).all()
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

    (user, user_auth) = get_user(database, auth)
    if not organization_id is None:
        membership_get = database.exec(select(sql_db_tables.organization_membership).where(and_(
            sql_db_tables.organization_membership.user_name == user_auth.username,
            sql_db_tables.organization_membership.organization_id == organization_id))).all()
        assert len(membership_get) > 0, "User not in organization"
        assert membership_get[0].role != "viewer", "Invalid Permissions"

        new_collection = sql_db_tables.organization_document_collection(
            name=name,
            hash_id=hash_id,
            author_organization_id=organization_id,
            creation_timestamp=time.time(),
            public=public,
            description=description
        )
    else:
        new_collection = sql_db_tables.user_document_collection(
            name=name,
            hash_id=hash_id,
            author_user_name=user_auth.username,
            creation_timestamp=time.time(),
            public=public,
            description=description
        )
    database.add(new_collection)
    database.commit()
    database.flush()
    # return {"success": True, "hash_id": new_collection.hash_id}
    return {"hash_id": new_collection.hash_id}

def fetch_all_collections(database : Session, 
                          auth : AuthType):
    """
    Fetches all collections that a user has priviledge to read.
    """
    (user, user_auth) = get_user(database, auth)

    fetch_memberships = database.exec(select(sql_db_tables.organization_membership).where(
            sql_db_tables.organization_membership.user_name == user_auth.username and sql_db_tables.organization_membership.invite_still_open == False)).all()
    organizations = []
    for membership in fetch_memberships:
        if not membership.invite_still_open:
            organizations.append(database.exec(select(sql_db_tables.organization).where(sql_db_tables.organization.id == membership.organization_id)).first())
    
    return_value = {"global_collections": [], "user_collections": []}

    organization_collections = {}
    for organization in organizations:
        collections = database.exec(select(sql_db_tables.organization_document_collection).where(sql_db_tables.organization_document_collection.author_organization_id == organization.id)).all()
        organization_collections[organization.id] = {"name": organization.name, "collections": []}
        for collection in collections:
            organization_collections[organization.id]["collections"].append({
                "hash_id": collection.hash_id,
                "name": collection.name,
                "document_count": collection.document_count,
                "type": "organization",
            })
    return_value["organization_collections"] = organization_collections
    global_collections = database.exec(select(sql_db_tables.global_document_collection)).all()
    for collection in global_collections:
        return_value["global_collections"].append({
            "name": collection.name,
            "hash_id": collection.hash_id,
            "document_count": collection.document_count,
            "type": "global"
        })
    user_collections = database.exec(select(sql_db_tables.user_document_collection).where(sql_db_tables.user_document_collection.author_user_name == user_auth.username)).all()
    for collection in user_collections:
        return_value["user_collections"].append({
            "name": collection.name,
            "hash_id": collection.hash_id,
            "document_count": collection.document_count,
            "type" : "user"
        })
    # return {"success": True, "result": return_value}
    return {"collections": return_value}

def fetch_collection(database : Session,
                     auth : AuthType,
                     collection_hash_id : str,
                     collection_type : str = "user"):
    """
    Retrieves details of a collection for user, 
    including all documents in the collection.
    """

    assert collection_type in ["user", "organization", "global"], "Invalid collection type"
    (user, user_auth) = get_user(database, auth)
    if collection_type == "user":
        collection = database.exec(select(sql_db_tables.user_document_collection).where(and_(sql_db_tables.user_document_collection.hash_id == collection_hash_id))).first()
        if collection.public == False:
            assert collection.author_user_name == user_auth.username, "User not authorized to view collection"
        documents = database.exec(select(sql_db_tables.document_raw).where(sql_db_tables.document_raw.user_document_collection_hash_id == collection_hash_id)).all()
        owner = "personal"
    else:
        collection = database.exec(select(sql_db_tables.organization_document_collection).where(sql_db_tables.organization_document_collection.hash_id == collection_hash_id)).first()
        user_membership = database.exec(select(sql_db_tables.organization_membership).where(and_(sql_db_tables.organization_membership.organization_id == collection.author_organization_id,
                                                                                          sql_db_tables.organization_membership.user_name == user_auth.username))).all()
        assert len(user_membership) > 0 or collection.public == True, "User not authorized to view collection"
        documents = database.exec(select(sql_db_tables.document_raw).where(sql_db_tables.document_raw.organization_document_collection_hash_id == collection_hash_id)).all()

        author_org = database.exec(select(sql_db_tables.organization).where(sql_db_tables.organization.id == collection.author_organization_id)).first()
        owner = author_org.name()

    data = {
        "title" : collection.name,
        "description": collection.description,
        "type": collection_type,
        "owner": owner,
        "public": collection.public,
        "document_list": []
    }
    for document in documents:
        data["document_list"].append({
            "title": document.file_name,
            "hash_id": document.hash_id,
            "size": file_size_as_string(document.size_bytes),
            "finished_processing": document.finished_processing,
        })
    # return {"success": True, "result": data}
    return data

def modify_document_collection(database : Session,
                                auth : AuthType,
                                collection_hash_id : str,
                                title : str = None,
                                description : str = None,
                                collection_type : str = "user"):
    """
    Changes document collection properties for a user.
    
    TODO: Implement public/private switch.
    """

    assert collection_type in ["user", "organization", "global"], "Invalid collection type"
    (user, user_auth) = get_user(database, auth)

    if collection_type == "user":
        collection = database.exec(select(sql_db_tables.user_document_collection).where(and_(sql_db_tables.user_document_collection.hash_id == collection_hash_id))).first()
        if collection.public == False:
            assert collection.author_user_name == user_auth.username, "User not authorized to modify collection"
    elif collection_type == "organization":
        collection = database.exec(select(sql_db_tables.organization_document_collection).where(sql_db_tables.organization_document_collection.hash_id == collection_hash_id)).first()
        memberships = database.exec(select(sql_db_tables.organization_membership).where(and_(sql_db_tables.organization_membership.organization_id == collection.author_organization_id,
                                                                                          sql_db_tables.organization_membership.user_name == user_auth.username))).all()
        assert len(memberships) > 0 or collection.public == True, "User not authorized to modify collection"
        assert len(memberships) > 0 and memberships[0].role in ["owner", "admin", "member"], "User not authorized to modify collection"
    else:
        collection = database.exec(select(sql_db_tables.global_document_collection).where(sql_db_tables.global_document_collection.hash_id == collection_hash_id)).first()
        assert user.is_admin == True, "User not authorized to modify collection"

    if not title is None:
        collection.name = title
    if not description is None:
        collection.description = description

    # return {"success": True}
    return True

