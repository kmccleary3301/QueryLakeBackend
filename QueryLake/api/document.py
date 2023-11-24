from hashlib import sha256
from typing import List
import os
from fastapi import UploadFile
from ..database import sql_db_tables
from sqlmodel import Session, select, and_
import time
from ..database import encryption
from io import BytesIO
from .user_auth import *
from .hashing import *
from .api import user_db_path
from threading import Thread
from ..vector_database.embeddings import query_database, create_embeddings_in_database
from chromadb.api import ClientAPI
import time
import json
import py7zr
from fastapi.responses import StreamingResponse

server_dir = "/".join(os.path.dirname(os.path.realpath(__file__)).split("/")[:-2])
user_db_path = server_dir+"/user_db/files/"

# def create_document_collection(database : Session, username : str, collection_name : str, public : bool = False):
#     new_collection = sql_db.user_document_collection(
#         name=collection_name,
#         author_user_name=username,
#         creation_timestamp=time.time()
#     )
#     database.add(new_collection)
#     database.commit()

def upload_document(database : Session, 
                    vector_database : ClientAPI,
                    username : str, 
                    password_prehash : str, 
                    file : UploadFile, 
                    collection_hash_id : str, 
                    collection_type : str = "user",
                    organization_hash_id : str = None,
                    public : bool = False) -> dict:
    """
    Upload file to server. Possibly with encryption.
    Can be a user document, organization document, or global document, or a toolchain_session document.
    In the very last case, provide the toolchain session hash id as the collection hash id.
    """
    time_start = time.time()
    
    print("Adding document to collection")
    collection_type_lookup = {
        "user": "user_document_collection_hash_id", 
        "organization": "organization_document_collection_hash_id",
        "global": "global_document_collection_hash_id",
        "toolchain_session": "toolchain_session_hash_id" 
    }

    assert collection_type in ["global", "organization", "user", "toolchain_session"]

    if collection_type == "global":
        public = True

    user = get_user(database, username, password_prehash)

    
    password_salt = user.password_salt
    password_hash_truth = user.password_hash
    password_hash = hash_function(password_prehash, password_salt, only_salt=True)
    if (password_hash != password_hash_truth):
        return {"file_upload_success": False, "note": "Incorrect Key"}
    # file_id = hash_function(file.filename+" "+str(time.time()))

    file_zip_save_path = user_db_path+random_hash()+".7z"
    file.file.seek(0)
    file_data_bytes = file.file.read()
    file_integrity = sha256(file_data_bytes).hexdigest()
    collection_author_kwargs = {collection_type_lookup[collection_type]: collection_hash_id}

    if collection_type == "user":
        collection = database.exec(select(sql_db_tables.user_document_collection).where(sql_db_tables.user_document_collection.hash_id == collection_hash_id)).first()
        assert collection.author_user_name == username, "User not authorized"
    elif collection_type == "organization":
        collection = database.exec(select(sql_db_tables.organization_document_collection).where(sql_db_tables.organization_document_collection.hash_id == collection_hash_id)).first()
        organization = database.exec(select(sql_db_tables.organization).where(sql_db_tables.organization.id == collection.author_organization_id)).first()
        memberships = database.exec(select(sql_db_tables.organization_membership).where(and_(sql_db_tables.organization_membership.organization_id == organization.id,
                                                                                    sql_db_tables.organization_membership.user_name == username))).all()
        assert len(memberships) > 0 and memberships[0].role in ["owner", "admin", "member"], "User not authorized"
    elif collection_type == "global":
        collection = database.exec(select(sql_db_tables.global_document_collection).where(sql_db_tables.global_document_collection.hash_id == collection_hash_id)).first()
        assert user.is_admin == True, "User not authorized"
    elif collection_type == "toolchain_session":
        session = database.exec(select(sql_db_tables.toolchain_session).where(sql_db_tables.toolchain_session.hash_id == collection)).first()
        assert type(session) is sql_db_tables.toolchain_session, "Session not found"

    if not public:
        if collection_type == "organization":
            public_key = organization.public_key
        else:
            public_key = user.public_key

    encryption_key = random_hash()

    new_db_file = sql_db_tables.document_raw(
        hash_id=random_hash(),
        server_zip_archive_path=file_zip_save_path,
        file_name=file.filename,
        integrity_sha256=file_integrity,
        size_bytes=len(file_data_bytes),
        creation_timestamp=time.time(),
        public=public,
        encryption_key_secure=encryption.ecc_encrypt_string(public_key, encryption_key),
        **collection_author_kwargs
    )
    
    assert not collection is None, "Collection not found"

    zip_start_time = time.time()
    if public:
        zip_thread = Thread(target=encryption.aes_encrypt_zip_file, kwargs={
            "key": None, 
            "file_data": {file.filename: BytesIO(file_data_bytes)}, 
            "save_path": file_zip_save_path
        })
    else:
        zip_thread = Thread(target=encryption.aes_encrypt_zip_file, kwargs={
            "key": encryption_key, 
            "file_data": {file.filename: BytesIO(file_data_bytes)}, 
            "save_path": file_zip_save_path
        })
    zip_thread.start()
    print("Saved file in %.2fs" % (time.time()-zip_start_time))


    collection.document_count += 1
    database.add(new_db_file)
    database.commit()
    if (file.filename.split(".")[-1].lower() == "pdf"):
        thread = Thread(target=create_embeddings_in_database, args=(database, file_data_bytes, new_db_file, vector_database, file.filename))
        thread.start()
    time_taken = time.time() - time_start

    print("Took %.2fs to upload" % (time_taken))
    return {"success": True, "note": f"file '{file.filename}' saved at '{file_zip_save_path}'"}
    
def delete_document(database : Session, 
                    username : str, 
                    password_prehash : str,
                    hash_id: str):
    """
    Authorizes that user has permission to delete document, then does so.
    """
    user = get_user(database, username, password_prehash)

    document = database.exec(select(sql_db_tables.document_raw).where(sql_db_tables.document_raw.hash_id == hash_id)).first()

    if not document.user_document_collection_hash_id is None:
        collection = database.exec(select(sql_db_tables.user_document_collection).where(sql_db_tables.user_document_collection.hash_id == document.user_document_collection_hash_id)).first()
        assert collection.author_user_name == username, "User not authorized"
    elif not document.organization_document_collection_hash_id is None:
        collection = database.exec(select(sql_db_tables.organization_document_collection).where(sql_db_tables.organization_document_collection.hash_id == document.organization_document_collection_hash_id)).first()
        organization = database.exec(select(sql_db_tables.organization).where(sql_db_tables.organization.id == collection.author_organization_id)).first()

        memberships = database.exec(select(sql_db_tables.organization_membership).where(and_(sql_db_tables.organization_membership.organization_id == organization.id,
                                                                                    sql_db_tables.organization_membership.user_name == username))).all()
        
        assert len(memberships) > 0 and memberships[0].role in ["owner", "admin", "member"], "User not authorized"
    
    os.remove(document.server_zip_archive_path)

    database.delete(document)

    collection.document_count -= 1
    database.commit()
    
    return {"success": True}

def get_document_secure(database : Session, 
                        username : str, 
                        password_prehash : str,
                        hash_id: str,
                        return_document : bool = False):
    """
    Returns the document entry withing the system database.
    Primarily used for internal calls.
    """

    user = get_user(database, username, password_prehash)

    document = database.exec(select(sql_db_tables.document_raw).where(sql_db_tables.document_raw.hash_id == hash_id)).first()

    if not document.user_document_collection_hash_id is None:
        collection = database.exec(select(sql_db_tables.user_document_collection).where(sql_db_tables.user_document_collection.hash_id == document.user_document_collection_hash_id)).first()
        assert collection.author_user_name == username, "User not authorized"
        private_key_encryption_salt = user.private_key_encryption_salt
        user_private_key_decryption_key = hash_function(password_prehash, private_key_encryption_salt, only_salt=True)

        user_private_key = encryption.aes_decrypt_string(user_private_key_decryption_key, user.private_key_secured)

        document_password = encryption.ecc_decrypt_string(user_private_key, document.encryption_key_secure)
    elif not document.organization_document_collection_hash_id is None:
        collection = database.exec(select(sql_db_tables.organization_document_collection).where(sql_db_tables.organization_document_collection.hash_id == document.organization_document_collection_hash_id)).first()
        organization = database.exec(select(sql_db_tables.organization).where(sql_db_tables.organization.id == collection.author_organization_id)).first()

        memberships = database.exec(select(sql_db_tables.organization_membership).where(and_(sql_db_tables.organization_membership.organization_id == organization.id,
                                                                                    sql_db_tables.organization_membership.user_name == username))).all()
        assert len(memberships) > 0, "User not authorized"

        private_key_encryption_salt = user.private_key_encryption_salt
        user_private_key_decryption_key = hash_function(password_prehash, private_key_encryption_salt, only_salt=True)

        user_private_key = encryption.ecc_decrypt_string(user_private_key_decryption_key, user.private_key_secured)

        organization_private_key = encryption.ecc_decrypt_string(user_private_key, memberships[0].organization_private_key_secure)

        document_password = encryption.ecc_decrypt_string(organization_private_key, document.encryption_key_secure)
    if return_document:
        return document
    return {"password": document_password, "database_path": document.server_zip_archive_path}

def query_vector_db(database : Session, 
                    vector_database : ClientAPI, 
                    username : str, 
                    password_prehash : str,
                    query: str,
                    collection_hash_ids: List[str],
                    k : int = 10,
                    use_rerank : bool = False):
    """
    Query from the vector database.
    """
    user = get_user(database, username, password_prehash)
    return {"success": True, "results": query_database(vector_database, query, collection_hash_ids, k=k, use_rerank=use_rerank)}

def craft_document_access_token(database : Session, 
                                public_key: str,
                                username : str, 
                                password_prehash : str,
                                hash_id: str,
                                validity_window: float = 60): # In seconds
    """
    Craft a document access token using the global server public key.
    Default expiration is 60 seconds, but client can specify otherwise.
    """
    document = get_document_secure(database, username, password_prehash, hash_id, return_document=True)
    token_hash = random_hash()
    
    new_document_access_token = sql_db_tables.document_access_token(
        hash_id=token_hash,
        expiration_timestamp=time.time()+validity_window
    ) 

    database.add(new_document_access_token)
    database.commit()

    access_encrypted = encryption.ecc_encrypt_string(public_key, json.dumps({
        "username": username,
        "password_prehash": password_prehash,
        "document_hash_id": hash_id,
        "token_hash": token_hash
    }))
    return {"success": True, "result": [document.file_name, access_encrypted]}
    

def fetch_document(database : Session,
                   auth_access : str,
                   server_private_key : str):
    """
    Decrypt document in memory for the user's viewing.
    """
    try:
        auth_access = json.loads(encryption.ecc_decrypt_string(server_private_key, auth_access))

        auth_access["hash_id"] = auth_access["document_hash_id"]

        document_access_token =  database.exec(select(sql_db_tables.document_access_token).where(sql_db_tables.document_access_token.hash_id == auth_access["token_hash"])).first()
        assert document_access_token.expiration_timestamp > time.time(), "Document Access Token Expired"
        

        fetch_parameters = get_document_secure(**{
            "database" : database, 
            "username" : auth_access["username"], 
            "password_prehash" : auth_access["password_prehash"],
            "hash_id": auth_access["hash_id"],
        })
        path=fetch_parameters["database_path"]
        password = fetch_parameters["password"]

        def yield_single_file():
            with py7zr.SevenZipFile(path, mode='r', password=password) as z:
                file = z.read()
                keys = list(file.keys())
                print(keys)
                file_name = keys[0]
                file = file[file_name]
                yield file.getbuffer().tobytes()
        return StreamingResponse(yield_single_file())
    except Exception as e:
        return {"success": False, "note": str(e)}
