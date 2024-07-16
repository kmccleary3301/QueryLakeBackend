from hashlib import sha256
from typing import List, Callable, Awaitable, Dict, Any, Union
import os, sys
import re
from fastapi import UploadFile
from ..database import sql_db_tables
from sqlmodel import Session, select, and_
import time
from ..database import encryption
from io import BytesIO
from .user_auth import *
from .hashing import *
from threading import Thread
from ..vector_database.embeddings import query_database, create_embeddings_in_database
from ..vector_database.document_parsing import parse_PDFs
from ..database.encryption import aes_decrypt_zip_file
# from chromadb.api import ClientAPI
import time
import json
import py7zr
from fastapi.responses import StreamingResponse
import ocrmypdf
from ..typing.config import AuthType
import contextlib
import io
from ..misc_functions.function_run_clean import file_size_as_string
import asyncio

server_dir = "/".join(os.path.dirname(os.path.realpath(__file__)).split("/")[:-2])

async def upload_document(database : Session,
                          toolchain_function_caller: Callable[[Any], Union[Callable, Awaitable[Callable]]],
                          auth : AuthType, 
                          file : UploadFile, 
                          collection_hash_id : str, 
                          collection_type : str = "user",
                          public : bool = False,
                          return_file_hash : bool = False,
                          add_to_vector_db : bool = True,
                          await_embedding : bool = False) -> dict:
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
        "toolchain_session": "toolchain_session_id" 
    }

    assert collection_type in ["global", "organization", "user", "toolchain_session"]

    if collection_type == "global":
        public = True

    (user, user_auth) = get_user(database, auth)

    
    password_salt = user.password_salt
    password_hash_truth = user.password_hash
    password_hash = hash_function(user_auth.password_prehash, password_salt, only_salt=True)
    if (password_hash != password_hash_truth):
        return {"file_upload_success": False, "note": "Incorrect Key"}
    # file_id = hash_function(file.filename+" "+str(time.time()))

    # file_zip_save_path = user_db_path+random_hash()+".7z"
    file.file.seek(0)
    file_data_bytes = file.file.read()
    file_integrity = sha256(file_data_bytes).hexdigest()
    collection_author_kwargs = {collection_type_lookup[collection_type]: collection_hash_id}

    if collection_type == "user":
        collection = database.exec(select(sql_db_tables.user_document_collection).where(sql_db_tables.user_document_collection.id == collection_hash_id)).first()
        assert collection.author_user_name == user_auth.username, "User not authorized"
    elif collection_type == "organization":
        collection = database.exec(select(sql_db_tables.organization_document_collection).where(sql_db_tables.organization_document_collection.id == collection_hash_id)).first()
        organization = database.exec(select(sql_db_tables.organization).where(sql_db_tables.organization.id == collection.author_organization_id)).first()
        memberships = database.exec(select(sql_db_tables.organization_membership).where(and_(sql_db_tables.organization_membership.organization_id == organization.id,
                                                                                    sql_db_tables.organization_membership.user_name == user_auth.username))).all()
        assert len(memberships) > 0 and memberships[0].role in ["owner", "admin", "member"], "User not authorized"
    elif collection_type == "global":
        collection = database.exec(select(sql_db_tables.global_document_collection).where(sql_db_tables.global_document_collection.id == collection_hash_id)).first()
        assert user.is_admin == True, "User not authorized"
    
    elif collection_type == "toolchain_session":
        session = database.exec(select(sql_db_tables.toolchain_session).where(sql_db_tables.toolchain_session.id == collection_hash_id)).first()
        assert type(session) is sql_db_tables.toolchain_session, "Session not found"
        collection = database.exec(select(sql_db_tables.toolchain_session).where(sql_db_tables.toolchain_session.id == collection_hash_id)).first()
    
    if not public:
        if collection_type == "organization":
            public_key = organization.public_key
        else:
            public_key = user.public_key

    encryption_key = random_hash()
    
    zip_start_time = time.time()
    if public:
        encrypted_bytes = encryption.aes_encrypt_zip_file(
            key=None, 
            file_data=BytesIO(file_data_bytes)
        )
        # zip_thread = Thread(target=encryption.aes_encrypt_zip_file, kwargs={
        #     "key": None, 
        #     "file_data": {file.filename: BytesIO(file_data_bytes)}, 
        #     "save_path": file_zip_save_path
        # })
    else:
        encrypted_bytes = encryption.aes_encrypt_zip_file(
            key=encryption_key,
            file_data=BytesIO(file_data_bytes)
        )
        # zip_thread = Thread(target=encryption.aes_encrypt_zip_file, kwargs={
        #     "key": encryption_key, 
        #     "file_data": {file.filename: BytesIO(file_data_bytes)}, 
        #     "save_path": file_zip_save_path
        # })
    # zip_thread.start()
    
    new_db_file = sql_db_tables.document_raw(
        # server_zip_archive_path=file_zip_save_path,
        file_name=file.filename,
        integrity_sha256=file_integrity,
        size_bytes=len(file_data_bytes),
        creation_timestamp=time.time(),
        public=public,
        encryption_key_secure=encryption.ecc_encrypt_string(public_key, encryption_key),
        file_data=encrypted_bytes,
        md={"file_name": file.filename, "integrity_sha256": file_integrity, "size_bytes": len(file_data_bytes)},
        **collection_author_kwargs
    )
    
    assert not collection is None or collection_type, "Collection not found"
    
    print("Saved file in %.2fs" % (time.time()-zip_start_time))
    print("Collection type:", collection_type)
    if not collection_type == "toolchain_session":
        collection.document_count += 1
    database.add(new_db_file)
    database.commit()
    if add_to_vector_db:
        
        if await_embedding:
            await create_embeddings_in_database(
                                                toolchain_function_caller,
                                                auth, 
                                                file_data_bytes, 
                                                new_db_file.id, 
                                                file.filename)
        else:
            asyncio.create_task(create_embeddings_in_database(toolchain_function_caller,
                                                              auth, 
                                                              file_data_bytes, 
                                                              new_db_file.id, 
                                                              file.filename))
    
    time_taken = time.time() - time_start

    print("Took %.2fs to upload" % (time_taken))
    if return_file_hash:
        return {"hash_id": new_db_file.id, "file_name": file.filename, "finished_processing": new_db_file.finished_processing}

    return {"hash_id": new_db_file.id, "title": file.filename, "size": file_size_as_string(len(file_data_bytes)), "finished_processing": new_db_file.finished_processing}
    
def delete_document(database : Session, 
                    auth : AuthType,
                    hash_id: str):
    """
    Authorizes that user has permission to delete document, then does so.
    """
    (user, user_auth) = get_user(database, auth)

    document = database.exec(select(sql_db_tables.document_raw).where(sql_db_tables.document_raw.id == hash_id)).first()

    if not document.user_document_collection_hash_id is None:
        collection = database.exec(select(sql_db_tables.user_document_collection).where(sql_db_tables.user_document_collection.id == document.user_document_collection_hash_id)).first()
        assert collection.author_user_name == user_auth.username, "User not authorized"
    
    elif not document.organization_document_collection_hash_id is None:
        collection = database.exec(select(sql_db_tables.organization_document_collection).where(sql_db_tables.organization_document_collection.id == document.organization_document_collection_hash_id)).first()
        organization = database.exec(select(sql_db_tables.organization).where(sql_db_tables.organization.id == collection.author_organization_id)).first()

        memberships = database.exec(select(sql_db_tables.organization_membership).where(and_(sql_db_tables.organization_membership.organization_id == organization.id,
                                                                                    sql_db_tables.organization_membership.user_name == user_auth.username))).all()
        
        assert len(memberships) > 0 and memberships[0].role in ["owner", "admin", "member"], "User not authorized"

    document_embeddings = database.exec(select(sql_db_tables.DocumentChunk).where(sql_db_tables.DocumentChunk.document_id == hash_id)).all()
    for e in document_embeddings:
        database.delete(e)
    database.commit()
    
    
    database.delete(document)

    collection.document_count -= 1
    database.commit()
    
    # return {"success": True}
    return True

def get_document_secure(database : Session, 
                        auth: AuthType,
                        hash_id: str,
                        return_document : bool = False):
    """
    Returns the document entry withing the system database.
    Primarily used for internal calls.
    """

    (user, user_auth) = get_user(database, auth)

    document = database.exec(select(sql_db_tables.document_raw).where(sql_db_tables.document_raw.id == hash_id)).first()
    
    if not document.user_document_collection_hash_id is None:
        
        collection = database.exec(select(sql_db_tables.user_document_collection).where(sql_db_tables.user_document_collection.id == document.user_document_collection_hash_id)).first()
        assert collection.author_user_name == user_auth.username, "User not authorized"
        private_key_encryption_salt = user.private_key_encryption_salt
        user_private_key_decryption_key = hash_function(user_auth.password_prehash, private_key_encryption_salt, only_salt=True)

        user_private_key = encryption.aes_decrypt_string(user_private_key_decryption_key, user.private_key_secured)

        document_password = encryption.ecc_decrypt_string(user_private_key, document.encryption_key_secure)
    elif not document.organization_document_collection_hash_id is None:
        collection = database.exec(select(sql_db_tables.organization_document_collection).where(sql_db_tables.organization_document_collection.id == document.organization_document_collection_hash_id)).first()
        organization = database.exec(select(sql_db_tables.organization).where(sql_db_tables.organization.id == collection.author_organization_id)).first()

        memberships = database.exec(select(sql_db_tables.organization_membership).where(and_(sql_db_tables.organization_membership.organization_id == organization.id,
                                                                                    sql_db_tables.organization_membership.user_name == user_auth.username))).all()
        assert len(memberships) > 0, "User not authorized"

        private_key_encryption_salt = user.private_key_encryption_salt
        user_private_key_decryption_key = hash_function(user_auth.password_prehash, private_key_encryption_salt, only_salt=True)

        user_private_key = encryption.ecc_decrypt_string(user_private_key_decryption_key, user.private_key_secured)

        organization_private_key = encryption.ecc_decrypt_string(user_private_key, memberships[0].organization_private_key_secure)

        document_password = encryption.ecc_decrypt_string(organization_private_key, document.encryption_key_secure)
    elif not document.toolchain_session_id is None:
        private_key_encryption_salt = user.private_key_encryption_salt
        user_private_key_decryption_key = hash_function(user_auth.password_prehash, private_key_encryption_salt, only_salt=True)

        user_private_key = encryption.aes_decrypt_string(user_private_key_decryption_key, user.private_key_secured)

        document_password = encryption.ecc_decrypt_string(user_private_key, document.encryption_key_secure)
    
    if return_document:
        return document
    return {"password": document_password, "hash_id": document.id}

# async def query_vector_db(database : Session,
#                           toolchain_function_caller: Callable[[Any], Union[Callable, Awaitable[Callable]]],
#                           auth : AuthType,
#                           query: str,
#                           collection_hash_ids: List[str],
#                           k : int = 10,
#                           use_rerank : bool = False,
#                           minimum_relevance : float = 0.0):
#     """
#     Query from the vector database.
#     """
#     (_, _) = get_user(database, auth)
#     results = await query_database(database, 
#                                    auth,
#                                    toolchain_function_caller, 
#                                    query, 
#                                    collection_hash_ids, 
#                                    k=k, 
#                                    use_rerank=use_rerank,
#                                    minimum_relevance=minimum_relevance)
#     return {"result": results}

def craft_document_access_token(database : Session, 
                                public_key: str,
                                auth : AuthType,
                                hash_id: str,
                                validity_window: float = 60): # In seconds
    """
    Craft a document access token using the global server public key.
    Default expiration is 60 seconds, but client can specify otherwise.
    """
    (user, user_auth) = get_user(database, auth)
    
    document : sql_db_tables.document_raw = get_document_secure(database, auth, hash_id, return_document=True)
    token_hash = random_hash()
    
    new_document_access_token = sql_db_tables.document_access_token(
        hash_id=token_hash,
        expiration_timestamp=time.time()+validity_window
    )
    
    database.add(new_document_access_token)
    database.commit()

    access_encrypted = encryption.ecc_encrypt_string(public_key, json.dumps({
        "username": user_auth.username,
        "password_prehash": user_auth.password_prehash,
        "document_hash_id": hash_id,
        "token_hash": token_hash,
        "auth": auth
    }))
    # return {"success": True, "result": [document.file_name, access_encrypted]}
    return {"file_name": document.file_name, "access_encrypted": access_encrypted}

def get_file_bytes(database : Session,
                   hash_id : str,
                   encryption_key : str):
    
    document = database.exec(select(sql_db_tables.document_raw).where(sql_db_tables.document_raw.id == hash_id)).first()
    print("Header Data:", [document.server_zip_archive_path, encryption_key])


    file = encryption.aes_decrypt_zip_file(
        database,
        encryption_key, 
        document.id
    )
    keys = list(file.keys())
    file_name = keys[0]
    file_get = file[file_name]
    return file_get
    # with py7zr.SevenZipFile(document.server_zip_archive_path, mode='r', password=encryption_key) as z:
    #     file = z.read()
    #     keys = list(file.keys())
    #     print(keys)
    #     file_name = keys[0]
    #     file_get = file[file_name]
    #     z.close()
    #     return file_get.getbuffer().tobytes()

async def fetch_document(database : Session,
                         server_private_key : str,
                         document_auth_access : str):
    """
    Decrypt document in memory for the user's viewing.
    Return as a streaming response of bytes.
    """
    # print("Fetching document with auth:", document_auth_access)
    
    document_auth_access = json.loads(encryption.ecc_decrypt_string(server_private_key, document_auth_access))
    # print(json.dumps(document_auth_access, indent=4))

    document_auth_access["hash_id"] = document_auth_access["document_hash_id"]

    document_access_token =  database.exec(select(sql_db_tables.document_access_token).where(sql_db_tables.document_access_token.hash_id == document_auth_access["token_hash"])).first()
    assert document_access_token.expiration_timestamp > time.time(), "Document Access Token Expired"
    

    fetch_parameters = get_document_secure(**{
        "database" : database, 
        "auth" : document_auth_access["auth"],
        "hash_id": document_auth_access["hash_id"],
    })
    
    password = fetch_parameters["password"]

    # def yield_single_file():
    #     with py7zr.SevenZipFile(path, mode='r', password=password) as z:
    #         file = z.read()
    #         keys = list(file.keys())
    #         print(keys)
    #         file_name = keys[0]
    #         file = file[file_name]
    #         yield file.getbuffer().tobytes()
    # return StreamingResponse(yield_single_file())
    return StreamingResponse(aes_decrypt_zip_file(
        database, 
        password,
        fetch_parameters["hash_id"]
    ))

def ocr_pdf_file(database : Session,
                 auth : AuthType,
                 file: bytes):
    """
    OCR a pdf file and return the raw text.
    """
    (user, user_auth) = get_user(database, auth)
    ocr_bytes_target = BytesIO()
    ocr_bytes_target.seek(0)

    if isinstance(file, bytes):
        print("File is bytes!")
        file_input : BytesIO = BytesIO(file)
        file_input.seek(0)
    else:
        print("File is type", type(file), "not bytes!")
        file_input = file
    
    # with contextlib.redirect_stdout(io.StringIO()):
    
    sys.stdout = open(os.devnull, 'w')
    ocrmypdf.ocr(
        file_input, 
        ocr_bytes_target, 
        force_ocr=True
    )
    sys.stdout = sys.__stdout__
    
    text = parse_PDFs(ocr_bytes_target, return_all_text_as_string=True)
    return {
        "pdf_text": text
    }


