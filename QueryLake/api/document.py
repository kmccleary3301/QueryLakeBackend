from hashlib import sha256
from typing import List, Callable, Awaitable, Dict, Any, Union
import os, sys
import re
from fastapi import UploadFile
from ..database import sql_db_tables
from sqlmodel import Session, select, and_, delete
import time
from ..database import encryption
from io import BytesIO
from .user_auth import *
from .hashing import *
from threading import Thread
from ..vector_database.embeddings import chunk_documents
from ..vector_database.document_parsing import parse_PDFs
from ..database.encryption import aes_decrypt_zip_file, aes_delete_file_from_zip_blob
# from chromadb.api import ClientAPI
import time
import json
import py7zr, zipfile
from fastapi.responses import StreamingResponse
import ocrmypdf
from ..typing.config import AuthType
import contextlib
import io
from ..misc_functions.function_run_clean import file_size_as_string
import asyncio
import bisect
import concurrent.futures

async def upload_document(database : Session,
                          toolchain_function_caller: Callable[[Any], Union[Callable, Awaitable[Callable]]],
                          auth : AuthType, 
                          file : Union[UploadFile, BytesIO], 
                          collection_hash_id : str, 
                          file_name : str = None,
                          collection_type : str = "user",
                          return_file_hash : bool = False,
                          create_embeddings : bool = True,
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
    if isinstance(file, BytesIO):
        assert hasattr(file, 'name') or (not file_name is None), "upload_document recieved a BytesIO object without a name attribute"
        file_data_bytes_io : BytesIO = file
        file_name = file_data_bytes_io.name if file_name is None else file_name
        file_data_bytes = file_data_bytes_io.getvalue()
    else:
        file_name = file.filename
        file.file.seek(0)
        file_data_bytes = file.file.read()
        file_data_bytes_io = BytesIO(file_data_bytes)
    
    file_integrity = sha256(file_data_bytes).hexdigest()
    file_size = len(file_data_bytes)
    
    collection_author_kwargs = {collection_type_lookup[collection_type]: collection_hash_id}

    if collection_type == "user":
        collection = database.exec(select(sql_db_tables.user_document_collection).where(sql_db_tables.user_document_collection.id == collection_hash_id)).first()
        assert collection.author_user_name == user_auth.username, "User not authorized"
        public = collection.public
    elif collection_type == "organization":
        collection = database.exec(select(sql_db_tables.organization_document_collection).where(sql_db_tables.organization_document_collection.id == collection_hash_id)).first()
        organization = database.exec(select(sql_db_tables.organization).where(sql_db_tables.organization.id == collection.author_organization_id)).first()
        memberships = database.exec(select(sql_db_tables.organization_membership).where(and_(sql_db_tables.organization_membership.organization_id == organization.id,
                                                                                    sql_db_tables.organization_membership.user_name == user_auth.username))).all()
        assert len(memberships) > 0 and memberships[0].role in ["owner", "admin", "member"], "User not authorized"
        public = collection.public
    elif collection_type == "global":
        collection = database.exec(select(sql_db_tables.global_document_collection).where(sql_db_tables.global_document_collection.id == collection_hash_id)).first()
        assert user.is_admin == True, "User not authorized"
        public = True
    elif collection_type == "toolchain_session":
        session = database.exec(select(sql_db_tables.toolchain_session).where(sql_db_tables.toolchain_session.id == collection_hash_id)).first()
        assert type(session) is sql_db_tables.toolchain_session, "Session not found"
        collection = database.exec(select(sql_db_tables.toolchain_session).where(sql_db_tables.toolchain_session.id == collection_hash_id)).first()
        public = False
    
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
            file_data=file_data_bytes_io
        )
        # zip_thread = Thread(target=encryption.aes_encrypt_zip_file, kwargs={
        #     "key": None, 
        #     "file_data": {file.filename: BytesIO(file_data_bytes)}, 
        #     "save_path": file_zip_save_path
        # })
    else:
        encrypted_bytes = encryption.aes_encrypt_zip_file(
            key=encryption_key,
            file_data=file_data_bytes_io
        )
        # zip_thread = Thread(target=encryption.aes_encrypt_zip_file, kwargs={
        #     "key": encryption_key, 
        #     "file_data": {file.filename: BytesIO(file_data_bytes)}, 
        #     "save_path": file_zip_save_path
        # })
    # zip_thread.start()
    
    new_file_blob = sql_db_tables.document_zip_blob(
        file_count=1,
        size_bytes=len(encrypted_bytes),
        encryption_key_secure=encryption.ecc_encrypt_string(public_key, encryption_key),
        file_data=encrypted_bytes,
        **collection_author_kwargs,
    )
    
    database.add(new_file_blob)
    database.commit()
    
    new_db_file = sql_db_tables.document_raw(
        # server_zip_archive_path=file_zip_save_path,
        file_name=file_name,
        integrity_sha256=file_integrity,
        size_bytes=file_size,
        creation_timestamp=time.time(),
        public=public,
        encryption_key_secure=encryption.ecc_encrypt_string(public_key, encryption_key),
        # file_data=encrypted_bytes,
        blob_id=new_file_blob.id,
        blob_dir="file",
        md={"file_name": file_name, "integrity_sha256": file_integrity, "size_bytes": len(file_data_bytes)},
        **collection_author_kwargs
    )
    
    assert not collection is None or collection_type, "Collection not found"
    
    print("Saved file in %.2fs" % (time.time()-zip_start_time))
    print("Collection type:", collection_type)
    if not collection_type == "toolchain_session":
        collection.document_count += 1
    database.add(new_db_file)
    database.commit()
    if await_embedding:
        await chunk_documents(toolchain_function_caller,
                                     database,
                                     auth, 
                                     [file_data_bytes], 
                                     [new_db_file.id], 
                                     [file_name],
                                     create_embeddings=create_embeddings)
    else:
        asyncio.create_task(chunk_documents(toolchain_function_caller,
                                                   database,
                                                   auth, 
                                                   [file_data_bytes], 
                                                   [new_db_file.id], 
                                                   [file_name],
                                                   create_embeddings=create_embeddings))
    
    time_taken = time.time() - time_start

    
    database.refresh(new_db_file)
    
    print("Took %.2fs to upload" % (time_taken))
    if return_file_hash:
        return {"hash_id": new_db_file.id, "file_name": file_name, "finished_processing": new_db_file.finished_processing}

    return {
        "hash_id": new_db_file.id, 
        "title": file_name, 
        "size": file_size_as_string(len(file_data_bytes)), 
        "finished_processing": new_db_file.finished_processing
    }

async def upload_archive(database : Session,
                         toolchain_function_caller: Callable[[Any], Union[Callable, Awaitable[Callable]]],
                         auth : AuthType, 
                         file : UploadFile, 
                         collection_hash_id : str, 
                         collection_type : str = "user",
                         return_file_hash : bool = False,
                         create_embeddings : bool = True,
                         await_embedding : bool = False):
    """
    The batch version of upload_document.
    The user uploads a zip archive of all files.
    The files must all be in the top level directory of the zip archive.
    """
    
    
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

    collection_author_kwargs = {collection_type_lookup[collection_type]: collection_hash_id}

    if collection_type == "user":
        collection = database.exec(select(sql_db_tables.user_document_collection).where(sql_db_tables.user_document_collection.id == collection_hash_id)).first()
        assert collection.author_user_name == user_auth.username, "User not authorized"
        public = collection.public
    elif collection_type == "organization":
        collection = database.exec(select(sql_db_tables.organization_document_collection).where(sql_db_tables.organization_document_collection.id == collection_hash_id)).first()
        organization = database.exec(select(sql_db_tables.organization).where(sql_db_tables.organization.id == collection.author_organization_id)).first()
        memberships = database.exec(select(sql_db_tables.organization_membership).where(and_(sql_db_tables.organization_membership.organization_id == organization.id,
                                                                                    sql_db_tables.organization_membership.user_name == user_auth.username))).all()
        assert len(memberships) > 0 and memberships[0].role in ["owner", "admin", "member"], "User not authorized"
        public = collection.public
    elif collection_type == "global":
        collection = database.exec(select(sql_db_tables.global_document_collection).where(sql_db_tables.global_document_collection.id == collection_hash_id)).first()
        assert user.is_admin == True, "User not authorized"
        public = True
    elif collection_type == "toolchain_session":
        session = database.exec(select(sql_db_tables.toolchain_session).where(sql_db_tables.toolchain_session.id == collection_hash_id)).first()
        assert type(session) is sql_db_tables.toolchain_session, "Session not found"
        collection = database.exec(select(sql_db_tables.toolchain_session).where(sql_db_tables.toolchain_session.id == collection_hash_id)).first()
        public = False
    
    if not public:
        if collection_type == "organization":
            public_key = organization.public_key
        else:
            public_key = user.public_key

    encryption_key = random_hash()
    
    
    file_name = file.filename
    file_ext = file_name.split(".")[-1]
    file.file.seek(0)
    file_data_bytes = file.file.read()
    
    file_bytes_io = BytesIO(file_data_bytes)
    files_retrieved : List[BytesIO] = []
    files_retrieved_bytes : List[bytes] = []
    files_retrieved_names, coroutines = [], []
    
    
    
    if file_ext == "7z":
        with py7zr.SevenZipFile(file_bytes_io, mode='r') as z:

            with concurrent.futures.ThreadPoolExecutor() as executor:
                # future = executor.submit(z.read)
                future = executor.submit(z.getnames)
                try:
                    file_list = future.result(timeout=10)
                    # print("Extracted file names:", file_list)
                except concurrent.futures.TimeoutError:
                    raise concurrent.futures.TimeoutError("Reading the archive files took more than 10 seconds. This is usually a sign that the 7z file has too many entries.")
            
            assert len(file_list) <= 10000, "7z archive must contain <= 10,000 files"
            
            top_level_files, top_level_folders, folder_indices = [], [], []
            for entry in file_list:
                if '/' in entry:
                    top_level_folders.append(entry.split('/')[0])
                else:
                    bisect.insort(top_level_files, entry)

            for folder in top_level_folders:
                index = bisect.bisect_left(top_level_files, folder)
                if index < len(top_level_files) and top_level_files[index] == folder:
                    bisect.insort(folder_indices, index)
            
            for index in folder_indices[::-1]:
                del top_level_files[index]
            
            for name, file_entry_bytes_io in z.read(top_level_files).items():
                files_retrieved.append(file_entry_bytes_io)
                files_retrieved_bytes.append(file_entry_bytes_io.getvalue())
                files_retrieved_names.append(name)
    
    elif file_ext == "zip":
        with zipfile.ZipFile(file_bytes_io, 'r') as z:
            zip_file_list = z.namelist()
            
            for name in zip_file_list:
                if not name.endswith('/'):
                    with z.open(name, "r") as file:
                        file_entry_bytes = file.read()
                        file_entry_bytes_io = BytesIO(file_entry_bytes)
                        files_retrieved_bytes.append(file_entry_bytes)
                        files_retrieved.append(file_entry_bytes_io)
                        files_retrieved_names.append(name)
                        file.close()
                        
            z.close()
    else:
        raise ValueError(f"File extension `{file_ext}` not supported for archival, only .7z and .zip")
    
    if len(files_retrieved) == 0:
        return []
    
    gen_directories = list(map(lambda x: random_hash(), files_retrieved_names))
    file_blob_dict = {gen_directories[i]: files_retrieved[i] for i in range(len(files_retrieved))}
    
    if public:
        encrypted_bytes = encryption.aes_encrypt_zip_file_dict(
            key=None, 
            file_data=file_blob_dict
        )
    else:
        encrypted_bytes = encryption.aes_encrypt_zip_file_dict(
            key=encryption_key, 
            file_data=file_blob_dict
        )
    
    new_file_blob = sql_db_tables.document_zip_blob(
        file_count=len(gen_directories),
        size_bytes=len(encrypted_bytes),
        encryption_key_secure=encryption.ecc_encrypt_string(public_key, encryption_key),
        file_data=encrypted_bytes,
        **collection_author_kwargs,
    )
    
    database.add(new_file_blob)
    database.commit()
    
    results, new_doc_ids, new_doc_db_entries = [], [], []
    
    for i in range(len(files_retrieved)):
        file_integrity = sha256(files_retrieved_bytes[i]).hexdigest()
        file_size = len(files_retrieved_bytes[i])
        file_name = files_retrieved_names[i]
        new_db_file = sql_db_tables.document_raw(
            # server_zip_archive_path=file_zip_save_path,
            file_name=file_name,
            integrity_sha256=file_integrity,
            size_bytes=file_size,
            creation_timestamp=time.time(),
            public=public,
            encryption_key_secure=encryption.ecc_encrypt_string(public_key, encryption_key),
            # file_data=encrypted_bytes,
            blob_id=new_file_blob.id,
            blob_dir=gen_directories[i],
            md={"file_name": file_name, "integrity_sha256": file_integrity, "size_bytes": file_size},
            **collection_author_kwargs
        )
        database.add(new_db_file)
        results.append({
            "hash_id": new_db_file.id, 
            "title": file_name, 
            "size": file_size_as_string(file_size), 
            "finished_processing": new_db_file.finished_processing
        })
        new_doc_ids.append(new_db_file.id)
        new_doc_db_entries.append(new_db_file)
    
    collection.document_count += len(new_doc_ids)
    
    database.commit()
    
    if await_embedding:
        await chunk_documents(toolchain_function_caller,
                              database,
                              auth, 
                              files_retrieved_bytes, 
                              new_doc_db_entries, 
                              files_retrieved_names,
                              create_embeddings=create_embeddings)
    else:
        asyncio.create_task(chunk_documents(toolchain_function_caller,
                                            database,
                                            auth,
                                            files_retrieved_bytes, 
                                            new_doc_db_entries, 
                                            files_retrieved_names,
                                            create_embeddings=create_embeddings))
    
    # results = await asyncio.gather(*coroutines)
    return results

def delete_document(database : Session, 
                    auth : AuthType,
                    hash_id: Union[List[str], str] = None,
                    hash_ids: Union[List[str], str] = None):
    """
    Authorizes that user has permission to delete document, then does so.
    """
    (user, user_auth) = get_user(database, auth)

    assert not hash_id is None or not hash_ids is None, "No hash_id or hash_ids provided"
    
    document = database.exec(select(sql_db_tables.document_raw).where(sql_db_tables.document_raw.id == hash_id)).first()

    assert not document is None, "Document not found"
    
    if not document.user_document_collection_hash_id is None:
        collection = database.exec(select(sql_db_tables.user_document_collection).where(sql_db_tables.user_document_collection.id == document.user_document_collection_hash_id)).first()
        assert collection.author_user_name == user_auth.username, "User not authorized"
    
    elif not document.organization_document_collection_hash_id is None:
        collection = database.exec(select(sql_db_tables.organization_document_collection).where(sql_db_tables.organization_document_collection.id == document.organization_document_collection_hash_id)).first()
        organization = database.exec(select(sql_db_tables.organization).where(sql_db_tables.organization.id == collection.author_organization_id)).first()

        memberships = database.exec(select(sql_db_tables.organization_membership).where(and_(sql_db_tables.organization_membership.organization_id == organization.id,
                                                                                    sql_db_tables.organization_membership.user_name == user_auth.username))).all()
        
        assert len(memberships) > 0 and memberships[0].role in ["owner", "admin", "member"], "User not authorized"

    database.exec(delete(sql_db_tables.DocumentChunk).where(sql_db_tables.DocumentChunk.document_id == hash_id))
    
    aes_delete_file_from_zip_blob(database, document.id)
    
    database.delete(document)

    collection.document_count -= 1
    database.commit()
    
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
    
    assert not document is None, "Document not found"
    
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

    print("Document Request Access Auth:", document_auth_access)
    
    document_access_token =  database.exec(select(sql_db_tables.document_access_token).where(sql_db_tables.document_access_token.hash_id == document_auth_access["token_hash"])).first()
    assert document_access_token.expiration_timestamp > time.time(), "Document Access Token Expired"

    fetch_parameters = get_document_secure(**{
        "database" : database, 
        "auth" : document_auth_access["auth"],
        "hash_id": document_auth_access["hash_id"],
    })
    
    password = fetch_parameters["password"]

    file_io = aes_decrypt_zip_file(
        database, 
        password,
        fetch_parameters["hash_id"]
    )
    
    print("fetch_document got", type(file_io), "with size", len(file_io.getvalue()))
    
    return StreamingResponse(file_io)


async def fetch_toolchain_document(database : Session,
                                   auth: AuthType,
                                   document_hash_id: str):
    """
    Decrypt document in memory for the user's viewing.
    Return as a streaming response of bytes.
    """
    # print("Fetching document with auth:", document_auth_access)
    
    document_auth_access = {}
    # print(json.dumps(document_auth_access, indent=4))

    document_auth_access["hash_id"] = document_auth_access["document_hash_id"]

    print("Document Request Access Auth:", document_auth_access)
    
    document_access_token =  database.exec(select(sql_db_tables.document_access_token).where(sql_db_tables.document_access_token.hash_id == document_auth_access["token_hash"])).first()
    assert document_access_token.expiration_timestamp > time.time(), "Document Access Token Expired"

    fetch_parameters = get_document_secure(**{
        "database" : database, 
        "auth" : document_auth_access["auth"],
        "hash_id": document_auth_access["hash_id"],
    })
    
    password = fetch_parameters["password"]

    file_io = aes_decrypt_zip_file(
        database, 
        password,
        fetch_parameters["hash_id"]
    )
    
    print("fetch_document got", type(file_io), "with size", len(file_io.getvalue()))
    
    return StreamingResponse(file_io)

def ocr_pdf_file(database : Session,
                 auth : AuthType,
                 file: Union[bytes, BytesIO]):
    """
    TODO: OCR capabilities should be in a deployment class.
    
    OCR a pdf file and return the raw text.
    """
    
    (user, user_auth) = get_user(database, auth)
    ocr_bytes_target = BytesIO()
    ocr_bytes_target.seek(0)
    
    assert isinstance(file, (bytes, BytesIO)), "File must be bytes or BytesIO object"
    
    if isinstance(file, bytes):
        file_input : BytesIO = BytesIO(file)
        file_input.seek(0)
    elif isinstance(file, BytesIO):
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


