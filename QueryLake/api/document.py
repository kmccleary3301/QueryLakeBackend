from hashlib import sha256
from typing import List, Callable, Awaitable, Dict, Any, Union, Literal, Tuple, Optional
import os, sys
import re
from fastapi import UploadFile
import ray.cloudpickle
from ..database import sql_db_tables
from sqlmodel import Session, select, and_, delete, update, func
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
from ..typing.api_inputs import DocumentModifierArgs
from ..misc_functions.function_run_clean import file_size_as_string
import asyncio
import bisect
import concurrent.futures
from .collections import assert_collections_priviledge, get_collection_document_password


async def upload_document(database : Session,
                          toolchain_function_caller: Callable[[Any], Union[Callable, Awaitable[Callable]]],
                          auth : AuthType, 
                          file : Union[UploadFile, BytesIO], 
                          collection_hash_id : str, 
                          file_name : str = None,
                          collection_type : Literal["global", "organization", "user", "toolchain_session"] = "user",
                          return_file_hash : bool = False,
                          scan_text : bool = True,
                          create_embeddings : bool = True,
                          await_embedding : bool = False,
                          document_metadata : dict = None) -> dict:
    """
    Upload file to server. Possibly with encryption.
    Can be a user document, organization document, or global document, or a toolchain_session document.
    In the very last case, provide the toolchain session hash id as the collection hash id.
    """
    time_start = time.time()
    
    print("Adding document to collection")
    

    assert isinstance(scan_text, bool), "scan_text must be a boolean"

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
    
    collection = database.exec(select(sql_db_tables.document_collection).where(sql_db_tables.document_collection.id == collection_hash_id)).first()
    assert not collection is None, "Collection not found"
    
    if collection.collection_type == "global":
        public = True
    
    if collection.collection_type == "user":
        assert collection.author_user_name == user_auth.username, "User not authorized"
        public = collection.public
    elif collection.collection_type == "organization":
        organization = database.exec(select(sql_db_tables.organization).where(sql_db_tables.organization.id == collection.author_organization)).first()
        memberships = database.exec(select(sql_db_tables.organization_membership).where(and_(sql_db_tables.organization_membership.organization_id == organization.id,
                                                                                    sql_db_tables.organization_membership.user_name == user_auth.username))).all()
        assert len(memberships) > 0 and memberships[0].role in ["owner", "admin", "member"], "User not authorized"
        public = collection.public
    elif collection.collection_type == "global":
        assert user.is_admin == True, "User not authorized"
        public = True
    elif collection.collection_type == "toolchain_session":
        session = database.exec(select(sql_db_tables.toolchain_session).where(sql_db_tables.toolchain_session.id == collection.toolchain_session_id)).first()
        assert not session is None, "Session not found"
        assert session.author == user_auth.username, "User not authorized"
        public = False
    
    if not public:
        if collection.collection_type == "organization":
            public_key = organization.public_key
        else:
            public_key = user.public_key

    # TODO: Get document key from `document_collection`
    encryption_key = get_collection_document_password(database, auth, collection.id)
    
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
        file_data=encrypted_bytes,
        document_collection_id=collection.id,
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
        # file_data=encrypted_bytes,
        finished_processing=1,
        blob_id=new_file_blob.id,
        blob_dir="file",
        md={
            "file_name": file_name, 
            "integrity_sha256": file_integrity, 
            "size_bytes": len(file_data_bytes),
            **(document_metadata if not document_metadata is None else {})
        },
        document_collection_id=collection.id
    )
    
    print("Saved file in %.2fs" % (time.time()-zip_start_time))
    print("Collection type:", collection.collection_type)
    collection.document_count += 1
    database.add(new_db_file)
    database.commit()
    if await_embedding and scan_text:
        await chunk_documents(toolchain_function_caller,
                              database,
                              auth, 
                              [new_db_file.id], 
                              [file_name],
                              document_bytes_list=[file_data_bytes], 
                              create_embeddings=create_embeddings)
    elif scan_text:
        asyncio.create_task(chunk_documents(toolchain_function_caller,
                                            database,
                                            auth, 
                                            [new_db_file.id], 
                                            [file_name],
                                            document_bytes_list=[file_data_bytes], 
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
                         scan_text : bool = True,
                         create_embeddings : bool = True,
                         await_embedding : bool = False):
    """
    The batch version of upload_document.
    The user uploads a zip archive of all files.
    The files must all be in the top level directory of the zip archive.
    """
    
    print("Extracting document archive and adding to collection")
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
    
    collection = database.exec(select(sql_db_tables.document_collection).where(sql_db_tables.document_collection.id == collection_hash_id)).first()
    assert not collection is None, "Collection not found"
    
    if collection.collection_type == "user":
        assert collection.author_user_name == user_auth.username, "User not authorized"
        public = collection.public
    elif collection_type == "organization":
        organization = database.exec(select(sql_db_tables.organization).where(sql_db_tables.organization.id == collection.author_organization)).first()
        memberships = database.exec(select(sql_db_tables.organization_membership).where(and_(sql_db_tables.organization_membership.organization_id == organization.id,
                                                                                    sql_db_tables.organization_membership.user_name == user_auth.username))).all()
        assert len(memberships) > 0 and memberships[0].role in ["owner", "admin", "member"], "User not authorized"
        public = collection.public
    elif collection_type == "global":
        assert user.is_admin == True, "User not authorized"
        public = True
    elif collection_type == "toolchain_session":
        session = database.exec(select(sql_db_tables.toolchain_session).where(sql_db_tables.toolchain_session.id == collection.toolchain_session_id)).first()
        assert not session is None, "Session not found"
        assert session.author_user_name == user_auth.username, "User not authorized"
        public = False
    
    if not public:
        if collection_type == "organization":
            public_key = organization.public_key
        else:
            public_key = user.public_key

    
    # TODO: Get document key from `document_collection`
    encryption_key = get_collection_document_password(database, auth, collection.id)
    
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
            assert len(zip_file_list) <= 10000, "zip archive must contain <= 10,000 files"
            
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
        file_data=encrypted_bytes,
        document_collection_id=collection.id,
    )
    
    database.add(new_file_blob)
    database.commit()
    
    results, new_doc_ids = [], []
    new_doc_db_entries : List[sql_db_tables.document_raw] = []
    
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
            # file_data=encrypted_bytes,
            finished_processing=1,
            blob_id=new_file_blob.id,
            blob_dir=gen_directories[i],
            md={"file_name": file_name, "integrity_sha256": file_integrity, "size_bytes": file_size},
            document_collection_id=collection.id
        )
        database.add(new_db_file)
        
        new_doc_ids.append(new_db_file.id)
        new_doc_db_entries.append(new_db_file)
    
    # Increment the doc count
    collection.document_count += len(new_doc_ids)
    # Commit the docs, before we start chunking them.
    database.commit()
    
    if await_embedding and scan_text:
        await chunk_documents(toolchain_function_caller,
                              database,
                              auth, 
                              new_doc_db_entries, 
                              files_retrieved_names,
                              document_bytes_list=files_retrieved_bytes, 
                              create_embeddings=create_embeddings)
    elif scan_text:
        asyncio.create_task(chunk_documents(toolchain_function_caller,
                                            database,
                                            auth, 
                                            new_doc_db_entries, 
                                            files_retrieved_names,
                                            document_bytes_list=files_retrieved_bytes, 
                                            create_embeddings=create_embeddings))
    
    for i, new_db_file in enumerate(new_doc_db_entries):
        database.refresh(new_db_file)
        results.append({
            "hash_id": new_db_file.id, 
            "title": files_retrieved_names[i], 
            "size": file_size_as_string(len(files_retrieved_bytes[i])), 
            "finished_processing": new_db_file.finished_processing
        })
    
    return results

async def update_documents(database : Session,
                           toolchain_function_caller: Callable[[Any], Union[Callable, Awaitable[Callable]]],
                           auth : AuthType,
                           data : List[dict] = None,
                           file: Union[List[str], UploadFile] = None,
                           create_embeddings : bool = True,
                           await_embedding : bool = False):
    """
    Set the text for a given document in the database,
    and perform text chunking, and optionally embedding.
    
    
    You can upload a zipped JSONL file or a data field with the same content.
    If uploading a zip/7zip file, the file must be named `metadata.jsonl`.
    Each entry must match the following scheme:
    ```python
    # This is your input data
    class MainArgument(BaseModel):
        document_id: str
        text: Optional[Union[str, List[TextChunks]]] = None
        metadata: Optional[Dict[str, Any]] = None
        scan: Optional[bool] = False
        
    
    class TextChunks(BaseModel):
        text: str
        metadata: Optional[Dict[str, Any]] = None
    ```
    `scan` indicates whether to start scanning the file to extract text, chunk it, and put it into the database.
    `text` allows you to manually set the text of the document, and takes priority over `scan` if it is provided.
    `metadata` allows you to set the metadata of the document.
    
    """
    
    (user, user_auth) = get_user(database, auth)
    
    if not file is None:
        file_name = file.filename
        file_ext = file_name.split(".")[-1]
        file.file.seek(0)
        file_data_bytes = file.file.read()
        
        file_bytes_io = BytesIO(file_data_bytes)
        
        if file_ext == "7z":
            file_raw : BytesIO = aes_decrypt_zip_file(database, None, file_bytes_io, file_name="metadata.jsonl")
            data = [json.loads(line) for line in file_raw.getvalue().decode().split("\n") if len(line) > 0]
        elif file_ext == "zip":
            with zipfile.ZipFile(file_bytes_io, 'r') as z:
                with z.open("metadata.jsonl", "r") as file:
                    file_entry_bytes = file.read()
                    file_raw = BytesIO(file_entry_bytes)
                    file.close()
                z.close()
            data = [json.loads(line) for line in file_raw.getvalue().decode().split("\n") if len(line) > 0]
        elif file_ext == "jsonl":
            data = [json.loads(line) for line in file_data_bytes.decode().split("\n") if len(line) > 0]
        else:
            raise ValueError(f"File extension `{file_ext}` not supported for archival, only .7z, .zip, and .jsonl")
    else:
        assert not data is None, "Must provide data or file"
        data = [data] if not isinstance(data, list) else data

    data : List[DocumentModifierArgs] = [(DocumentModifierArgs(**entry) if not isinstance(entry, DocumentModifierArgs) else entry) for entry in data ]
    
    # print(json.dumps([e.model_dump() for e in data], indent=4))
    
    assert len(data) > 0, "No data provided"
    assert len(data) < 1000, "Too many documents to update at once"
    
    unique_doc_ids = list(set([entry.document_id for entry in data]))
    documents = list(database.exec(select(sql_db_tables.document_raw).where(sql_db_tables.document_raw.id.in_(unique_doc_ids))).all())
    # document_chunks = list(database.exec(select(sql_db_tables.DocumentChunk).where(sql_db_tables.DocumentChunk.document_id.in_(unique_doc_ids))).all())
    # document_collection_ids = list(set([c.collection_id for c in document_chunks]))
    document_collection_ids = list(set([getattr(d, c) for c in [
        "document_collection_id"
    ] for d in documents if not getattr(d, c) is None]))
    # print("Document Collection IDs:", document_collection_ids)
    
    assert_collections_priviledge(database, auth, document_collection_ids, modifying=True)
    
    documents = {document.id: document for document in documents}
    data_entries_lookup = {entry.document_id: entry for entry in data}
    
    
    # document_chunk_lookup : Dict[str, List[sql_db_tables.DocumentChunk]] = {document_id: [] for document_id in unique_doc_ids}
    # for chunk in document_chunks:
    #     document_chunk_lookup[chunk.document_id].append(chunk)
    
    # For updates to text or scan triggers, delete all prior chunks in the database.
    documents_to_clear_chunks = [
        entry.document_id
        for entry in data
        if (not entry.text is None) or (not entry.scan is False)
    ]
    database.exec(delete(sql_db_tables.DocumentChunk).where(sql_db_tables.DocumentChunk.document_id.in_(documents_to_clear_chunks)))
    
    text_updates, text_update_doc_ids = {}, []
    
    for entry in data:
        if not entry.text is None:
            text_updates[entry.document_id] = entry.text
            text_update_doc_ids.append(entry.document_id)
    
    for entry in data:
        document = documents[entry.document_id]
        
        if not entry.metadata is None:
            document.md = {
                **document.md,
                **entry.metadata
            }
            # TODO: Try to fix the auto-update trigger or find another way to get this to work.
            # This is efficient.
            database.exec(
                update(sql_db_tables.DocumentChunk)
                .where(sql_db_tables.DocumentChunk.document_id == document.id)
                .values(document_md=document.md)
            )
            # for chunk in document_chunk_lookup[document.id]:
            #     chunk.document_md = document.md
    
    text_update_doc_ids = list(set(text_update_doc_ids))
    chunking_docs = [documents[doc_id] for doc_id in text_update_doc_ids]
    chunking_texts = [text_updates[doc_id] for doc_id in text_update_doc_ids]
    chunking_names = [documents[doc_id].file_name for doc_id in text_update_doc_ids]
    chunking_metadata = [
        data_entries_lookup[doc_id].metadata 
        if not data_entries_lookup[doc_id].metadata is None
        else None
        for doc_id in text_update_doc_ids 
    ]
    
    # Run chunking on manually set texts.
    if await_embedding and len(chunking_docs) > 0:
        await chunk_documents(toolchain_function_caller,
                              database,
                              auth,
                              chunking_docs,
                              chunking_names,
                              document_texts=chunking_texts,
                              document_metadata=chunking_metadata,
                              create_embeddings=create_embeddings)
    elif len(chunking_docs) > 0:
        asyncio.create_task(chunk_documents(toolchain_function_caller,
                                            database,
                                            auth, 
                                            chunking_docs,
                                            chunking_names,
                                            document_texts=chunking_texts,
                                            document_metadata=chunking_metadata,
                                            create_embeddings=create_embeddings))
    
    database.commit()
    return True
        

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
    
    collection = database.exec(select(sql_db_tables.document_collection).where(sql_db_tables.document_collection.id == document.document_collection_id)).first()
    assert not collection is None, "Collection not found"
    
    if collection.collection_type == "user":
        assert (collection.author_user_name == user_auth.username) or user.is_admin, "User not authorized"
    
    elif collection.collection_type == "organization":
        organization = database.exec(select(sql_db_tables.organization).where(sql_db_tables.organization.id == collection.author_organization_id)).first()

        memberships = database.exec(select(sql_db_tables.organization_membership).where(and_(sql_db_tables.organization_membership.organization_id == organization.id,
                                                                                    sql_db_tables.organization_membership.user_name == user_auth.username))).all()
        
        assert (len(memberships) > 0 and memberships[0].role in ["owner", "admin", "member"]) \
            or user.is_admin, "User not authorized"
    else:
        raise ValueError(f"Collection type `{collection.collection_type}` not supported on this method yet.")

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
    
    document_password = get_collection_document_password(database, auth, document.document_collection_id)
    
    
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
    # print("Header Data:", [document.server_zip_archive_path, encryption_key])


    file = encryption.aes_decrypt_zip_file(
        database,
        encryption_key, 
        document.id
    )
    keys = list(file.keys())
    file_name = keys[0]
    file_get = file[file_name]
    return file_get

async def download_document(database : Session,
                            server_private_key : str,
                            document_auth_access : str):
    """
    Decrypt document in memory for the user's viewing.
    Return as a streaming response of bytes.
    
    You must first get an access token with 
    `/api/craft_document_access_token` to use this.
    """
    # print("Fetching document with auth:", document_auth_access)
    
    document_auth_access = json.loads(encryption.ecc_decrypt_string(server_private_key, document_auth_access))
    # print(json.dumps(document_auth_access, indent=4))

    document_auth_access["hash_id"] = document_auth_access["document_hash_id"]

    # print("Document Request Access Auth:", document_auth_access)
    
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


def fetch_document(
    database : Session,
    auth: AuthType,
    document_id: str,
    get_chunk_count: bool = False,
):
    """
    Fetch a document's row entry from the database.
    """
    (_, _) = get_user(database, auth)
    
    document = list(database.exec(select(sql_db_tables.document_raw).where(sql_db_tables.document_raw.id == document_id)).all())
    
    document = document[0] if len(document) > 0 else None
    assert not document is None, "Document not found"
    document_dict = document.model_dump()
    
    collection_fields = [
        "document_collection_id",
    ]
    
    collection_ids = [document_dict[c] for c in collection_fields]
    
    collection_id = [c for c in collection_ids if not c is None][0]
    
    assert_collections_priviledge(database, auth, [collection_id])
    
    document_dict = {
        k : v
        for k, v in document_dict.items()
        if (not k in [
            "blob_id",
            "blob_dir",
            "encryption_key_secure",
            "file_data"
        ] + collection_fields) and (not v is None)
    }
    
    document_dict.update({"collection_id": collection_id})
    
    if not get_chunk_count:
        return document_dict
    
    stmt = select(func.count()).select_from(sql_db_tables.DocumentChunk).where(
        sql_db_tables.DocumentChunk.document_id == document_id
    )
    chunk_count = int(database.scalar(stmt))
    
    return {
        **document_dict,
        "chunk_count": chunk_count
    }

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

def trigger_database_sql_error(database : Session,
                               auth: AuthType):
    """
    Trigger a SQL error for testing purposes.
    """
    
    _, _ = get_user(database, auth)
    new_text_chunk = sql_db_tables.DocumentChunk(
        id="test",
        document_name="test.py",
        text="test"
    )
    new_text_chunk.id = "ERROR_TEST"
    new_text_chunk_2 = sql_db_tables.DocumentChunk(
        id="test",
        document_name="test.py",
        text="test"
    )
    new_text_chunk_2.id = "ERROR_TEST"
    database.add(new_text_chunk)
    database.add(new_text_chunk_2)
    database.commit()
    return True


def call_surya_model(
        database : Session,
        server_surya_handles: Dict[str, Any],
        # toolchain_function_caller: Callable[[Any], Union[Callable, Awaitable[Callable]]],
        # auth : AuthType, 
        # file : Union[UploadFile, BytesIO],
        # model: Literal["texify", "layout", "detection", "recognition", "ordering", "table"]
    ):
    """
    
    """
    # if isinstance(file, UploadFile):
    #     file_name = file.filename
    #     file.file.seek(0)
    #     file = BytesIO(file.file.read())
    #     file.name = file_name
    
    print("Server surya handles:", server_surya_handles)





