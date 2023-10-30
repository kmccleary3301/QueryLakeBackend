from hashlib import sha256
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

def upload_document_to_collection(database : Session, 
              username : str, 
              password_prehash : str, 
              file : UploadFile, 
              collection_hash_id : str, 
              collection_type : str = "user",
              organization_hash_id : str = None,
              public : bool = False) -> dict:
    """
    Upload file to server. Possibly with encryption.
    Can be a user document, organization document, or global document.
    """

    print("Adding document to collection")
    collection_type_lookup = {
        "user": "user_document_collection_hash_id", 
        "organization": "organization_document_collection_hash_id",
        "global": "global_document_collection_hash_id"    
    }

    assert collection_type in ["global", "organization", "user"]

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
    file_data_raw = file.file.read()
    file_integrity = sha256(file_data_raw).hexdigest()
    
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
        size_bytes=len(file_data_raw),
        creation_timestamp=time.time(),
        public=public,
        encryption_key_secure=encryption.ecc_encrypt_string(public_key, encryption_key),
        **collection_author_kwargs
    )
    

    if public:
        encryption.aes_encrypt_zip_file(key=None, file_data={file.filename: BytesIO(file_data_raw)}, save_path=file_zip_save_path)
    else:
        encryption.aes_encrypt_zip_file(key=encryption_key, file_data={file.filename: BytesIO(file_data_raw)}, save_path=file_zip_save_path)
    
    print("Collection:", collection)
    assert not collection is None, "Collection not found"

    collection.document_count += 1
    database.add(new_db_file)
    database.commit()

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
                        hash_id: str):
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

    return {"password": document_password, "database_path": document.server_zip_archive_path}


