from hashlib import sha256
import random
from ..database import sql_db_tables
from sqlmodel import Session, select, and_
import time
from ..database import encryption
from .hashing import *
import os, json
from .organizations import fetch_memberships
# from .toolchains import get_available_toolchains

server_dir = "/".join(os.path.dirname(os.path.realpath(__file__)).split("/")[:-1])
user_db_path = server_dir+"/user_db/files/"

server_dir = "/".join(os.path.dirname(os.path.realpath(__file__)).split("/")[:-1])
upper_server_dir = "/".join(os.path.dirname(os.path.realpath(__file__)).split("/")[:-2])+"/"
user_db_path = server_dir+"/user_db/files/"
with open(upper_server_dir+"config.json", 'r', encoding='utf-8') as f:
    file_read = f.read()
    f.close()
GLOBAL_SETTINGS = json.loads(file_read)

def add_user(database : Session, username : str, password : str) -> dict:
    """
    Add user to the database.
    """
    assert len(username) <= 32, "Name too long"
    assert len(password) <= 32, "Password too long"
    statement = select(sql_db_tables.user).where(sql_db_tables.user.name == username) 
    result = database.exec(statement)
    assert len(result.all()) == 0, "Username already exists"
    
    random_salt_1 = sha256(str(random.getrandbits(512)).encode('utf-8')).hexdigest()

    private_key_encryption_salt = random_hash()
    (public_key, private_key) = encryption.ecc_generate_public_private_key()

    private_key_encryption_key = hash_function(password, private_key_encryption_salt)

    print("Adding user with private key:", [private_key])
    tmp_encrypt = encryption.aes_encrypt_string(private_key_encryption_key, private_key)
    print("Encrypted to:", [tmp_encrypt, type(tmp_encrypt)])

    new_user = sql_db_tables.user(
        name=username,
        password_salt=random_salt_1,
        password_hash=hash_function(password, random_salt_1),
        creation_timestamp=time.time(),
        public_key=public_key,
        private_key_encryption_salt=private_key_encryption_salt,
        private_key_secured=encryption.aes_encrypt_string(private_key_encryption_key, private_key)
    )

    database.add(new_user)
    database.commit()

    new_access_token = sql_db_tables.access_token(
        type="user_primary_token",
        creation_timestamp=time.time(),
        author_user_name=username,
        hash_id=random_hash(),
    )
    database.add(new_access_token)
    database.commit()

    password_prehash = hash_function(password)

    fetch_memberships_get = fetch_memberships(database, username, password_prehash, return_subset="all")
    # available_toolchains_get = get_available_toolchains(database, username, password_prehash)
    return {
        "account_made": True,
        "password_single_hash": password_prehash,
        "memberships": fetch_memberships_get["memberships"],
        "admin": fetch_memberships_get["admin"],
        "available_models": get_available_models(database, username, password_prehash)["available_models"],
    }

def login(database : Session, username : str, password : str) -> dict:
    """
    This is for verifying a user login, and providing them their password prehash.
    """
    print("Logging in user:", [username, password])
    
    statement = select(sql_db_tables.user).where(sql_db_tables.user.name == username)
    retrieved = database.exec(statement).all()
    if len(retrieved) > 0:
        # with open(user_db_path+name_hash+".json", 'r', encoding='utf-8') as f:
        #     user_data = json.load(f)
        user_data = sql_db_tables.data_dict(retrieved[0])
        password_salt = user_data["password_salt"]
        password_hash_truth = user_data["password_hash"]
        password_hash = hash_function(password, password_salt)
        password_prehash = hash_function(password)
        if (password_hash == password_hash_truth):
            fetch_memberships_get = fetch_memberships(database, username, password_prehash, return_subset="all")
            # available_toolchains_get = get_available_toolchains(database, username, password_prehash)
            return {
                "password_single_hash": password_prehash,
                "memberships": fetch_memberships_get["memberships"],
                "admin": fetch_memberships_get["admin"],
                "available_models": get_available_models(database, username, password_prehash)["available_models"],
                # "available_toolchains": available_toolchains_get["toolchains"],
                # "default_toolchain": available_toolchains_get["default"]
            }
        # return {"successful": False, "note": "Incorrect Password"}
        assert False, "Incorrect Password"

    else:
        # return {"successful": False, "note": "User not found"}
        assert False, "User not found"

def get_user_id(database : Session, username : str, password_prehash : str) -> int:
    """
    Authenticate a user and return the id field of their entry in the SQL database.
    Returns -1 if the username doesn't exist.
    Returns -2 if the username exists but the hash is invalid.
    """
    statement = select(sql_db_tables.user).where(sql_db_tables.user.name == username)
    
    # print("2")
    retrieved = database.exec(statement).all()
    # print("3")
    if len(retrieved) > 0:
        user_data = sql_db_tables.data_dict(retrieved[0])
        password_salt = user_data["password_salt"]
        password_hash_truth = user_data["password_hash"]
        password_hash = hash_function(password_prehash, password_salt, only_salt=True)
        if (password_hash != password_hash_truth):
            return -2
        return user_data["id"]
    else:
        return -1

def get_user(database : Session, username : str, password_prehash : str) -> sql_db_tables.user:
    """
    Returns the a user by lookup after verifying, raises an error otherwise.
    """
    statement = select(sql_db_tables.user).where(sql_db_tables.user.name == username)
    retrieved = database.exec(statement).all()
    if len(retrieved) > 0:
        password_salt = retrieved[0].password_salt
        password_hash_truth = retrieved[0].password_hash
        password_hash = hash_function(password_prehash, password_salt, only_salt=True)
        if password_hash == password_hash_truth:
            return retrieved[0]
        else:
            raise ValueError("User Verification Failed")
    else:
        raise IndexError("User Not Found")
    # except:
    #     raise ValueError("Error Validating User")

def get_user_private_key(database : Session, username : str, password_prehash : str) -> str:
    """
    Fetch user private key.
    """
    user = get_user(database, username, password_prehash)

    private_key_encryption_salt = user.private_key_encryption_salt
    user_private_key_decryption_key = hash_function(password_prehash, private_key_encryption_salt, only_salt=True)
    
    user_private_key = encryption.aes_decrypt_string(user_private_key_decryption_key, user.private_key_secured)

    return {"private_key": user_private_key}

def get_organization_private_key(database : Session, username : str, password_prehash : str, organization_hash_id : str) -> str:
    """
    Decrypt organization private key from user's membership entry in the database.
    """
    user = get_user(database, username, password_prehash)

    organization = database.exec(select(sql_db_tables.organization).where(sql_db_tables.organization.hash_id == organization_hash_id)).first()

    memberships = database.exec(select(sql_db_tables.organization_membership).where(and_(sql_db_tables.organization_membership.organization_id == organization.id,
                                                                                sql_db_tables.organization_membership.user_name == username))).all()
    assert len(memberships) > 0, "User not authorized with organization"

    private_key_encryption_salt = user.private_key_encryption_salt
    user_private_key_decryption_key = hash_function(password_prehash, private_key_encryption_salt, only_salt=True)

    user_private_key = encryption.ecc_decrypt_string(user_private_key_decryption_key, user.private_key_secured)

    organization_private_key = encryption.ecc_decrypt_string(user_private_key, memberships[0].organization_private_key_secure)

    return {"private_key": organization_private_key}

def get_available_models(database : Session, username : str, password_prehash : str):
    """
    Gets a list of all models on the server available to the given user.
    Plan on making changes so that organizations can have private models.
    """
    user = get_user(database, username, password_prehash)
    models = database.exec(select(sql_db_tables.model)).all()
    external_models = {
        "openai": [key for key in GLOBAL_SETTINGS["external_model_providers"]["openai"].keys()]
    }
    results = {
        "default_model": GLOBAL_SETTINGS["default_model"],
        "local_models": [model.name for model in models],
        "external_models": external_models
    }
    # return {"success" : True, "result" : results}
    return {"available_models": results}

def set_user_openai_api_key(database : Session, 
                            username : str, 
                            password_prehash : str,
                            openai_api_key : str):
    """
    Sets user OpenAI API key in SQL db.
    Necessary to use OpenAI models for chat outputs.
    """
    user = get_user(database, username, password_prehash)
    encrypted_api_key = encryption.ecc_encrypt_string(user.public_key, openai_api_key)
    user.openai_api_key_encrypted = encrypted_api_key
    database.commit()
    # return {"success": True}
    return True

def set_organization_openai_id(database : Session, 
                                username : str, 
                                password_prehash : str,
                                openai_organization_id : str,
                                organization_hash_id : str):
    """
    Sets organization OpenAI ID Key.
    Using this allows users to use OpenAI models with charges made
    to the OpenAI organization instead.
    """
    user = get_user(database, username, password_prehash)
    organization = database.exec(select(sql_db_tables.organization).where(sql_db_tables.organization.hash_id == organization_hash_id)).first()

    memberships = database.exec(select(sql_db_tables.organization_membership).where(and_(sql_db_tables.organization_membership.organization_id == organization.id,
                                                                                sql_db_tables.organization_membership.user_name == username))).all()
    assert len(memberships) > 0, "User not authorized with organization"
    assert memberships[0].role in ["owner", "admin", "member"], "User not authorized to set SERP key"

    encrypted_openai_organization_id = encryption.ecc_encrypt_string(organization.public_key, openai_organization_id)

    organization.openai_organization_id_encrypted = encrypted_openai_organization_id
    database.commit()

    # return {"success": True}
    return True

def get_openai_api_key(database : Session, 
                        username : str, 
                        password_prehash : str,
                        organization_hash_id : str = None):
    """
    Retrieve user OpenAI API key.
    If organization is specified, return an array with the former plus
    the organization OpenAI ID.
    """
    user = get_user(database, username, password_prehash)
    return_result = []
    
    if not organization_hash_id is None:
        organization_private_key = get_organization_private_key(database, username, password_prehash, organization_hash_id)["private_key"]
        organization = database.exec(select(sql_db_tables.organization).where(sql_db_tables.organization.hash_id == organization_hash_id)).first()
        organization_openai_id_encrypted = organization.openai_organization_id_encrypted
        assert not organization_openai_id_encrypted is None, "Organization OpenAI ID not set"
        openai_organization_id = encryption.ecc_decrypt_string(organization_private_key, organization_openai_id_encrypted)

    user_private_key = get_user_private_key(database, username, password_prehash)["private_key"]
    encrypted_openai_api_key = user.openai_api_key_encrypted
    assert not encrypted_openai_api_key is None, "User OpenAI API key not set"
    user_openai_api_key = encryption.ecc_decrypt_string(user_private_key, encrypted_openai_api_key)
    if not organization_hash_id is None:
        # return {"success": True, "result": [user_openai_api_key, openai_organization_id]}
        return {"api_key": user_openai_api_key, "organization_id": openai_organization_id}
    # return {"success": True, "result": user_openai_api_key}
    return {"api_key": user_openai_api_key}

    