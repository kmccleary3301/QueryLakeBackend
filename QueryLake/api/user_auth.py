from hashlib import sha256
import random
from ..database import sql_db_tables
from sqlmodel import Session, select, and_
import time
from ..database import encryption
from .hashing import *
import os, json
from .organizations import fetch_memberships
from ..typing.config import Config, AuthType, getUserType, AuthType1, AuthType2, AuthType3, AuthType4, AuthInputType
from ..typing.toolchains import deleteAction
from typing import Tuple, Callable, Awaitable, Any, Union, List
from .single_user_auth import get_user, OAUTH_SECRET_KEY, process_input_as_auth_type
from ..database.encryption import aes_decrypt_string, aes_encrypt_string
from fastapi_login import LoginManager
from jose import JWTError, jwt
from fastapi import FastAPI, HTTPException
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.security.oauth2 import OAuth2PasswordRequestFormStrict
from passlib.context import CryptContext
from datetime import datetime, timedelta, timezone
from ..misc_functions.toolchain_state_management import run_sequence_action_on_object


# pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
# oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
# manager = LoginManager(SECRET_KEY, tokenUrl='/auth/token', use_cookie=True)

def add_user(
    database : Session,
    toolchain_function_caller: Callable[[Any], Union[Callable, Awaitable[Callable]]],
    global_config : Config,
    username : str, 
    password : str
) -> dict:
    """
    Add user to the database.
    
    Depending on your user configuration, you can send a confirmation email for signup,
    and communicate this in the response via {"pending_email": True}.
    
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
    
    response = login(database, toolchain_function_caller, global_config, AuthType3(username=username, password=password))
    response.update({
        "pending_email": False
    })
    return response

def login(
    database : Session,
    toolchain_function_caller: Callable[[Any], Union[Callable, Awaitable[Callable]]],
    global_config : Config,
    auth : AuthInputType
) -> dict:
    """
    This is for verifying a user login, and providing them their password prehash.
    """
    
    auth : AuthType = process_input_as_auth_type(auth)
    (user, user_auth) = get_user(database, auth)
    
    assert not isinstance(auth, AuthType2), "API keys cannot be used to login."
    
    fetch_memberships_get = fetch_memberships(database, auth, return_subset="all")

    get_toolchains_function = toolchain_function_caller("get_available_toolchains")
    
    toolchain_info = get_toolchains_function(database, global_config, auth)
    
    return {
        "username": user_auth.username,
        "auth": auth.oauth2 if isinstance(auth, AuthType4) else create_oauth2_token(database, auth),
        "memberships": fetch_memberships_get["memberships"],
        "admin": fetch_memberships_get["admin"],
        "available_models": get_available_models(database, global_config, auth)["available_models"],
        "available_toolchains": toolchain_info["toolchains"],
        "default_toolchain": toolchain_info["default"],
        "user_set_providers": list(get_user_external_providers_dict(database, auth).keys()),
        "providers": global_config.providers
    }

def create_oauth2_token(
    database : Session,
    auth : AuthInputType
) -> dict:
    """
    Create an OAuth2 token for the user.
    """
    auth : AuthType = process_input_as_auth_type(auth)
    
    assert isinstance(auth, (AuthType1, AuthType3)), "OAuth2 tokens can only be created with username and password."
    
    (_, user_auth) = get_user(database, auth)
    
    to_encode = {
        "username": user_auth.username,
        "pwd_hash": user_auth.password_prehash
    }
    
    expire = datetime.now(timezone.utc) + timedelta(days=30)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, OAUTH_SECRET_KEY, algorithm="HS256")
    return encoded_jwt
    
def get_user_id(
    database : Session, 
    username : str, 
    password_prehash : str
) -> int:
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

def get_user_private_key(
    database : Session, 
    auth : AuthType
) -> str:
    """
    Fetch user private key.
    """
    (user, user_auth) = get_user(database, auth)

    private_key_encryption_salt = user.private_key_encryption_salt
    user_private_key_decryption_key = hash_function(user_auth.password_prehash, private_key_encryption_salt, only_salt=True)
    
    user_private_key = encryption.aes_decrypt_string(user_private_key_decryption_key, user.private_key_secured)

    return {"private_key": user_private_key}

def get_organization_private_key(
    database : Session, 
    auth : AuthType, 
    organization_hash_id : str
) -> str:
    """
    Decrypt organization private key from user's membership entry in the database.
    """
    (user, user_auth) = get_user(database, auth)

    organization = database.exec(select(sql_db_tables.organization).where(sql_db_tables.organization.hash_id == organization_hash_id)).first()

    memberships = database.exec(select(sql_db_tables.organization_membership).where(and_(sql_db_tables.organization_membership.organization_id == organization.id,
                                                                                sql_db_tables.organization_membership.user_name == user_auth.username))).all()
    assert len(memberships) > 0, "User not authorized with organization"

    private_key_encryption_salt = user.private_key_encryption_salt
    user_private_key_decryption_key = hash_function(user_auth.password_prehash, private_key_encryption_salt, only_salt=True)

    user_private_key = encryption.ecc_decrypt_string(user_private_key_decryption_key, user.private_key_secured)

    organization_private_key = encryption.ecc_decrypt_string(user_private_key, memberships[0].organization_private_key_secure)

    return {"private_key": organization_private_key}

def get_available_models(
    database : Session,
    global_config : Config,
    auth : AuthType
):
    """
    Gets a list of all models on the server available to the given user.
    Plan on making changes so that organizations can have private models.
    """
    (user, user_auth) = get_user(database, auth)
    models = database.exec(select(sql_db_tables.model)).all()
    global_config_json = global_config.dict()
    
    external_models = {
        "openai": global_config.external_model_providers["openai"]
    }
    
    results = {
        "default_models": global_config.default_models,
        # "local_models": [{k : e[k] for k in ["name", "modelcard", "max_model_len"]} for e in global_config.models],
        "local_models": [{k:v for k,v in e.dict().items() if k in ["name", "id", "modelcard"]} for e in global_config.models],
        "external_models": external_models
    }
    # return {"success" : True, "result" : results}
    return {"available_models": results}

def set_user_openai_api_key(
    database : Session, 
    auth : AuthType,
    openai_api_key : str
):
    """
    Sets user OpenAI API key in SQL db.
    Necessary to use OpenAI models for chat outputs.
    """
    (user, user_auth) = get_user(database, auth)
    encrypted_api_key = encryption.ecc_encrypt_string(user.public_key, openai_api_key)
    user.openai_api_key_encrypted = encrypted_api_key
    database.commit()
    # return {"success": True}
    return True

def set_organization_openai_id(
    database : Session, 
    auth : AuthType,
    openai_organization_id : str,
    organization_hash_id : str
):
    """
    Sets organization OpenAI ID Key.
    Using this allows users to use OpenAI models with charges made
    to the OpenAI organization instead.
    """
    (user, user_auth) = get_user(database, auth)
    organization = database.exec(select(sql_db_tables.organization).where(sql_db_tables.organization.hash_id == organization_hash_id)).first()
    
    memberships = database.exec(select(sql_db_tables.organization_membership).where(and_(sql_db_tables.organization_membership.organization_id == organization.id,
                                                                                sql_db_tables.organization_membership.user_name == user_auth.username))).all()
    assert len(memberships) > 0, "User not authorized with organization"
    assert memberships[0].role in ["owner", "admin", "member"], "User not authorized to set SERP key"

    encrypted_openai_organization_id = encryption.ecc_encrypt_string(organization.public_key, openai_organization_id)

    organization.openai_organization_id_encrypted = encrypted_openai_organization_id
    database.commit()

    # return {"success": True}
    return True

def get_openai_api_key(
    database : Session, 
    auth : AuthType,
    organization_hash_id : str = None
):
    """
    Retrieve user OpenAI API key.
    If organization is specified, return an array with the former plus
    the organization OpenAI ID.
    """
    (user, user_auth) = get_user(database, auth)
    return_result = []
    
    if not organization_hash_id is None:
        organization_private_key = get_organization_private_key(database, auth, organization_hash_id)["private_key"]
        organization = database.exec(select(sql_db_tables.organization).where(sql_db_tables.organization.hash_id == organization_hash_id)).first()
        organization_openai_id_encrypted = organization.openai_organization_id_encrypted
        assert not organization_openai_id_encrypted is None, "Organization OpenAI ID not set"
        openai_organization_id = encryption.ecc_decrypt_string(organization_private_key, organization_openai_id_encrypted)

    user_private_key = get_user_private_key(database, auth)["private_key"]
    encrypted_openai_api_key = user.openai_api_key_encrypted
    assert not encrypted_openai_api_key is None, "User OpenAI API key not set"
    user_openai_api_key = encryption.ecc_decrypt_string(user_private_key, encrypted_openai_api_key)
    if not organization_hash_id is None:
        # return {"success": True, "result": [user_openai_api_key, openai_organization_id]}
        return {"api_key": user_openai_api_key, "organization_id": openai_organization_id}
    # return {"success": True, "result": user_openai_api_key}
    return {"api_key": user_openai_api_key}

def create_api_key(
    database : Session, 
    auth : AuthInputType,
    title : str = None
):
    """
    Create a new API key for the user.
    """
    
    auth : AuthType = process_input_as_auth_type(auth)
    assert not isinstance(auth, AuthType2), "API keys cannot be used to create API keys."
    
    (user, user_auth) = get_user(database, auth)
    
    random_key_hash = random_hash(base=62, length=48)
    
    api_key_actual = f"sk-{random_key_hash}"
    api_key_preview = f"sk-...{random_key_hash[-4:]}"
    
    api_key_hash = hash_function(api_key_actual)
    
    new_api_key = sql_db_tables.ApiKey(
        key_hash=api_key_hash,
        creation_timestamp=time.time(),
        author=user.name,
        **{"title": title} if not title is None else {},
        key_preview=api_key_preview,
        user_password_prehash_encrypted=aes_encrypt_string(api_key_actual, user_auth.password_prehash)
    )
    
    database.add(new_api_key)
    database.commit()
    return {
        "api_key": api_key_actual, 
        "id": new_api_key.id,
        "title": new_api_key.title,
        "created": new_api_key.creation_timestamp,
        "last_used": new_api_key.last_used,
        "key_preview": new_api_key.key_preview
    }

def delete_api_key(
    database : Session, 
    auth : AuthInputType,
    api_key_id : str
):
    """
    Delete an API key by its id.
    """
    auth : AuthType = process_input_as_auth_type(auth)
    assert not isinstance(auth, AuthType2), "API keys cannot be used to delete API keys."
    
    (user, user_auth) = get_user(database, auth)
    
    api_key = database.exec(select(sql_db_tables.ApiKey).where(sql_db_tables.ApiKey.id == api_key_id)).first()
    assert not api_key is None, "API Key not found"
    
    assert api_key.author == user.name, "API Key does not belong to user"
    
    database.delete(api_key)
    database.commit()
    
    return True

def fetch_api_keys(
    database : Session, 
    auth : AuthInputType
):
    """
    Fetch all API keys belonging to the user.
    """
    auth : AuthType = process_input_as_auth_type(auth)
    assert not isinstance(auth, AuthType2), "API keys cannot be used to fetch API keys."
    
    (user, _) = get_user(database, auth)
    
    api_keys = database.exec(select(sql_db_tables.ApiKey).where(sql_db_tables.ApiKey.author == user.name)).all()
    
    api_keys = [{
        "id": e.id, 
        "title": e.title,
        "created": e.creation_timestamp,
        "last_used": e.last_used,
        "key_preview": e.key_preview
    } for e in api_keys]
    
    return {"api_keys": api_keys}

def get_user_external_providers_dict(
    database : Session,
    auth : AuthType
) -> dict:
    """
    Get user external providers dictionary.
    """
    # print("Auth Type:", type(auth))
    
    (user, user_auth) = get_user(database, auth)
    
    private_key_encryption_salt = user.private_key_encryption_salt
    user_private_key_decryption_key = hash_function(user_auth.password_prehash, private_key_encryption_salt, only_salt=True)
    user_private_key = encryption.aes_decrypt_string(user_private_key_decryption_key, user.private_key_secured)
    
    if user.external_providers_encrypted is None:
        return {}
    
    external_providers = encryption.aes_decrypt_string(user_private_key, user.external_providers_encrypted)
    external_providers = json.loads(external_providers)
    
    return external_providers


def modify_user_external_providers(
    database : Session,
    global_config : Config,
    auth : AuthInputType,
    update : dict = None,
    delete : List[Union[str, int]] = None
):
    """
    Modify user external providers.
    """
    auth : AuthType = process_input_as_auth_type(auth)
    assert not isinstance(auth, AuthType2), "API keys cannot be used to change external providers."
    assert not update is None or not delete is None, "Must specify either update or delete."
    if not update is None:
        assert all([e in global_config.providers for e in update.keys()]), "Invalid provider"
    else:
        assert delete[0] in global_config.providers, "Invalid provider"
    
    (user, user_auth) = get_user(database, auth)
    
    private_key_encryption_salt = user.private_key_encryption_salt
    user_private_key_decryption_key = hash_function(user_auth.password_prehash, private_key_encryption_salt, only_salt=True)
    user_private_key = encryption.aes_decrypt_string(user_private_key_decryption_key, user.private_key_secured)
    
    if user.external_providers_encrypted is None:
        external_providers = {}
    else:
        external_providers = encryption.aes_decrypt_string(user_private_key, user.external_providers_encrypted)
        external_providers = json.loads(external_providers)
    
    if not update is None:
        external_providers.update(update)
    else:
        external_providers = run_sequence_action_on_object(
            external_providers,
            {}, {}, {}, {}, [deleteAction(route=delete)]
        )
    
    write_external_providers = json.dumps(external_providers)
    write_external_providers = encryption.aes_encrypt_string(user_private_key, write_external_providers)
    
    user.external_providers_encrypted = write_external_providers if len(external_providers) > 0 else None
    
    database.commit()
    
    return True