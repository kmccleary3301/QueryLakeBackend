from ..database import sql_db_tables
from sqlmodel import Session, select, and_
from .hashing import *
from ..typing.config import AuthType, getUserType, AuthType1, AuthType2, AuthType3, AuthType4, AuthInputType, user, getUserAuthType
from ..database.encryption import aes_decrypt_string, ecc_generate_public_private_key
from typing import Union, Tuple
from datetime import datetime, timedelta, timezone
from jose import JWTError, jwt
from time import time


global_public_key, global_private_key = ecc_generate_public_private_key()
OAUTH_SECRET_KEY = random_hash()

def process_input_as_auth_type(auth: AuthInputType) -> AuthType:
    conversion_success = False
    
    if isinstance(auth, (AuthType1, AuthType2, AuthType3, AuthType4)):
        return auth
    
    if isinstance(auth, str):
        return AuthType4(oauth2=auth)
    
    for auth_type in [AuthType1, AuthType2, AuthType3, AuthType4]:
        try:
            auth = auth_type(**auth)
            conversion_success = True
            break
        except:
            pass
    
    assert conversion_success, f"Auth type not recognized; {auth}"
    return auth


def get_user(database : Session,
             auth: AuthInputType) -> Tuple[user, getUserAuthType]:
    """
    Returns the a user by lookup after verifying, raises an error otherwise.
    The return value must always be a tuple of two items.
    The first is the user db entry.
    The second is an object with the username and the user's password prehash.
    
    TODO: Add OAuth2 support.
    """
    
    auth = process_input_as_auth_type(auth)
    
    # Username and password prehash case
    if isinstance(auth, AuthType1):
        statement = select(sql_db_tables.user).where(sql_db_tables.user.name == auth.username)
        retrieved = database.exec(statement).all()
        if len(retrieved) > 0:
            password_salt = retrieved[0].password_salt
            password_hash_truth = retrieved[0].password_hash
            password_hash = hash_function(auth.password_prehash, password_salt, only_salt=True)
            if password_hash == password_hash_truth:
                return (retrieved[0], auth)
            else:
                raise ValueError("User Verification Failed")
        else:
            raise IndexError("User Not Found")
    
    # Api Key case
    elif isinstance(auth, AuthType2):
        key_hash = hash_function(auth.api_key)
        
        statement = select(sql_db_tables.ApiKey).where(sql_db_tables.ApiKey.key_hash == key_hash)
        retrieved = database.exec(statement).first()
        
        assert retrieved is not None, "API Key Not Found"
        
        retrieved.last_used = time()
        database.commit()
        
        user_password_prehash = aes_decrypt_string(auth.api_key, retrieved.user_password_prehash_encrypted)
        
        user_db_entry_statement = select(sql_db_tables.user).where(sql_db_tables.user.name == retrieved.author)
        user_db_entry = database.exec(user_db_entry_statement).first()
        
        user_auth = AuthType1(username=retrieved.author, password_prehash=user_password_prehash)
        
        return (user_db_entry, user_auth)
    
    # Raw Password Login Case
    elif isinstance(auth, AuthType3):
        statement = select(sql_db_tables.user).where(sql_db_tables.user.name == auth.username)
        retrieved = database.exec(statement).first()
        if not retrieved is None:
            password_salt = retrieved.password_salt
            password_hash_truth = retrieved.password_hash
            password_hash = hash_function(auth.password, password_salt)
            if password_hash == password_hash_truth:
                return (retrieved, AuthType1(username=auth.username, password_prehash=hash_function(auth.password)))
            else:
                raise ValueError("Invalid Password.")
        else:
            raise IndexError("User Not Found")
        
    # OAuth2 Case
    elif isinstance(auth, AuthType4):
        
        payload = jwt.decode(auth.oauth2, OAUTH_SECRET_KEY, algorithms=["HS256"])
        username : str = payload.get("username")
        pwd_hash : str = payload.get("pwd_hash")
        token_expiration : datetime = datetime.fromtimestamp(payload.get("exp"), timezone.utc)
        
        print("Payload ->", payload)
        
        if username is None or pwd_hash is None:
            raise Exception("Your OAuth2 token has incomplete information.")
        
        temp_auth = AuthType1(username=username, password_prehash=pwd_hash)
        assert datetime.now(timezone.utc) < token_expiration, "Your OAuth2 token expired on " + str(token_expiration)
        
        return get_user(database, temp_auth)

    else:
        raise ValueError("Auth type not recognized.")