from ..database import sql_db_tables
from sqlmodel import Session, select, and_
from .hashing import *
from ..typing.config import AuthType, getUserType, AuthType1, AuthType2
from ..database.encryption import aes_decrypt_string

def get_user(database : Session, 
             auth: AuthType) -> getUserType:
    """
    Returns the a user by lookup after verifying, raises an error otherwise.
    The return value must always be a tuple of two items.
    The first is the user db entry.
    The second is an object with the username and the user's password prehash.
    
    TODO: Add OAuth2 support.
    """
    
    if type(auth) is dict:
        conversion_success = False
        for auth_type in [AuthType1, AuthType2]:
            try:
                auth = auth_type(**auth)
                conversion_success = True
                break
            except:
                pass
        assert conversion_success, "Auth type not recognized."
    
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
        
        user_password_prehash = aes_decrypt_string(auth.api_key, retrieved.user_password_prehash_encrypted)
        
        user_db_entry_statement = select(sql_db_tables.user).where(sql_db_tables.user.name == retrieved.author)
        user_db_entry = database.exec(user_db_entry_statement).first()
        
        user_auth = AuthType1(username=retrieved.author, password_prehash=user_password_prehash)
        
        return (user_db_entry, user_auth)