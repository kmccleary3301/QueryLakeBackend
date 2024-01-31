from ..database import sql_db_tables
from sqlmodel import Session, select, and_
from .hashing import *
from ..typing.config import AuthType, getUserType, AuthType1, AuthType2

def get_user(database : Session, 
             auth: AuthType) -> getUserType:
    """
    Returns the a user by lookup after verifying, raises an error otherwise.
    """
    print("Auth:", auth)
    
    if type(auth) is dict:
        if "password_prehash" in auth:
            auth = AuthType1(**auth)
        else:
            raise TypeError("Invalid Auth Type")
        # conversion_success = False
        # for auth_type in [AuthType1, AuthType2]:
        #     try:
        #         auth = auth_type.parse_obj(auth)
        #         conversion_success = True
        #         break
        #     except:
        #         pass
        # if not conversion_success:
    
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