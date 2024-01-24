from ..database import sql_db_tables
from sqlmodel import Session, select, and_
from .hashing import *
from ray.actor import ActorHandle
from ray import get

def get_user(database : ActorHandle, username : str, password_prehash : str) -> sql_db_tables.user:
    """
    Returns the a user by lookup after verifying, raises an error otherwise.
    """
    statement = select(sql_db_tables.user).where(sql_db_tables.user.name == username)
    retrieved = get(database.exec.remote(statement)).all()
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