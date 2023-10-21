from hashlib import sha256
import random
from .. import sql_db
from sqlmodel import Session, select
import time
from .. import encryption
from .hashing import *
import os

server_dir = "/".join(os.path.dirname(os.path.realpath(__file__)).split("/")[:-1])
user_db_path = server_dir+"/user_db/files/"

def add_user(database : Session, username : str, password : str) -> bool:
    """
    Add user to the database.
    """
    if len(username) > 32:
        return {"account_made": False, "note": "Name too long"}
    if len(password) > 32:
        return {"account_made": False, "note": "Password too long"}
    statement = select(sql_db.user).where(sql_db.user.name == username) 
    if len(database.exec(statement).all()) > 0:
        return {"account_made": False, "note": "Username already exists"}
    random_salt_1 = sha256(str(random.getrandbits(512)).encode('utf-8')).hexdigest()

    private_key_encryption_salt = random_hash()
    (public_key, private_key) = encryption.ecc_generate_public_private_key()

    private_key_encryption_key = hash_function(password, private_key_encryption_salt)

    new_user = sql_db.user(
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

    new_access_token = sql_db.access_token(
        type="user_primary_token",
        creation_timestamp=time.time(),
        author_user_name=username,
        hash_id=random_hash(),
    )
    database.add(new_access_token)
    database.commit()
    return {"account_made": True, "password_single_hash": hash_function(password)}

def login(database : Session, username : str, password : str):
    """
    This is for verifying a user login, and providing them their password prehash.
    """
    statement = select(sql_db.user).where(sql_db.user.name == username)
    retrieved = database.exec(statement).all()
    if len(retrieved) > 0:
        # with open(user_db_path+name_hash+".json", 'r', encoding='utf-8') as f:
        #     user_data = json.load(f)
        user_data = sql_db.data_dict(retrieved[0])
        password_salt = user_data["password_salt"]
        password_hash_truth = user_data["password_hash"]
        password_hash = hash_function(password, password_salt)
        if (password_hash == password_hash_truth):
            return {"successful": True, "password_single_hash": hash_function(password)}
        return {"successful": False, "note": "Incorrect Password"}
    else:
        return {"successful": False, "note": "User not found"}


def get_user_id(database : Session, username : str, password_prehash : str) -> int:
    """
    Authenticate a user and return the id field of their entry in the SQL database.
    Returns -1 if the username doesn't exist.
    Returns -2 if the username exists but the hash is invalid.
    """
    statement = select(sql_db.user).where(sql_db.user.name == username)
    
    # print("2")
    retrieved = database.exec(statement).all()
    # print("3")
    if len(retrieved) > 0:
        user_data = sql_db.data_dict(retrieved[0])
        password_salt = user_data["password_salt"]
        password_hash_truth = user_data["password_hash"]
        password_hash = hash_function(password_prehash, password_salt, only_salt=True)
        if (password_hash != password_hash_truth):
            return -2
        return user_data["id"]
    else:
        return -1

def get_user(database : Session, username : str, password_prehash : str) -> sql_db.user:
    """
    Returns the a user by lookup after verifying, raises an error otherwise.
    """
    statement = select(sql_db.user).where(sql_db.user.name == username)
    retrieved = database.exec(statement).all()
    if len(retrieved) > 0:
        password_salt = retrieved[0].password_salt
        password_hash_truth = retrieved[0].password_hash
        password_hash = hash_function(password_prehash, password_salt, only_salt=True)
        if password_hash == password_hash_truth:
            return retrieved[0]
        else:
            return ValueError("User Verification Failed")
    else:
        raise IndexError("User Not Found")
    # except:
    #     raise ValueError("Error Validating User")
