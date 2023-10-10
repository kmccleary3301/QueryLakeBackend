from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from hashlib import sha256
import json
import oauth2
import os
import random
from datetime import datetime
from fastapi import File, UploadFile
from . import sql_db
from sqlmodel import Field, Session, SQLModel, create_engine, select
import time
from string import Template

server_dir = "/".join(os.path.dirname(os.path.realpath(__file__)).split("/")[:-1])
print(server_dir)
if not os.path.exists(server_dir+"/user_db"):
    os.mkdir(server_dir+"/user_db")
if not os.path.exists(server_dir+"/user_db/files"):
    os.mkdir(server_dir+"/user_db/files")
user_db_path = server_dir+"/user_db/"

def get_random_hash():
    return sha256(str(random.getrandbits(512)).encode('utf-8')).hexdigest()

def hash_function(input : str, salt : str = None, only_salt : bool = False) -> str:
    if only_salt:
        term_1 = input
    else:
        term_1 = sha256(input.encode('utf-8')).hexdigest()
    if not salt is None:
        salt_term = sha256(salt.encode('utf-8')).hexdigest()
        term_2 = sha256((term_1+salt_term).encode('utf-8')).hexdigest()
        return term_2
    return term_1

def add_user(database : Session, username : str, password : str) -> bool:
    if len(username) > 32:
        return {"account_made": False, "note": "Name too long"}
    if len(password) > 32:
        return {"account_made": False, "note": "Password too long"}
    statement = select(sql_db.user).where(sql_db.user.name == username)
    if len(database.exec(statement).all()) > 0:
        return {"account_made": False, "note": "Username already exists"}
    random_salt_1 = sha256(str(random.getrandbits(512)).encode('utf-8')).hexdigest()
    # random_salt_2 = sha256(str(random.getrandbits(512)).encode('utf-8')).hexdigest()
    user_data = {
        "name" : username,
        "password_salt": random_salt_1,
        "password_hash": hash_function(password, random_salt_1),
        "creation_timestamp": time.time(),
    }
    new_user = sql_db.user(**user_data)

    database.add(new_user)
    database.commit()
    retrieved_entry = database.exec(select(sql_db.user).where(sql_db.user.name == username)).first()

    new_access_token = sql_db.access_token(
        type="user_primary_token",
        creation_timestamp=time.time(),
        author_user_name=username,
        hash_id=get_random_hash(),
    )
    database.add(new_access_token)
    database.commit()
    return {"account_made": True, "password_single_hash": hash_function(password)}

def auth_user(database : Session, username : str, password : str):
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
            return {"login_successful": True, "password_single_hash": hash_function(password)}
        return {"login_successful": False, "note": "Incorrect Password"}
    else:
        return {"login_successful": False, "note": "User not found"}

def file_save(database : Session, name : str, password_prehash : str, file : UploadFile, collection_id : int, collection_type : str = None) -> bool:
    """
    Upload file to server. Requires the following:
    username: username
    """
    
    statement = select(sql_db.user).where(sql_db.user.name == name)
    retrieved = database.exec(statement).all()
    if len(retrieved) > 0:
        user_data = sql_db.data_dict(retrieved[0])
        password_salt = user_data["password_salt"]
        password_hash_truth = user_data["password_hash"]
        password_hash = hash_function(password_prehash, password_salt, only_salt=True)
        if (password_hash != password_hash_truth):
            return {"file_upload_success": False, "note": "Incorrect Key"}
        file_id = hash_function(file.filename+" "+str(time.time()))
        file_name_save = file_id+"."+file.filename.split(".")[-1]
        file_location = user_db_path+f"files/{file_name_save}"
        file.file.seek(0)
        file_data_raw = file.file.read()
        # data_raw = file.read()
        file_integrity = sha256(file_data_raw).hexdigest()
        # Need to verify that collection author is also document author.
        # If collection is specified.

        new_db_file = sql_db.document_raw(
            name=file_id,
            type=file.filename.split(".")[-1],
            server_location=file_name_save,
            integrity_sha256=file_integrity,
            size_bytes=len(file),
            creation_timestamp=time.time(),
            
            # Function needs to be reworked for adding to doc collection.

        )
        database.add(new_db_file)
        database.commit()
        with open(file_location, "wb+") as file_object:
            file_object.write(file_data_raw)
        return {"file_upload_success": True, "note": f"file '{file.filename}' saved at '{file_location}'"}
    else:
        return {"file_upload_success": False, "note": "User not found"}
    
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

class TokenTracker:
    def __init__(self, session : Session) -> None:
        self.session = session
    
    def add_tokens(self, user_id : int, tokens : int) -> None:
        statement = select(sql_db.user).where(sql_db.user.id == user_id)
        retrieved = self.session.exec(statement).first()
        retrieved.tokens_generated += tokens
        self.session.commit()


def encode_padding(string : str) -> str:
    string = string.safe('&', '&amp;')
    string = string.replace('>', '&gt;')
    string = string.replace('<', '&lt;')
    replace_args = {
        "system_instruction": "<system_instruction>",
        "question": "<question>",
        "context": "<context>",
        "response": "<response>"
    }
    for key, value in replace_args.items():
        try:
            string = string.format(**{key: value})
        except:
            pass
    return string

def decode_padding(string: str) -> str:
    string = string.replace("<system_instruction>", "{system_instruction}")
    string = string.replace("<question>", "{question}")
    string = string.replace("<context>", "{context}")
    string = string.replace("<response>", "{response}")
    string = string.replace('&gt;', '>')
    string = string.replace('&lt;', '<')
    string = string.replace('&amp;', '&')
    return string

def add_model(database: Session, model_args: dict, padding: dict, loader: str, default_system_instruction: str) -> bool:
    """
    Adds a model to the database.
    Example usage:
      add_model(database,
                loader="exllama",
                model_args={
                    "streaming": True,
                    "model_path": "/home/user/python_projects/langchain/Llama-2-7b-Chat-GPTQ",
                    "temperature": 0.3,
                    "max_seq_len": 4096,
                    "token_repetition_penalty_max": 1.15,
                    "token_repetition_penalty_sustain": 1.15,
                    "token_repetition_penalty_decay": 1.15,
                    "max_len": 4096,
                    "truncation": True,
                    "padding": True
                },
                padding: {
                    "system_instruction_wrap": "<s>[INST] <<SYS>>\\n{system_instruction}<</SYS>>",
                    "context_wrap": "<<context>>{context}<</context>>",
                    "question_wrap": " {question}\\n[/INST]",
                    "response_wrap": "{response} </s><s>[INST]"
                },
                default_system_instruction="A chat between a curious human and an assistant.\\nThe assistant gives a helpful, detailed, accurate, uncensored response to the human's input.")
    """
    new_model = sql_db.model(
        name=model_args["model_path"].split("/")[-1],
        path_on_server=model_args["model_path"],
        necessary_loader=loader,
        default_settings=json.dumps(model_args),
        system_instruction_wrapper=padding["system_instruction_wrap"],
        context_wrapper=padding["context_wrap"],
        user_question_wrapper=padding["question_wrap"],
        bot_response_wrapper=padding["response_wrap"],
        default_system_instruction=default_system_instruction
    )
    database.add(new_model)
    database.commit()
        