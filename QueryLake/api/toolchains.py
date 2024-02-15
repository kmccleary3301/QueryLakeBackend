import os, json
# from ..models.model_manager import LLMEnsemble
from .user_auth import get_user
from sqlmodel import Session, select, and_, not_
from sqlalchemy.sql.operators import is_
from ..database import sql_db_tables
from copy import deepcopy, copy
from time import sleep
from .hashing import random_hash
from ..toolchain_functions import toolchain_node_functions
from fastapi import UploadFile
from sse_starlette.sse import EventSourceResponse
import asyncio
from threading import Thread
from chromadb.api import ClientAPI
# from ..models.model_manager import LLMEnsemble
import time
from ..api.document import get_file_bytes, get_document_secure
from ..api.user_auth import get_user_private_key
from ..database.encryption import aes_encrypt_zip_file, aes_decrypt_zip_file
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi import WebSocket
import zipfile
from ..function_run_clean import run_function_safe
from ..typing.config import AuthType, getUserType
from typing import Callable, Any, List, Dict, Union
from ..typing.toolchains import *
from ..operation_classes.toolchain_session import ToolchainSession
from ..misc_functions.toolchain_state_management import safe_serialize

default_toolchain = "chat_session_normal"

server_dir = "/".join(os.path.dirname(os.path.realpath(__file__)).split("/")[:-2])
upper_server_dir = "/".join(os.path.dirname(os.path.realpath(__file__)).split("/")[:-2])+"/"
user_db_path = server_dir+"/user_db/files/"

async def save_toolchain_session(database : Session, 
                                 session : ToolchainSession,
                                 ws : WebSocket = None):
    """
    Commit toolchain session to SQL database.
    """
    # print("Saving session %s" % (session_id))
    # assert session_id in TOOLCHAIN_SESSION_CAROUSEL, "Toolchain Session not found"
    toolchain_data = session.dump()
    existing_session = database.exec(select(sql_db_tables.toolchain_session).where(sql_db_tables.toolchain_session.hash_id == session.session_hash
                                                                                   
                                                                                   )).first()
    existing_session.title = toolchain_data["title"]
    existing_session.state_arguments = json.dumps(toolchain_data["state_arguments"])
    existing_session.firing_queue = json.dumps(toolchain_data["firing_queue"])
    database.commit()
    if not ws is None:
        await session.send_websocket_msg({
            "type": "session_saved",
            "title": existing_session.title
        })
    
def retrieve_toolchain_from_db(database : Session,
                               toolchain_function_caller,
                               auth : AuthType,
                               session_id : str,
                               ws : WebSocket) -> ToolchainSession:
    user_retrieved : getUserType  = get_user(database, auth)
    (user, user_auth) = user_retrieved
    session_db_entry = database.exec(select(sql_db_tables.toolchain_session).where(sql_db_tables.toolchain_session.hash_id == session_id)).first()
    
    
    assert session_db_entry.author == auth["username"], "User not authorized"
    assert not session_db_entry is None, "Session not found" 
    
    toolchain_get = get_toolchain_from_db(database, session_db_entry.toolchain_id, auth)
    
    session = ToolchainSession(session_db_entry.toolchain_id, toolchain_get, toolchain_function_caller, session_db_entry.hash_id, user_auth.username)
    session.load({
        "title": session_db_entry.title,
        "toolchain_id": session_db_entry.toolchain_id,
        "state_arguments": json.loads(session_db_entry.state_arguments) if session_db_entry.state_arguments != "" else {},
        "session_hash_id": session_db_entry.hash_id,
        "firing_queue": json.loads(session_db_entry.firing_queue) if session_db_entry.firing_queue != "" else {}
    }, toolchain_get)
    return session

def get_available_toolchains(database : Session,
                             auth : AuthType):
    """
    Returns available toolchains with chat window settings and all.
    Will find organization locked
    If there are organization locked toolchains, they will be added to the database.
    """
    user_retrieved : getUserType  = get_user(database, auth)
    (user, user_auth) = user_retrieved
    
    toolchains_db : List[sql_db_tables.toolchain] = database.exec(select(sql_db_tables.toolchain)).all()
    
    toolchains_get : List[ToolChain] = [ToolChain(**json.loads(toolchain.content)) for toolchain in toolchains_db]
    
    default_toolchain_loaded : ToolChain = None
    
    toolchains_available_dict : Dict[str, List[dict]] = {}
    for t_i, toolchain in enumerate(toolchains_get):
        
        if toolchain.category not in toolchains_available_dict:
            toolchains_available_dict[toolchain.category] = []
        
        toolchains_available_dict[toolchain.category].append({
            "name": toolchain.name,
            "id": toolchain.id,
            "category": toolchain.category,
            "chat_window_settings": toolchain.display_configuration
        })
        
        if toolchain.id == default_toolchain:
            default_toolchain_loaded = toolchains_available_dict[toolchain.category][-1]
        
    result = {
        "toolchains": [{"category": key, "entries": value} for key, value in toolchains_available_dict.items()],
        "default": default_toolchain_loaded
    }
    return result

def get_toolchain_from_db(database: Session,
                          toolchain_id : str,
                          auth : AuthType):
    """
    TODO: Revisit this for permission locks.
    """
    toolchain_db : sql_db_tables.toolchain = database.exec(select(sql_db_tables.toolchain).where(sql_db_tables.toolchain.toolchain_id == toolchain_id)).first()
    return ToolChain(**json.loads(toolchain_db.content))
    

def create_toolchain_session(database : Session,
                             toolchain_function_caller : Callable[[], Callable],
                             auth : AuthType,
                             toolchain_id : str,
                             ws : WebSocket) -> ToolchainSession:
    """
    Initiate a toolchain session with a random access token.
    This token is the session ID, and the session will be stored in the
    database accordingly.
    """
    user_retrieved : getUserType  = get_user(database, auth)
    (user, user_auth) = user_retrieved
    session_hash = random_hash()
    # toolchain_get = TOOLCHAINS[toolchain_id]
    
    toolchain_get = get_toolchain_from_db(database, toolchain_id, auth)
    created_session = ToolchainSession(toolchain_id, toolchain_get, toolchain_function_caller, session_hash, user_auth.username)
    # return {"success": True, "session_id": session_hash}

    new_session_in_database = sql_db_tables.toolchain_session(
        title=created_session.state["title"],
        hash_id=session_hash,
        state_arguments=json.dumps(created_session.state),
        creation_timestamp=time.time(),
        toolchain_id=toolchain_id,
        author=user_auth.username
    )
    database.add(new_session_in_database)
    database.commit()
    
    return created_session

def fetch_toolchain_sessions(database : Session, 
                             auth : AuthType,
                             cutoff_date: float = None):
    """
    Get previous toolchain sessions of user. 
    Returned as a list of objects sorted by timestamp.
    
    Optional cutoff date provided in unix time.
    """

    user_retrieved : getUserType  = get_user(database, auth)
    (user, user_auth) = user_retrieved
    if not cutoff_date is None:
        condition = and_(sql_db_tables.toolchain_session.author == user_auth.username, 
                        not_(is_(sql_db_tables.toolchain_session.title, None)),
                        sql_db_tables.toolchain_session.hidden == False,
                        sql_db_tables.toolchain_session.creation_timestamp > cutoff_date)
    else:
        condition = and_(sql_db_tables.toolchain_session.author == user_auth.username, 
                        not_(is_(sql_db_tables.toolchain_session.title, None)),
                        sql_db_tables.toolchain_session.hidden == False)

    user_sessions = database.exec(select(sql_db_tables.toolchain_session).where(condition)).all()
    
    # print("sessions:", user_sessions)
    user_sessions = sorted(user_sessions, key=lambda x: x.creation_timestamp)
    return_sessions = []
    for session in user_sessions:
        return_sessions.append({
            "time": session.creation_timestamp,
            "title": session.title,
            "hash_id": session.hash_id
        })
    return {"sessions": return_sessions[::-1]}

def fetch_toolchain_session(database : Session,
                            toolchain_function_caller,
                            auth : AuthType,
                            session_id : str,
                            ws : WebSocket):
    """
    Retrieve toolchain session from session id.
    If not in memory, it is loaded from the database.
    """
    user_retrieved : getUserType  = get_user(database, auth)
    (user, user_auth) = user_retrieved
    return retrieve_toolchain_from_db(database, toolchain_function_caller, auth, session_id, ws)

def get_session_state(database : Session,
                      toolchain_function_caller,
                      auth: dict,
                      session_id : str,
                      session : ToolchainSession = None):
    """
    Get the session state of a given toolchain.
    """

    user = get_user(database, **auth)
    if session is None:
        session = retrieve_toolchain_from_db(database, toolchain_function_caller, auth, session_id)
    # session["last_activity"] = time.time()
    return {"success": True, "result": session.state_arguments}

def retrieve_files_for_session(database : Session,
                               session : ToolchainSession,
                               auth : AuthType):
    """
    TODO: revisit this.
    
    Retrieve uploaded files for a session, return them as a list of bytes objects.
    """
    user_retrieved : getUserType  = get_user(database, auth)
    (user, user_auth) = user_retrieved
    # session = retrieve_toolchain_from_db(database, toolchain_function_caller, toolchains_available, auth, session_id)
    assert session.author == user.name, "User not authorized"
    file_db_entries = database.exec(select(sql_db_tables.document_raw).where(sql_db_tables.document_raw.toolchain_session_hash_id == session.session_hash)).all()
    return [get_file_bytes(database, doc.hash_id, get_user_private_key(database, auth)["private_key"]) for doc in file_db_entries]

async def toolchain_file_upload_event_call(database : Session,
                                           session : ToolchainSession,
                                           auth : AuthType,
                                           session_id : str,
                                           event_parameters : dict,
                                           document_hash_id : str,
                                           file_name : str):
    """
    Trigger file upload event call in toolchain.
    """
    user_retrieved : getUserType  = get_user(database, auth)
    (user, user_auth) = user_retrieved
    assert session.author == user.name, "User not authorized"

    system_args = {
        "database": database,
    }
    system_args.update(auth)
    # TOOLCHAIN_SESSION_CAROUSEL[session_id]["last_activity"] = time.time()


    file_db_entry = database.exec(select(sql_db_tables.document_raw).where(and_(sql_db_tables.document_raw.toolchain_session_hash_id == session_id,
                                                                                sql_db_tables.document_raw.hash_id == document_hash_id
                                                                                ))).first()
    file_bytes = get_file_bytes(database, file_db_entry.hash_id, get_user_private_key(database, auth)["private_key"])
    
    event_parameters.update({
        "user_file": file_bytes,
        "file_name": file_name
    })
    result = await session.event_prop("user_file_upload_event", event_parameters, system_args)
    await save_toolchain_session(database, session)
    return result

# async def toolchain_entry_call(database : Session,
#                                session : ToolchainSession,
#                                auth : AuthType,
#                                session_id : str,
#                                entry_parameters : dict):
#     """
#     TODO: Revisit this. May deprecate.
    
#     Call entry point in toolchain and propagate forward.
#     entry parameters can be provided, however there must be special cases for
#     things like files.
#     """
#     user_retrieved : getUserType  = get_user(database, auth)
#     (user, user_auth) = user_retrieved
#     assert session.author == user.name, "User not authorized"
    

#     system_args = {
#         "database": database,
#     }
#     system_args.update(auth)
#     # return {"success": True, "result": await TOOLCHAIN_SESSION_CAROUSEL[session_id]["session"].entry_prop(entry_parameters)}
#     # TOOLCHAIN_SESSION_CAROUSEL[session_id]["last_activity"] = time.time()
#     save_to_db_flag = False
#     if not session.entry_called:
#         save_to_db_flag = True
#     result = await session.entry_prop(entry_parameters, system_args)
#     # if save_to_db_flag:
#     await save_toolchain_session(database, session)

#     return result

async def toolchain_event_call(database : Session,
                               session : ToolchainSession,
                               auth: AuthType,
                               session_id : str,
                               event_node_id : str,
                               event_parameters : dict,
                               ws : WebSocket = None,
                               return_file_response : bool = False):
    """
    Call an event node in provided toolchain session and propagate forward.
    Entry parameters can be provided, however there must be special cases for
    things like files.
    """
    user_retrieved : getUserType  = get_user(database, auth)
    (user, user_auth) = user_retrieved
    assert session.author == user.name, "User not authorized"
    
    system_args = {
        "database": database
    }
    system_args.update(auth)
    # TOOLCHAIN_SESSION_CAROUSEL[session_id]["last_activity"] = time.time()
    if event_node_id == "user_file_upload_event":
        file_db_entry = database.exec(select(sql_db_tables.document_raw).where(and_(
                                                                                    sql_db_tables.document_raw.toolchain_session_hash_id == session_id,
                                                                                    sql_db_tables.document_raw.hash_id == event_parameters["hash_id"]
                                                                                ))).first()
        
        document_values = get_document_secure(database, auth["username"], auth["password_prehash"], event_parameters["hash_id"])

        file_bytes = get_file_bytes(database, file_db_entry.hash_id, document_values["password"])

        event_parameters.update({
            "user_file": file_bytes,
        })

    
    event_parameters.update({"auth": auth})

    result = await session.event_prop(database,
                                      event_node_id, 
                                      event_parameters, 
                                      system_args,
                                      ws)
    
    await save_toolchain_session(database, session)
    await session.send_websocket_msg({
        "type": "finished_event_prop",
        "node_id": event_node_id
    }, ws)
    
    if return_file_response:
        assert "file_bytes" in result and "file_name" in result, "Output doesn't contain file bytes"
        file_name_hash, encryption_key = random_hash(), random_hash()
        save_dir = {}
        save_dir[result["file_name"]] = result["file_bytes"]
        file_zip_save_path = user_db_path+file_name_hash+".7z"
        aes_encrypt_zip_file(encryption_key, save_dir, file_zip_save_path)
        return {"flag": "file_response", "server_zip_hash": file_name_hash, "password": encryption_key, "file_name": result["file_name"]}
    
    return result

async def toolchain_session_notification(database : Session,
                                         toolchain_function_caller,
                                         session : ToolchainSession,
                                         auth : AuthType,
                                         session_id : str,
                                         message : dict,
                                         ws : WebSocket = None):
    user_retrieved : getUserType  = get_user(database, auth)
    (user, user_auth) = user_retrieved
    assert session.author == user.name, "User not authorized"
    await session.send_websocket_msg(message, ws)

async def get_toolchain_output_file_response(server_zip_hash : str, 
                                             document_password : str) -> FileResponse:
    file_zip_save_path = user_db_path+server_zip_hash+".7z"
    file = aes_decrypt_zip_file(document_password, file_zip_save_path)
    keys = list(file.keys())
    file_name = keys[0]
    file_get = file[file_name]
    temp_raw_path = user_db_path+file_name
    temp_ref = {}
    temp_ref[file_name] = file_get
    new_file_zip_save_path = user_db_path+server_zip_hash+".zip"
    # aes_encrypt_zip_file(None, temp_ref, new_file_zip_save_path)

    with zipfile.ZipFile(new_file_zip_save_path, mode="w") as myzip:
        with myzip.open(file_name, mode="w") as myfile:
            myfile.write(file_get.read())
            myfile.close()
        myzip.close()
    
    Thread(target=delete_file_on_delay, kwargs={"file_path": new_file_zip_save_path}).start()
    return FileResponse(new_file_zip_save_path)

def delete_file_on_delay(file_path : str):
    time.sleep(20)
    os.remove(file_path)
