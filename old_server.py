import asyncio
# import logging

from typing import Annotated
import json, os
import uvicorn
from fastapi import FastAPI, File, UploadFile, APIRouter, Request, WebSocket
from sse_starlette.sse import EventSourceResponse
from starlette.requests import Request
from fastapi.middleware.cors import CORSMiddleware
# from QueryLake.models.model_manager import LLMEnsemble
from QueryLake import instruction_templates
from fastapi.responses import StreamingResponse
# from QueryLake.sql_db import User, File, 
# from sqlmodel import Field, Session, SQLModel, create_engine, select
# from sqlmodel import Field, Session, SQLModel, create_engine, select
# from typing import Optional
from sqlmodel import Session, SQLModel, create_engine, select
import time
# import py7zr
import inspect
import re
import chromadb
from copy import deepcopy
import time

from QueryLake.models.langchain_sse import ErrorAsGenerator
from QueryLake.api import api
from QueryLake.database import database_admin_operations, encryption, sql_db_tables
from QueryLake.toolchain_functions import template_functions
from QueryLake.models.langchain_sse import ThreadedGenerator
from threading import Timer
# from QueryLake.api import toolchains

# import httpx
# from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
from fastapi.responses import StreamingResponse, FileResponse, Response
from starlette.testclient import TestClient

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     app.requests_client = httpx.AsyncClient()
#     yield
#     await app.requests_client.aclose()


app = FastAPI(
    # lifespan=lifespan
)

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.chdir(os.path.dirname(os.path.realpath(__file__)))
with open("config.json", 'r', encoding='utf-8') as f:
    file_read = f.read()
    f.close()
GLOBAL_SETTINGS = json.loads(file_read)
# if GLOBAL_SETTINGS["use_llama_cpp"]:
#     from langchain.llms import LlamaCpp
#     GLOBAL_SETTINGS["loader_class"] = LlamaCpp
# else:
#     from QueryLake import Exllama
#     # from QueryLake import ExllamaV2
#     GLOBAL_SETTINGS["loader_class"] = Exllama

# GLOBAL_LLM_CONFIG = GLOBAL_SETTINGS["default_llm_config"]

ACTIVE_SESSIONS = {}

# engine = create_engine("sqlite:///database.db")
engine = create_engine("sqlite:///user_data.db", connect_args={"check_same_thread" : False})
SQLModel.metadata.create_all(engine)
# session_factory = sessionmaker(bind=engine)
# Session = scoped_session(session_factory)
database = Session(engine)
vector_database = chromadb.PersistentClient(path="vector_database")
database_admin_operations.add_models_to_database(database, GLOBAL_SETTINGS["models"])
# GlobalLLMEnsemble = LLMEnsemble(GLOBAL_LLM_CONFIG, GLOBAL_SETTINGS["loader_class"])
# GlobalLLMEnsemble = LLMEnsemble(database, GLOBAL_SETTINGS["default_model"], GLOBAL_SETTINGS)
global_public_key, global_private_key = encryption.ecc_generate_public_private_key()


print("Model Loaded")

TEMPLATE_FUNCTIONS = [pair[0] for pair in inspect.getmembers(template_functions, inspect.isfunction)]

API_FUNCTIONS = [pair[0] for pair in inspect.getmembers(api, inspect.isfunction)]
API_FUNCTIONS = [func for func in API_FUNCTIONS if (not re.match(r"__.*?__", func) and func not in api.excluded_member_function_descriptions)]
API_FUNCTION_DOCSTRINGS = [getattr(api, func).__doc__ for func in API_FUNCTIONS]
API_FUNCTION_PARAMETERS = [inspect.signature(getattr(api, func)) for func in API_FUNCTIONS]
API_FUNCTION_HELP_DICTIONARY = {}
API_FUNCTION_HELP_GUIDE = ""
for func in API_FUNCTIONS:
    arguments_list = list(inspect.signature(getattr(api, func)).parameters.items())
    function_argument_string = "(%s)" % (", ".join([str(pair[1]) for pair in arguments_list if str(pair[0]) not in api.system_arguments]))
    function_docstring = "       "+re.sub(r"[\n|\t]+", str(getattr(api, func).__doc__), "\n").replace("\n", "\n\t\t\t").strip()
    API_FUNCTION_HELP_DICTIONARY[func] = {
        "arguments": function_argument_string,
        "description": function_docstring
    }
    API_FUNCTION_HELP_GUIDE += "%s %s\n\n\tDESCRIPTION: %s\n\n\n\n" % (func, function_argument_string, function_docstring)


def clean_function_arguments_for_api(system_args : dict, user_args : dict, function_name : str = None, function_args = None, bypass_disabled : bool = False) -> dict:
    synth_args = deepcopy(user_args)
    # if function_name is None:


    if not function_name is None and (function_name in API_FUNCTIONS or bypass_disabled) and function_args is None:
        function_args = list(inspect.signature(getattr(api, function_name)).parameters.items())
        function_args = [arg[0] for arg in function_args]
    keys_get = list(synth_args.keys())

    print("Cleaning args with", function_args, "and", keys_get)
    for key in keys_get:
        if key in system_args or (not function_args is None and not key in function_args):
            del synth_args[key]
    for key in system_args.keys():
        if function_args is None or key in function_args:
            synth_args[key] = system_args[key]
    return synth_args

def toolchain_function_caller(function_name):
    assert function_name in API_FUNCTIONS or function_name in TEMPLATE_FUNCTIONS, "Function not available"
    if function_name in API_FUNCTIONS:
        return getattr(api, function_name)
    return getattr(template_functions, function_name)

@app.post("/api/async/upload_document")
async def create_document_collection(req: Request, file : UploadFile):
    # try:
    # arguments = req.query_params._dict
    route = req.scope['path']
    route_split = route.split("/")
    print("/".join(route_split[:4]), req.query_params._dict)
    arguments = json.loads(req.query_params._dict["parameters"]) 
    true_arguments = clean_function_arguments_for_api({
        "database": database,
        "vector_database": vector_database,
        "file": file,
    }, arguments, "upload_document")

    return api.upload_document(**true_arguments)

@app.post("/api/async/upload_document_to_session")
async def create_document_collection(req: Request, file : UploadFile):
    # try:
    # arguments = req.query_params._dict
    route = req.scope['path']
    route_split = route.split("/")
    print("/".join(route_split[:4]), req.query_params._dict)
    arguments = json.loads(req.query_params._dict["parameters"]) 
    true_arguments = clean_function_arguments_for_api({
        "database": database,
        "vector_database": vector_database,
        "file": file,
        "return_file_hash": True,
        "add_to_vector_db": False
    }, arguments, "upload_document")

    upload_result = api.upload_document(**true_arguments)

    true_args_2 = clean_function_arguments_for_api({
        "database": database,
        "vector_database": vector_database,
        "llm_ensemble": GlobalLLMEnsemble,
        "public_key": global_public_key,
        "server_private_key": global_private_key,
        "toolchain_function_caller": toolchain_function_caller,
        "message": {
            "type": "file_uploaded",
            "hash_id": upload_result["hash_id"],
            "file_name": upload_result["file_name"]
        }
    }, arguments, "toolchain_session_notification", bypass_disabled=True)
    function_actual = getattr(api, "toolchain_session_notification")
    args_get = await function_actual(**true_args_2)
    if args_get is True:
        return {"success": True}
    return {"success": True, "result": args_get}


@app.get("/api/async/{rest_of_path:path}")
async def api_general_call_async(req: Request, rest_of_path: str):
    arguments = json.loads(req.query_params._dict["parameters"]) if "parameters" in req.query_params._dict else {}
    route = req.scope['path']

    route_split = route.split("/")
    path_split = rest_of_path.split("/")
    function_target = path_split[0]

    print("/".join(route_split[:4]), req.query_params._dict)
    
    assert function_target in API_FUNCTIONS, "Invalid API Function Called"
    assert function_target in api.async_member_functions, "Synchronous function called. Try api/"+function_target
    function_actual = getattr(api, function_target)
    true_args = clean_function_arguments_for_api({
        "database": database,
        "vector_database": vector_database,
        "llm_ensemble": GlobalLLMEnsemble,
        "public_key": global_public_key,
        "server_private_key": global_private_key,
        "toolchain_function_caller": toolchain_function_caller,
    }, arguments, function_target)
    call_result = await function_actual(**true_args)
    if type(call_result) is ThreadedGenerator:
        return EventSourceResponse(call_result)
    elif type(call_result) is StreamingResponse:
        return call_result
    elif type(call_result) is FileResponse:
        return call_result
    elif type(call_result) is Response:
        return call_result
    if call_result is True:
        return {"success": True}
    return {"success": True, "result": call_result}
    # except Exception as e:
    #     return ErrorAsGenerator(str(e))

# @app.get("/api/{rest_of_path:path}")
@app.post("/api/{rest_of_path:path}")
async def api_general_call(req: Request, rest_of_path: str):
    try:
        # arguments = req.query_params._dict
        print("Calling:", rest_of_path)
        arguments = json.loads(req.query_params._dict["parameters"]) if "parameters" in req.query_params._dict else {}
        route = req.scope['path']
        route_split = route.split("/")
        print("/".join(route_split[:3]), req.query_params._dict)
        if "/".join(route_split[:3]) == "/api/help":
            if len(route_split) > 3:
                function_name = route_split[3]
                return {"success": True, "note": API_FUNCTION_HELP_DICTIONARY[function_name]}
            else:
                print(API_FUNCTION_HELP_GUIDE)
                return {"success": True, "note": API_FUNCTION_HELP_GUIDE} 
        else:
            assert rest_of_path in API_FUNCTIONS, "Invalid API Function Called"
            assert not rest_of_path in api.async_member_functions, "Async function called. Try api/async/"+rest_of_path
            function_actual = getattr(api, rest_of_path)
            true_args = clean_function_arguments_for_api({
                "database": database,
                "vector_database": vector_database,
                "llm_ensemble": GlobalLLMEnsemble,
                "public_key": global_public_key,
                "toolchain_function_caller": toolchain_function_caller
            }, arguments, rest_of_path)
            args_get = function_actual(**true_args)
            if args_get is True:
                return {"success": True}
            return {"success": True, "result": args_get}
    except Exception as e:
        return {"success": False, "note": str(e)}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    data = await websocket.receive_text()
    await websocket.send_text(f"Message text was: {data}")

if __name__ == "__main__":
    print(API_FUNCTION_HELP_GUIDE)
    Timer(30, api.prune_inactive_toolchain_sessions, args=(database, 240)) # Doesn't work, why??????
    uvicorn.run(app, host="0.0.0.0", port=5000, log_level="trace", log_config=None)  # type: ignore
