import asyncio
# import logging

from typing import Annotated
import json, os
import uvicorn
from fastapi import FastAPI, File, UploadFile, APIRouter
from sse_starlette.sse import EventSourceResponse
from starlette.requests import Request
from fastapi.middleware.cors import CORSMiddleware
from QueryLake.model_manager import LLMEnsemble
from QueryLake import instruction_templates, encryption, sql_db, database_admin_operations
from fastapi.responses import StreamingResponse
# from QueryLake.sql_db import User, File, 
# from sqlmodel import Field, Session, SQLModel, create_engine, select
# from sqlmodel import Field, Session, SQLModel, create_engine, select
# from typing import Optional
from sqlmodel import Field, Session, SQLModel, create_engine, select
import time
import py7zr
import inspect
import re

from QueryLake import api


app = FastAPI()

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
if GLOBAL_SETTINGS["use_llama_cpp"]:
    from langchain.llms import LlamaCpp
    GLOBAL_SETTINGS["loader_class"] = LlamaCpp
else:
    from QueryLake import Exllama
    # from QueryLake import ExllamaV2
    GLOBAL_SETTINGS["loader_class"] = Exllama

GLOBAL_LLM_CONFIG = GLOBAL_SETTINGS["default_llm_config"]

ACTIVE_SESSIONS = {}

GlobalLLMEnsemble = LLMEnsemble(GLOBAL_LLM_CONFIG, GLOBAL_SETTINGS["loader_class"])
# engine = create_engine("sqlite:///database.db")
engine = create_engine("sqlite:///user_data.db", connect_args={"check_same_thread" : False})
SQLModel.metadata.create_all(engine)
# session_factory = sessionmaker(bind=engine)
# Session = scoped_session(session_factory)
database = Session(engine)

print("Model Loaded")

API_FUNCTIONS = [ pair[0] for pair in inspect.getmembers(api, inspect.isfunction)]
API_FUNCTIONS = [func for func in API_FUNCTIONS if (not re.match(r"__.*?__", func) and func not in api.excluded_member_function_descriptions)]
API_FUNCTION_DOCSTRINGS = [getattr(api, func).__doc__ for func in API_FUNCTIONS]
API_FUNCTION_PARAMETERS = [inspect.signature(getattr(api, func)) for func in API_FUNCTIONS]
API_FUNCTION_HELP_DICTIONARY = {}
API_FUNCTION_HELP_GUIDE = ""
for func in API_FUNCTIONS:
    arguments_list = list(inspect.signature(getattr(api, func)).parameters.items())
    function_argument_string = "(%s)" % (", ".join([str(pair[1]) for pair in arguments_list if str(pair[0]) != "database"]))
    function_docstring = "       "+re.sub(r"[\n|\t]+", str(getattr(api, func).__doc__), "\n").replace("\n", "\n\t\t\t").strip()
    API_FUNCTION_HELP_DICTIONARY[func] = {
        "arguments": function_argument_string,
        "description": function_docstring
    }
    API_FUNCTION_HELP_GUIDE += "%s %s\n\n\tDESCRIPTION: %s\n\n\n\n" % (func, function_argument_string, function_docstring)


@app.get("/api/async/chat")
async def chat(req: Request):
    """
    Returns Langchain generation via SSE
    """

    arguments = req.query_params._dict
    print("request:", arguments)
    if "query" in arguments:
        prompt = arguments["query"]
    else:
        prompt = "..."
    tmp_params = {
        "temperature": 0.2,
        "top_k": 50,
        # "max_seq_len": 4095,
        'token_repetition_penalty_max': 1.2,
        'token_repetition_penalty_sustain': 1,
        'token_repetition_penalty_decay': 1,
    }

    get_user_id = api.get_user_id(database, arguments["username"], arguments["password_prehash"])
    if get_user_id < 0:
        return EventSourceResponse("Invalid Authentication") # This response doesn't work. It ends the call obviously, but doesn't communicate it to the user.

    result = EventSourceResponse(GlobalLLMEnsemble.chain(
                                    user_name=arguments["username"],
                                    session_hash=arguments["session_hash"],
                                    database=database,
                                    question=prompt,
                                ))
    return result
    
@app.post("/api/async/upload_document")
async def create_document_collection(req: Request, file : UploadFile):
    # try:
    arguments = req.query_params._dict
    return api.upload_document_to_collection(database=database, **arguments, file=file)
    # except Exception as e:
    #     return {"success": False, "note": str(e)}


@app.get("/api/async/fetch_document")
async def fetch_document(req: Request):
    """
    Takes arguments: file hash id, username, password_prehash.
    """
    try:
        arguments = req.query_params._dict

        fetch_parameters = api.get_document_secure(database=database, **arguments)
        path=fetch_parameters["database_path"]
        password = fetch_parameters["password"]

        def yield_single_file():
            with py7zr.SevenZipFile(path, mode='r', password=password) as z:
                file = z.read()
                keys = list(file.keys())
                print(keys)
                file_name = keys[0]
                file = file[file_name]
                yield file.getbuffer().tobytes()
        return StreamingResponse(yield_single_file())
    except:
        return {"success": False}

@app.post("/api/{rest_of_path:path}")
def api_general_call(req: Request, rest_of_path: str):
    print("api called.")
    print(req.query_params._dict)
    try:
        arguments = req.query_params._dict
        route = req.scope['path']
        route_split = route.split("/")
        print("/".join(route_split[:3]))
        if "/".join(route_split[:3]) == "/api/help":
            if len(route_split) > 3:
                function_name = route_split[3]
                return {"success": True, "note": API_FUNCTION_HELP_DICTIONARY[function_name]}
            else:
                print(API_FUNCTION_HELP_GUIDE)
                return {"success": True, "note": API_FUNCTION_HELP_GUIDE}
        elif route == "/api/chat":
            return chat(req)
        elif route == "/api/fetch_document":
            return fetch_document(req)
        else:
            function_actual = getattr(api, rest_of_path)
            return function_actual(database, **arguments)
    except Exception as e:
        return {"success": False, "note": str(e)}

if __name__ == "__main__":
    # try:
    # database_admin_operations.add_llama2_to_db(database)
    # except:
    #     pass
    uvicorn.run(app, host="0.0.0.0", port=5000, log_level="trace", log_config=None)  # type: ignore