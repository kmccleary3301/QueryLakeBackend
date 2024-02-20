import asyncio
import traceback
# import logging

from typing import Annotated, Callable, Any
import json, os
import ujson
import uvicorn
from fastapi import FastAPI, File, UploadFile, APIRouter, Request, WebSocket, Form
from starlette.requests import Request
from fastapi.responses import StreamingResponse
from sqlmodel import Session, SQLModel, create_engine, select
import time
# import py7zr
import inspect
import re
import chromadb
from copy import deepcopy
import time

from QueryLake.api import api
from QueryLake.database import database_admin_operations, encryption, sql_db_tables
from threading import Timer

from contextlib import asynccontextmanager
from fastapi.responses import StreamingResponse, FileResponse, Response
from starlette.testclient import TestClient

import re, json
from typing import AsyncGenerator, Optional, Literal, Tuple, Union, List, AsyncIterator, Dict, Awaitable
from pydantic import BaseModel

from fastapi import FastAPI, WebSocket, BackgroundTasks
from starlette.requests import Request
from starlette.responses import StreamingResponse, Response
from starlette.websockets import WebSocketDisconnect

from ray import serve
from ray.serve.handle import DeploymentHandle, DeploymentResponseGenerator

import openai

from QueryLake.typing.config import Config, AuthType, getUserType, Padding, ModelArgs, Model
from QueryLake.typing.toolchains import *


from QueryLake.operation_classes.toolchain_session import ToolchainSession
from QueryLake.operation_classes.ray_llm_class import LLMDeploymentClass
from QueryLake.operation_classes.ray_embedding_class import EmbeddingDeployment
from QueryLake.operation_classes.ray_reranker_class import RerankerDeployment
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware import Middleware
import asyncio

origins = [
    "http://localhost:5173",
    "http://localhost:5173/",
    "localhost:5173",
    "localhost:5173/",
    "0.0.0.0:5173"
]

middleware = [
    Middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=['*'],
        allow_headers=['*']
    )
]


fastapi_app = FastAPI(
    # lifespan=lifespan
    middleware=middleware
)

ACTIVE_SESSIONS = {}

global_public_key, global_private_key = encryption.ecc_generate_public_private_key()

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

def clean_function_arguments_for_api(system_args : dict, 
                                     user_args : dict, 
                                     function_name : str = None, 
                                     function_args = None, 
                                     bypass_disabled : bool = False,
                                     function_object : Callable = None) -> dict:
    synth_args = deepcopy(user_args)
    if not function_object is None:
        function_args = list(inspect.signature(function_object).parameters.items())
        function_args = [arg[0] for arg in function_args]
    elif not function_name is None and (function_name in API_FUNCTIONS or bypass_disabled) and function_args is None:
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

@serve.deployment
@serve.ingress(fastapi_app)
class UmbrellaClass:
    def __init__(self,
                 configuration: Config,
                 toolchains: Dict[str, ToolChain],
                 llm_handles: Dict[str, DeploymentHandle],
                 embedding_handle: DeploymentHandle,
                 rerank_handle: DeploymentHandle):
        
        self.config : Config = configuration
        self.toolchain_configs : Dict[str, ToolChain] = toolchains
        # self.llm_handle = llm_handle.options(
        #     # This is what enables streaming/generators. See: https://docs.ray.io/en/latest/serve/api/doc/ray.serve.handle.DeploymentResponseGenerator.html
        #     stream=True,
        # )
        self.llm_handles : Dict[str, DeploymentHandle] = { k : handle.options(
            # This is what enables streaming/generators. See: https://docs.ray.io/en/latest/serve/api/doc/ray.serve.handle.DeploymentResponseGenerator.html
            stream=True,
        ) for k, handle in llm_handles.items()}
        
        self.embedding_handle = embedding_handle
        self.rerank_handle = rerank_handle
        
        self.engine = create_engine("postgresql://admin:admin@localhost:5432/server_database")
        
        SQLModel.metadata.create_all(self.engine)
        self.database = Session(self.engine)
        
        database_admin_operations.add_models_to_database(self.database, self.config.models)
        database_admin_operations.add_toolchains_to_database(self.database, self.toolchain_configs)
        
        # all_users = self.database.exec(select(sql_db_tables.user)).all()
        # print("All users:")
        # print(all_users)
    
    async def embedding_call(self, 
                             auth : AuthType,
                             inputs : List[str]):
        (user, user_auth) = api.get_user(self.database, auth)
        return await self.embedding_handle.run.remote({"text": inputs})
    
    async def rerank_call(self, 
                          auth : AuthType,
                          inputs : List[Tuple[str, str]]):
        (user, user_auth) = api.get_user(self.database, auth)
        print("Calling rerank remote function")
        return await self.rerank_handle.run.remote({"text": inputs})
    
    # @fastapi_app.post("/direct/{rest_of_path:path}")
    async def text_models_callback(self, request_dict: dict, model_choice: Literal["embedding", "rerank"]):
        assert model_choice in ["embedding", "rerank"]
        if model_choice == "embedding":
            return_tmp = await self.embedding_handle.run.remote(request_dict)
        elif model_choice == "rerank":
            return_tmp = await self.rerank_handle.run.remote(request_dict)
        return return_tmp
    
    async def stream_results(self, 
                             results_generator: DeploymentResponseGenerator,
                             encode_output : bool = False,
                             on_new_token: Awaitable[Callable[[str], None]] = None) -> AsyncGenerator[bytes, None]:
        
        num_returned, tokens_returned = 0, []
        async for request_output in results_generator:
            text_outputs = [output.text for output in request_output.outputs]
            assert len(text_outputs) == 1
            text_output = text_outputs[0][num_returned:]
            ret = {"text": text_output}
            
            if not on_new_token is None:
                if inspect.iscoroutinefunction(on_new_token):
                    await on_new_token(text_output)
                else:
                    on_new_token(text_output)
                
            if encode_output:
                yield (json.dumps(ret) + "\n").encode("utf-8")
            else:
                yield text_output
            num_returned += len(text_output)
            tokens_returned.append(text_output)
        # return tokens_returned
    
    async def llm_call(self,
                       auth : AuthType, 
                       model_parameters : dict,
                       chat_history : List[dict] = None,
                       stream_callables: Dict[str, Awaitable[Callable[[str], None]]] = None):
        """
        Call an LLM model, possibly with parameters.
        
        TODO: Move OpenAI calls here for integration.
        TODO: Add optionality via default values to the model parameters.
        """
        (user, user_auth) = api.get_user(self.database, auth)
        assert "model_choice" in model_parameters, "Model choice not specified"
        model_choice = model_parameters.pop("model_choice")
        
        if not chat_history is None:
            model_parameters["chat_history"] = chat_history
        
        on_new_token = None
        if not stream_callables is None and "output" in stream_callables:
            on_new_token = stream_callables["output"]
        
        assert model_choice in self.llm_handles, "Model choice not available"
        
        llm_handle : DeploymentHandle = self.llm_handles[model_choice]
        gen: DeploymentResponseGenerator = (
            llm_handle.generator.remote(model_parameters)
        )
        return_stream_response = model_parameters.pop("stream_response_normal", False)
        if return_stream_response:
            return StreamingResponse(
                self.stream_results(gen, on_new_token=on_new_token, encode_output=True),
            )
        else:
            results = []
            async for result in self.stream_results(gen, on_new_token=on_new_token):
                results.append(result)
        
        # strings_to_remove = model_parameters["stop"] if "stop" in model_parameters else []
        
        text_outputs = "".join(results)
        
        # if len(strings_to_remove) > 0:
        #     # Create a regex pattern that matches any of the strings
        #     pattern = '|'.join(map(re.escape, strings_to_remove))
        #     pattern = re.compile(f"[{pattern}]+(.*?)$")
        #     # Text to remove strings from
        #     # Remove the strings
        #     text_outputs = re.sub(pattern, '', text_outputs)
        
        return {"output": text_outputs, "token_count": len(results)}
    
    async def openai_llm_call(self,
                              api_kwargs : dict,
                              model_choice: str,
                              chat_history : List[dict],
                              model_parameters : dict,
                              on_new_token: Callable[[str], None] = None):
        # openai.Completion
        auth_args = {"api_key": api_kwargs["api_key"]}
        if "organization_id" in api_kwargs:
            auth_args["organization"] = api_kwargs["organization_id"]
        # client = openai.OpenAI(**api_kwargs)
        response = ""
        for chunk in openai.ChatCompletion.create(
            model=model_choice,
            messages=chat_history,
            **model_parameters,
            stream=True,
        ): 
            try:
                content = chunk.choices[0].delta.content
                if not content is None:
                    response.append(content)
                    if not on_new_token is None:
                        if inspect.iscoroutinefunction(on_new_token):
                            await on_new_token(content)
                        else:
                            on_new_token(content)
            except:
                pass
        return {"output": "".join(response), "token_count": len(response)}
    
    def api_function_getter(self, function_name):
        if function_name == "llm":
            # return self.run_llm_new
            return self.llm_call
        elif function_name == "text_models_callback":
            return self.text_models_callback
        elif function_name == "embedding":
            return self.embedding_call
        elif function_name == "rerank":
            return self.rerank_call
        
        assert function_name in API_FUNCTIONS, f"Invalid API Function '{function_name}' Called"
        return getattr(api, function_name)
    
    @fastapi_app.post("/upload_document")
    async def upload_document_new(self, req : Request, file : UploadFile):
        try:
            # arguments = req.query_params._dict
            
            
            # print(req.__dict__)
            # route = req.scope['path']
            # route_split = route.split("/")
            # print("/".join(route_split[:4]), req.query_params._dict)
            arguments = json.loads(req.query_params._dict["parameters"]) 
            # arguments = await req.json()
            # arguments = json.loads(data) if data else {}
            true_arguments = clean_function_arguments_for_api({
                "database": self.database,
                "toolchain_function_caller": self.api_function_getter,
                "file": file,
            }, arguments, "upload_document")

            return await api.upload_document(**true_arguments)
        except Exception as e:
            error_message = str(e)
            stack_trace = traceback.format_exc()
            return_msg = {"success": False, "note": error_message, "trace": stack_trace}
            print(return_msg)
            return return_msg

    @fastapi_app.get("/fetch_document")
    async def get_document_2(self, req: Request):
        try:
            if "parameters" in req.query_params._dict:
                arguments = json.loads(req.query_params._dict["parameters"])
            else:
                arguments = await req.json()
            
            function_actual = getattr(api, "fetch_document")
            true_args = clean_function_arguments_for_api({
                "database": self.database,
                "text_models_callback": self.text_models_callback,
                "public_key": global_public_key,
                "server_private_key": global_private_key,
                "toolchain_function_caller": self.api_function_getter,
                "global_config": self.config,
            }, arguments, "fetch_document")
            
            print("Created args:", true_args)
            
            # Check if function is async
            if inspect.iscoroutinefunction(function_actual):
                args_get = await function_actual(**true_args)
            else:
                args_get = function_actual(**true_args)
            
            return args_get
        except Exception as e:
            error_message = str(e)
            stack_trace = traceback.format_exc()
            return_dict = {"success": False, "note": error_message, "trace": stack_trace}
            print(json.dumps(return_dict, indent=4))
            return return_dict
    
    @fastapi_app.get("/ping")
    async def ping_function(self, req: Request):
        print("GOT PING!!!")
        return {"success": True, "note": "Pong"}
    
    # Callable by POST and GET
    @fastapi_app.post("/api/{rest_of_path:path}")
    @fastapi_app.get("/api/{rest_of_path:path}")
    async def api_general_call(self, req: Request, rest_of_path: str):
        """
        This is a wrapper around every api function that is allowed. 
        It will call the function with the arguments provided, after filtering them for security.
        """
        
        try:
            
            # body = await req.body()
            # print("Got request with body:", body)
            # arguments = req.query_params._dict
            print("Calling:", rest_of_path)
            if "parameters" in req.query_params._dict:
                arguments = json.loads(req.query_params._dict["parameters"])
            else:
                # We use ujson because normal `await req.json()` completely stalls on large inputs.
                # print("Awaiting JSON")
                
                arguments = await asyncio.wait_for(req.json(), timeout=10)
                
                
            # print("arguments:", arguments)
            route = req.scope['path']
            route_split = route.split("/")
            print("/".join(route_split[:3]), req.query_params._dict)
            if rest_of_path == "help":
                if len(route_split) > 3:
                    function_name = route_split[3]
                    return {"success": True, "note": API_FUNCTION_HELP_DICTIONARY[function_name]}
                else:
                    print(API_FUNCTION_HELP_GUIDE)
                    return {"success": True, "note": API_FUNCTION_HELP_GUIDE}
            else:
                function_actual = self.api_function_getter(rest_of_path)
                true_args = clean_function_arguments_for_api({
                    "database": self.database,
                    "text_models_callback": self.text_models_callback,
                    "toolchains_available": self.toolchain_configs,
                    "public_key": global_public_key,
                    "server_private_key": global_private_key,
                    "toolchain_function_caller": self.api_function_getter,
                    "global_config": self.config,
                }, arguments, function_object=function_actual)
                
                if inspect.iscoroutinefunction(function_actual):
                    args_get = await function_actual(**true_args)
                else:
                    args_get = function_actual(**true_args)
                
                print("Type of args_get:", type(args_get))
                
                if type(args_get) is StreamingResponse:
                    return args_get
                elif type(args_get) is FileResponse:
                    return args_get
                elif type(args_get) is Response:
                    return args_get
                elif args_get is True:
                    return {"success": True}
                return {"success": True, "result": args_get}
        except Exception as e:
            error_message = str(e)
            stack_trace = traceback.format_exc()
            return_dict = {"success": False, "note": error_message, "trace": stack_trace}
            print("RETURNING:", json.dumps(return_dict, indent=4))
            return return_dict
            # return {"success": False, "note": str(e)}
    
    @fastapi_app.websocket("/toolchain")
    async def toolchain_websocket_handler(self, ws: WebSocket):
        """
        Toolchain websocket API point.
        On connection, there is no session.
        All client messages must decode as a JSON with the fields `command`, `arguments`, and `auth`.
        The `auth` field is a dictionary with the fields `username` and `password_prehash`.
        The client can send the following to load an existing toolchain:
        ```json
        {
            "command": "toolchain/load",
            "arguments": {
                "session_id": "..."
            }
        }
        ```
        Or, the client can send the following to create one:
        ```json
        {
            "command": "toolchain/create",
            "arguments": {
                "toolchain_id": "..."
            }
        }
        ```
        Some other commands include:
        * `toolchain/retrieve_files` - Retrieve files belonging to the toolchain session.
        * `toolchain/file_upload_event_call` - Call the file upload event node of the toolchain.
        * `toolchain/entry` - Call the entry node of the toolchain.
        * `toolchain/event` - Call an event node of the toolchain.
        
        All session state updates will be sent back to the client.
        
        For nodes with stream output, the websocket will send back a mapping id to the state variable/index.
        The client will then be sent each piece of the generated output as a json, which also includes the mapping id.
        
        For file uploads, the client must make a separate POST request, then do the following:
        On upload completion, the file will be added to the database, where the client must then send
        the new id to the websocket via `toolchain/file_upload_event_call`.
        """
        
        system_args = {
            "database": self.database,
            "toolchains_available": self.toolchain_configs,
            "text_models_callback": self.text_models_callback,
            "public_key": global_public_key,
            "server_private_key": global_private_key,
            "toolchain_function_caller": self.api_function_getter,
            "global_config": self.config,
            "ws": ws,
        }
        
        await ws.accept()
        
        toolchain_session : ToolchainSession = None
        
        await ws.send_text((json.dumps({"success": True})).encode("utf-8"))
        
        try:
            while True:
                text = await ws.receive_text()
                # print("Got text:", text)
                try:
                    arguments_websocket = json.loads(text)
                    assert "auth" in arguments_websocket, "No auth provided"
                    assert "command" in arguments_websocket, "No command provided"
                    command : str = arguments_websocket["command"]
                    auth : AuthType = arguments_websocket["auth"]
                    arguments : dict = arguments_websocket["arguments"]
                    
                    arguments.update({"auth": auth})
                    
                    assert command in [
                        "toolchain/load",
                        "toolchain/create",
                        # "toolchain/retrieve_files",
                        "toolchain/file_upload_event_call",
                        "toolchain/entry",
                        "toolchain/event",
                    ], "Invalid command"
                    
                    (user, user_auth) = api.get_user(self.database, auth)
                    
                    result_message = {}
                    
                    # Make sure session is in system args
                    # if not toolchain_session is None and not "session" in system_args:
                    #     system_args["session"] = toolchain_session
                    # elif toolchain_session is None and "session" in system_args:
                    #     del system_args["session"]
                    
                    if command == "toolchain/load":
                        if not toolchain_session is None:
                            await api.save_toolchain_session(self.database, toolchain_session)
                            toolchain_session = None
                        true_args = clean_function_arguments_for_api(system_args, arguments, function_object=api.fetch_toolchain_session)
                        toolchain_session : ToolchainSession = api.fetch_toolchain_session(**true_args)
                        # system_args["session"] = toolchain_session
                    
                    elif command == "toolchain/create":
                        if not toolchain_session is None:
                            api.save_toolchain_session(self.database, toolchain_session)
                            toolchain_session = None
                        true_args = clean_function_arguments_for_api(system_args, arguments, function_object=api.create_toolchain_session)
                        toolchain_session : ToolchainSession = api.create_toolchain_session(**true_args)
                        result = {
                            "success": True,
                            "toolchain_session_id": toolchain_session.session_hash,
                            "state": toolchain_session.state,
                        }
                    
                    elif command == "toolchain/file_upload_event_call":
                        true_args = clean_function_arguments_for_api(system_args, arguments, function_object=api.toolchain_file_upload_event_call)
                        result = await api.toolchain_file_upload_event_call(**true_args, session=toolchain_session)
                    
                    # Entries are deprecated.
                    # elif command == "toolchain/entry":
                    #     true_args = clean_function_arguments_for_api(system_args, arguments, function_object=api.toolchain_entry_call)
                    #     result = await api.toolchain_entry_call(**true_args, session=toolchain_session)

                    elif command == "toolchain/event":
                        true_args = clean_function_arguments_for_api(system_args, arguments, function_object=api.toolchain_event_call)
                        event_result = await api.toolchain_event_call(**true_args, session=toolchain_session)
                        result = {"event_result": event_result}
                    
                    await ws.send_text((json.dumps(result)).encode("utf-8"))
                    await ws.send_text((json.dumps({"ACTION": "END_WS_CALL"})).encode("utf-8"))
                    
                    del result_message
                    del result
                    
                    await api.save_toolchain_session(self.database, toolchain_session)
                
                except WebSocketDisconnect:
                    raise WebSocketDisconnect
                except Exception as e:
                    error_message = str(e)
                    stack_trace = traceback.format_exc()
                    await ws.send_text(json.dumps({"error": error_message, "trace": stack_trace}))
                    await ws.send_text((json.dumps({"ACTION": "END_WS_CALL"})).encode("utf-8"))
        except WebSocketDisconnect as e:
            print("Websocket disconnected")
            if not toolchain_session is None:
                print("Unloading Toolchain")
                toolchain_session.write_logs()
                toolchain_session = None
                del toolchain_session
    
os.chdir(os.path.dirname(os.path.realpath(__file__)))

GLOBAL_CONFIG, TOOLCHAINS = Config.parse_file('config.json'), {}

default_toolchain = "chat_session_normal"

toolchain_files_list = os.listdir("toolchains")
for toolchain_file in toolchain_files_list:
    if not toolchain_file.split(".")[-1] == "json":
        continue
    with open("toolchains/"+toolchain_file, 'r', encoding='utf-8') as f:
        toolchain_retrieved = json.loads(f.read())
        f.close()
    TOOLCHAINS[toolchain_retrieved["id"]] = ToolChain(**toolchain_retrieved)

LOCAL_MODEL_BINDINGS : Dict[str, DeploymentHandle] = {}
for model_entry in GLOBAL_CONFIG.models:
    LOCAL_MODEL_BINDINGS[model_entry.id] = LLMDeploymentClass.bind(
        model_config=model_entry,
        model=model_entry.system_path, 
        max_model_len=model_entry.max_model_len, 
        quantization=model_entry.quantization
    )


deployment = UmbrellaClass.bind(
    configuration=GLOBAL_CONFIG,
    toolchains=TOOLCHAINS,
    llm_handles=LOCAL_MODEL_BINDINGS,
    embedding_handle=EmbeddingDeployment.bind(model_key="/home/kyle_m/QueryLake_Development/alt_ai_models/bge-large-en-v1.5"),
    rerank_handle=RerankerDeployment.bind(model_key="/home/kyle_m/QueryLake_Development/alt_ai_models/bge-reranker-large")
)

if __name__ == "__main__":
    serve.run(deployment)
