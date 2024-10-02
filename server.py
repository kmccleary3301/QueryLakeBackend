import asyncio
import traceback
# import logging

from typing import Annotated, Callable, Any
import json, os
from fastapi import FastAPI, File, UploadFile, APIRouter, Request, WebSocket, Form
from starlette.requests import Request
from sqlmodel import Session, SQLModel, create_engine, select
import inspect
import re
from copy import deepcopy

from QueryLake.api import api
from QueryLake.database import database_admin_operations, encryption, sql_db_tables
from QueryLake.database.create_db_session import initialize_database_engine
from threading import Timer

from contextlib import asynccontextmanager
from fastapi.responses import StreamingResponse, FileResponse, Response
from starlette.background import BackgroundTask
from starlette.testclient import TestClient

import re, json
from typing import AsyncGenerator, Optional, Literal, Tuple, Union, List, AsyncIterator, Dict, Awaitable
from pydantic import BaseModel

from fastapi import FastAPI, WebSocket, BackgroundTasks
from starlette.requests import Request
from starlette.websockets import WebSocketDisconnect

from ray import serve, get
from ray.serve.handle import DeploymentHandle, DeploymentResponseGenerator


from QueryLake.typing.config import Config, AuthType, getUserType, Padding, ModelArgs, Model
from QueryLake.typing.toolchains import *
from QueryLake.operation_classes.toolchain_session import ToolchainSession
from QueryLake.operation_classes.ray_vllm_class import VLLMDeploymentClass
from QueryLake.operation_classes.ray_exllamav2_class import ExllamaV2DeploymentClass
from QueryLake.operation_classes.ray_embedding_class import EmbeddingDeployment
from QueryLake.operation_classes.ray_reranker_class import RerankerDeployment
from QueryLake.operation_classes.ray_web_scraper import WebScraperDeployment
from QueryLake.misc_functions.function_run_clean import get_function_call_preview, get_function_specs
from QueryLake.typing.function_calling import FunctionCallDefinition

from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware import Middleware
import asyncio

from psycopg2.errors import InFailedSqlTransaction

from fastapi import FastAPI, HTTPException
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.security.oauth2 import OAuth2PasswordRequestFormStrict
from fastapi_login import LoginManager
from fastapi_login.exceptions import InvalidCredentialsException
# from passlib.context import CryptContext
from fastapi import Depends

from QueryLake.api.single_user_auth import global_public_key, global_private_key
from QueryLake.misc_functions.external_providers import external_llm_generator
from QueryLake.misc_functions.server_class_functions import stream_results_tokens, find_function_calls

from asyncio import gather

origins = [
    "http://localhost:3001",
    "localhost:3001",
    "0.0.0.0:3001"
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

API_FUNCTIONS = [str(pair[0]) for pair in inspect.getmembers(api, inspect.isfunction)]
API_FUNCTIONS_ALLOWED = list(set(API_FUNCTIONS) - set(api.excluded_member_function_descriptions))
API_FUNCTIONS_ALLOWED = [func for func in API_FUNCTIONS_ALLOWED if (not re.match(r"__.*?__", func))]

API_FUNCTIONS = [func for func in API_FUNCTIONS_ALLOWED]
API_FUNCTION_HELP_DICTIONARY, API_FUNCTION_HELP_GUIDE = {}, ""
for func in API_FUNCTIONS:
    if not hasattr(api, func):
        continue
    
    API_FUNCTION_HELP_DICTIONARY[func] = {
        "result": get_function_call_preview(getattr(api, func), api.system_arguments),
    }
    API_FUNCTION_HELP_GUIDE += "%s\n\n\n\n" % get_function_call_preview(getattr(api, func), api.system_arguments)

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

@fastapi_app.get("/api/python")
def hello_world():
    return {"message": "Hello World"}

@serve.deployment
@serve.ingress(fastapi_app)
class UmbrellaClass:
    def __init__(self,
                 configuration: Config,
                 toolchains: Dict[str, ToolChain],
                 web_scraper_handle: DeploymentHandle,
                 llm_handles: Dict[str, DeploymentHandle] = {},
                 embedding_handles: Dict[str, DeploymentHandle] = None,
                 rerank_handles: Dict[str, DeploymentHandle] = None,
                 ):
        
        print("INITIALIZING UMBRELLA CLASS")
        self.config : Config = configuration
        self.toolchain_configs : Dict[str, ToolChain] = toolchains
        
        self.llm_handles_no_stream = llm_handles
        self.llm_handles : Dict[str, DeploymentHandle] = { k : handle.options(
            # This is what enables streaming/generators. See: https://docs.ray.io/en/latest/serve/api/doc/ray.serve.handle.DeploymentResponseGenerator.html
            stream=True,
        ) for k, handle in llm_handles.items()} if self.config.enabled_model_classes.llm else {}
        self.llm_configs : Dict[str, Model] = { config.id : config for config in self.config.models }
        
        self.embedding_handles = embedding_handles
        self.rerank_handles = rerank_handles
        self.web_scraper_handle = web_scraper_handle
        
        self.database, self.engine = initialize_database_engine()
        
        database_admin_operations.add_models_to_database(self.database, self.config.models)
        database_admin_operations.add_toolchains_to_database(self.database, self.toolchain_configs)
        
        self.default_function_arguments = {
            "database": self.database,
            "text_models_callback": self.text_models_callback,
            "toolchains_available": self.toolchain_configs,
            "public_key": global_public_key,
            "server_private_key": global_private_key,
            "toolchain_function_caller": self.api_function_getter,
            "global_config": self.config,
        }
        
        self.special_function_table = {
            "llm": self.llm_call,
            "llm_count_tokens": self.llm_count_tokens,
            "text_models_callback": self.text_models_callback,
            "embedding": self.embedding_call,
            "rerank": self.rerank_call,
            "find_function_calls": find_function_calls,
            "web_scrape": self.web_scrape_call,
            "function_help": self.get_all_function_descriptions,
        }
        
        self.all_functions = API_FUNCTIONS_ALLOWED+list(self.special_function_table.keys())
        self.all_function_descriptions = [
            {
                "endpoint": f"/api/{func_name}",
                "api_function_id": func_name,
                **get_function_specs(
                    self.api_function_getter(func_name),
                    excluded_arguments=list(self.default_function_arguments.keys()),
                    querylake_auth_sub=True
                )
            }
            for func_name in self.all_functions
        ]
        
        print("DONE INITIALIZING UMBRELLA CLASS")
        print("CONFIG ENABLED MODELS:", self.config.enabled_model_classes)
        # all_users = self.database.exec(select(sql_db_tables.user)).all()
        # print("All users:")
        # print(all_users)
    
    @fastapi_app.post("/get_auth_token")
    async def login(self, form_data: Annotated[OAuth2PasswordRequestForm, Depends()]):
        try:
            arguments = {"username": form_data.username, "password": form_data.password}
            
            true_args = clean_function_arguments_for_api(
                self.default_function_arguments, 
                arguments, 
                function_object=api.create_oauth2_token
            )
            return api.create_oauth2_token(**true_args)
        except Exception as e:
            if isinstance(e, InFailedSqlTransaction):
                self.database.rollback()
            error_message = str(e)
            stack_trace = traceback.format_exc()
            return {"success": False, "error": error_message, "trace": stack_trace}
    
    async def embedding_call(self, 
                             auth : AuthType,
                             inputs : List[str],
                             model: str = None):
        assert self.config.enabled_model_classes.embedding, "Embedding models are disabled on this QueryLake Deployment"
        if model is None:
            model = self.config.default_models.embedding
        assert model in self.embedding_handles, "Model choice not available"
        (user, user_auth, original_auth, auth_type) = api.get_user(self.database, auth, return_auth_type=True)
        result = await self.embedding_handles[model].run.remote({"text": inputs})
        
        if isinstance(result, list):
            embedding = [e["embedding"] for e in result]
            total_tokens = sum([e["token_count"] for e in result])
        else:
            embedding = result["embedding"]
            total_tokens = result["token_count"]
        
        api.increment_usage_tally(self.database, user_auth, {
            "embedding": {
                self.config.default_models.embedding: {"tokens": total_tokens}
            }
        }, **({"api_key_id": original_auth} if auth_type == 2 else {}))
        
        return embedding
    
    async def rerank_call(self, 
                          auth : AuthType,
                          inputs : Union[List[Tuple[str, str]], Tuple[str, str]],
                          normalize : Union[bool, List[bool]] = True,
                          model: str = None):
        assert self.config.enabled_model_classes.rerank, "Rerank models are disabled on this QueryLake Deployment"
        if model is None:
            model = self.config.default_models.rerank
        assert model in self.rerank_handles, "Model choice not available"
        (user, user_auth, original_auth, auth_type) = api.get_user(self.database, auth, return_auth_type=True)
        
        if isinstance(inputs, list):
            if not isinstance(normalize, list):
                normalize = [normalize for _ in range(len(inputs))]
            assert len(normalize) == len(inputs), \
                "All input lists must be the same length"
            result = await gather(*[self.rerank_handles[model].run.remote(
                inputs[i],
                normalize=normalize[i]
            ) for i in range(len(inputs))])
            scores = [e["score"] for e in result]
            total_tokens = sum([e["token_count"] for e in result])
            
        else:
            result = await self.rerank_handles[model].run.remote(inputs, normalize=normalize)
            scores = result["score"]
            total_tokens = result["token_count"]
        
        api.increment_usage_tally(self.database, user_auth, {
            "rerank": {
                self.config.default_models.rerank: {"tokens": total_tokens}
            }
        }, **({"api_key_id": original_auth} if auth_type == 2 else {}))
        
        return scores
    
    async def web_scrape_call(self, 
                              auth : AuthType,
                              inputs : Union[str, List[str]],
                              timeout : Union[float, List[float]] = 10,
                              markdown : Union[bool, List[bool]] = True,
                              load_strategy : Union[Literal["full", "eager", "none"], list] = "full",
                              summary: Union[bool, List[bool]] = False) -> List[List[float]]:
        
        (_, _) = api.get_user(self.database, auth)
        
        if isinstance(inputs, list):
            if not isinstance(timeout, list):
                timeout = [timeout for _ in range(len(inputs))]
            if not isinstance(markdown, list):
                markdown = [markdown for _ in range(len(inputs))]
            if not isinstance(summary, list):
                summary = [summary for _ in range(len(inputs))]
            if not isinstance(load_strategy, list):
                load_strategy = [load_strategy for _ in range(len(inputs))]
            
            assert all([len(timeout) == len(inputs), len(markdown) == len(inputs), len(summary) == len(inputs)]), \
                "All input lists must be the same length"   
            
            return await gather(*[self.web_scraper_handle.run.remote(
                inputs[i],
                timeout=timeout[i],
                markdown=markdown[i],
                summary=summary[i]
            ) for i in range(len(inputs))])
        else:
            return await self.web_scraper_handle.run.remote(inputs, timeout, markdown, summary)
    
    async def text_models_callback(self, request_dict: dict, model_choice: Literal["embedding", "rerank"]):
        assert model_choice in ["embedding", "rerank"]
        if model_choice == "embedding":
            return_tmp = await self.embedding_handle.run.remote(request_dict)
        elif model_choice == "rerank":
            return_tmp = await self.rerank_handle.run.remote(request_dict)
        return return_tmp
    
    async def llm_count_tokens(self, model_id : str, input_string : str):
        assert model_id in self.llm_handles, "Model choice not available"
        return await self.llm_handles_no_stream[model_id].count_tokens.remote(input_string)
    
    async def llm_call(self,
                       auth : AuthType, 
                       question : str = None,
                       model_parameters : dict = {},
                       sources : List[dict] = [],
                       chat_history : List[dict] = None,
                       stream_callables: Dict[str, Awaitable[Callable[[str], None]]] = None,
                       functions_available: List[Union[FunctionCallDefinition, dict]] = None,
                       only_format_prompt: bool = False):
        """
        Call an LLM model, possibly with parameters.
        
        TODO: Move OpenAI calls here for integration.
        TODO: Add optionality via default values to the model parameters.
        """
        (_, user_auth, original_auth, auth_type) = api.get_user(self.database, auth, return_auth_type=True)
        
        if not question is None:
            chat_history = [{"role": "user", "content": question}]
        
        if not chat_history is None:
            model_parameters["chat_history"] = chat_history
        
        on_new_token = None
        if not stream_callables is None and "output" in stream_callables:
            on_new_token = stream_callables["output"]
        
        model_choice = model_parameters.pop("model_choice", self.config.default_models.llm)
        assert model_choice in self.llm_handles, "Model choice not available"
        
        model_entry : sql_db_tables.model = self.database.exec(select(sql_db_tables.model)
                                            .where(sql_db_tables.model.id == self.config.default_models.llm)).first()
        
        model_parameters_true = {
            **json.loads(model_entry.default_settings),
        }
        
        model_parameters_true.update(model_parameters)
        
        return_stream_response = model_parameters_true.pop("stream", False)
        
        model_specified = model_choice.split("/")
        
        stop_sequences = model_parameters_true["stop"] if "stop" in model_parameters_true else []
        # print("Stop sequences:", stop_sequences)
        
        if len(model_specified) > 1:
            # External LLM provider (OpenAI, Anthropic, etc)
            
            gen = external_llm_generator(self.database, 
                                         auth, 
                                         *model_specified,
                                         model_parameters_true)
            input_token_count = -1
        else:
            assert self.config.enabled_model_classes.llm, "LLMs are disabled on this QueryLake Deployment"
            
            llm_handle : DeploymentHandle = self.llm_handles[model_choice]
            gen : DeploymentResponseGenerator = (
                llm_handle.get_result_loop.remote(deepcopy(model_parameters_true), sources=sources, functions_available=functions_available)
            )
            print("GOT LLM REQUEST GENERATOR")
            
            if self.llm_configs[model_choice].engine == "exllamav2":
                async for result in gen:
                    print("GENERATOR OUTPUT:", result)
            
            
            # input_token_count = self.llm_count_tokens(model_choice, model_parameters_true["text"])
            generated_prompt = await self.llm_handles_no_stream[model_choice].generate_prompt.remote(
                deepcopy(model_parameters_true), 
                sources=sources, 
                functions_available=functions_available
            )
            if only_format_prompt:
                return generated_prompt
            input_token_count = generated_prompt["tokens"]
        
        if "n" in model_parameters and model_parameters["n"] > 1:
            results = []
            async for result in gen:
                results = result
            # print(results)
            return [e.text for e in results.outputs]
        
        increment_usage_args = {
            "database": self.database, 
            "auth": user_auth,
            **({"api_key_id": original_auth} if auth_type == 2 else {})
        }
        
        total_output_tokens = 0
        def increment_token_count():
            nonlocal total_output_tokens
            total_output_tokens += 1
        
        def on_finish():
            nonlocal total_output_tokens, input_token_count, user_auth, original_auth, auth_type
            api.increment_usage_tally(self.database, user_auth, {
                "llm": {
                    model_choice: {
                        "input_tokens": input_token_count,
                        "output_tokens": total_output_tokens
                    }
                }
            }, **({"api_key_id": original_auth} if auth_type == 2 else {}))
        
        
        if return_stream_response:
            return StreamingResponse(
                stream_results_tokens(
                    gen, 
                    self.llm_configs[model_choice],
                    on_new_token=on_new_token,
                    increment_token_count=increment_token_count,
                    encode_output=False, 
                    stop_sequences=stop_sequences
                ),
                background=BackgroundTask(on_finish)
            )
        else:
            results = []
            async for result in stream_results_tokens(
                gen, 
                self.llm_configs[model_choice],
                on_new_token=on_new_token,
                increment_token_count=increment_token_count,
                stop_sequences=stop_sequences
            ):
                results.append(result)
            on_finish()
        
        text_outputs = "".join(results)
        
        call_results = {}
        if not functions_available is None:
            calls = find_function_calls(text_outputs)
            calls_possible = [
                e.name if isinstance(e, FunctionCallDefinition) else e["name"] 
                for e in functions_available 
            ]
            
            calls = [e for e in calls if ("function" in e and e["function"] in calls_possible)]
            call_results = {"function_calls": calls}
        
        return {"output": text_outputs, "output_token_count": len(results), "input_token_count": input_token_count, **call_results}
    
    def get_all_function_descriptions(self):
        """
        Return a list of all available API functions with their specifications.
        """
        return self.all_function_descriptions

    def api_function_getter(self, function_name):
        if function_name in self.special_function_table:
            return self.special_function_table[function_name]
        
        assert function_name in API_FUNCTIONS_ALLOWED, f"Invalid API Function '{function_name}' Called"
        return getattr(api, function_name)
    
    @fastapi_app.post("/upload_document/{rest_of_path:path}")
    async def upload_document_new(self, req : Request, rest_of_path: str, file : UploadFile):
        try:
            file_name = file.filename
            arguments = json.loads(req.query_params._dict["parameters"])
            file_ext = file_name.split(".")[-1]
            if file_ext in ["zip", "7z", "rar", "tar"]:
                target_func, target_func_str = api.upload_archive, "upload_archive"
            else:
                target_func, target_func_str = api.upload_document, "upload_document"
            
            print("Calling:", target_func_str, "with file:", file_name)
            
            true_arguments = clean_function_arguments_for_api({
                **self.default_function_arguments,
                "file": file,
            }, arguments, target_func_str)
            self.database.rollback()
            return {"success": True, "result": await target_func(**true_arguments)}
        
        except Exception as e:
            self.database.rollback()
            error_message = str(e)
            stack_trace = traceback.format_exc()
            return_msg = {"success": False, "note": error_message, "trace": stack_trace}
            # print(return_msg)
            print(error_message[:2000])
            return return_msg

    @fastapi_app.get("/fetch_document")
    async def get_document_2(self, req: Request):
        try:
            if "parameters" in req.query_params._dict:
                arguments = json.loads(req.query_params._dict["parameters"])
            else:
                arguments = await req.json()
            
            function_actual = getattr(api, "fetch_document")
            true_args = clean_function_arguments_for_api(
                self.default_function_arguments, 
                arguments, 
                "fetch_document"
            )
            
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
            return_dict = {"success": False, "error": error_message, "trace": stack_trace}
            print(json.dumps(return_dict, indent=4))
            return return_dict
    
    @fastapi_app.get("/api/ping")
    async def ping_function(self, req: Request):
        print("GOT PING!!!")
        return {"success": True, "note": "Pong"}
    
    # Callable by POST and GET
    @fastapi_app.post("/api/{rest_of_path:path}")
    @fastapi_app.get("/api/{rest_of_path:path}")
    async def api_general_call(self, req: Request, rest_of_path: str, file: UploadFile = None):
        """
        This is a wrapper around every api function that is allowed. 
        It will call the function with the arguments provided, after filtering them for security.
        """
        
        try:
            # body = await req.body()
            # print("Got request with body:", body)
            # arguments = req.query_params._dict
            print("Calling:", rest_of_path)
            
            if not file is None:
                print("File:", file.filename)
            
            if "parameters" in req.query_params._dict:
                arguments = json.loads(req.query_params._dict["parameters"])
            else:
                # We use ujson because normal `await req.json()` completely stalls on large inputs.
                # print("Awaiting JSON")
                
                arguments = await asyncio.wait_for(req.json(), timeout=10)
            
            
            # print("arguments:", arguments)
            route = req.scope['path']
            route_split = route.split("/")
            print("/".join(route_split[:3]))
            if rest_of_path == "help":
                if len(route_split) > 3:
                    function_name = route_split[3]
                    return {"success": True, "note": API_FUNCTION_HELP_DICTIONARY[function_name]}
                else:
                    print(API_FUNCTION_HELP_GUIDE)
                    return {"success": True, "note": API_FUNCTION_HELP_GUIDE}
            else:
                function_actual = self.api_function_getter(rest_of_path.split("/")[0])
                true_args = clean_function_arguments_for_api(
                    self.default_function_arguments, 
                    arguments, 
                    function_object=function_actual
                )
                
                if inspect.iscoroutinefunction(function_actual):
                    args_get = await function_actual(**true_args)
                else:
                    args_get = function_actual(**true_args)
                
                # print("Type of args_get:", type(args_get))
                
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
            
            if isinstance(e, InFailedSqlTransaction):
                self.database.rollback()
            
            error_message = str(e)
            stack_trace = traceback.format_exc()
            return_dict = {"success": False, "error": error_message, "trace": stack_trace}
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
            **self.default_function_arguments,
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
                    
                    auth = api.process_input_as_auth_type(auth)
                    
                    arguments : dict = arguments_websocket["arguments"]
                    
                    (_, _) = api.get_user(self.database, auth)
                    
                    arguments.update({"auth": auth})
                    
                    assert command in [
                        "toolchain/load",
                        "toolchain/create",
                        "toolchain/file_upload_event_call",
                        "toolchain/event",
                    ], "Invalid command"
                    
                    if command == "toolchain/load":
                        if not toolchain_session is None and toolchain_session.first_event_fired:
                            await api.save_toolchain_session(self.database, toolchain_session)
                            toolchain_session = None
                        true_args = clean_function_arguments_for_api(system_args, arguments, function_object=api.fetch_toolchain_session)
                        toolchain_session : ToolchainSession = api.fetch_toolchain_session(**true_args)
                        result = {
                            "success": True,
                            "loaded": True,
                            "toolchain_session_id": toolchain_session.session_hash,
                            "toolchain_id": toolchain_session.toolchain_id,
                            "state": toolchain_session.state,
                        }
                    
                    elif command == "toolchain/create":
                        if not toolchain_session is None and toolchain_session.first_event_fired:
                            await api.save_toolchain_session(self.database, toolchain_session)
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
                        event_result = await api.toolchain_event_call(**true_args, system_args=system_args, session=toolchain_session)
                        result = {"event_result": event_result}
                        toolchain_session.first_event_fired = True
                    
                    if toolchain_session.first_event_fired:
                        print("SAVING TOOLCHAIN")
                        await api.save_toolchain_session(self.database, toolchain_session)
                    
                    await ws.send_text((json.dumps(result)).encode("utf-8"))
                    await ws.send_text((json.dumps({"ACTION": "END_WS_CALL"})).encode("utf-8"))
                    
                    del result
                    
                    # await api.save_toolchain_session(self.database, toolchain_session)
                
                except WebSocketDisconnect:
                    raise WebSocketDisconnect
                except Exception as e:
                    error_message = str(e)
                    stack_trace = traceback.format_exc()
                    await ws.send_text(json.dumps({"error": error_message, "trace": stack_trace}))
                    await ws.send_text((json.dumps({"ACTION": "END_WS_CALL_ERROR"})).encode("utf-8"))
        except WebSocketDisconnect as e:
            print("Websocket disconnected")
            if not toolchain_session is None:
                print("Unloading Toolchain")
                
                if toolchain_session.first_event_fired:
                    await api.save_toolchain_session(self.database, toolchain_session)
                    
                toolchain_session.write_logs()
                toolchain_session = None
                del toolchain_session


SERVER_DIR = os.path.dirname(os.path.realpath(__file__))
# print("SERVER DIR:", SERVER_DIR)
os.chdir(SERVER_DIR)


with open("config.json", 'r', encoding='utf-8') as f:
    GLOBAL_CONFIG = f.read()
    f.close()
GLOBAL_CONFIG, TOOLCHAINS = Config.model_validate_json(GLOBAL_CONFIG), {}

default_toolchain = "chat_session_normal"

toolchain_files_list = os.listdir("toolchains")
for toolchain_file in toolchain_files_list:
    if not toolchain_file.split(".")[-1] == "json":
        continue
    with open("toolchains/"+toolchain_file, 'r', encoding='utf-8') as f:
        toolchain_retrieved = json.loads(f.read())
        f.close()
    try:
        TOOLCHAINS[toolchain_retrieved["id"]] = ToolChain(**toolchain_retrieved)
    except Exception as e:
        print("Toolchain error:", json.dumps(toolchain_retrieved, indent=4))
        raise e
        

LOCAL_MODEL_BINDINGS : Dict[str, DeploymentHandle] = {}

ENGINE_CLASSES = {"vllm": VLLMDeploymentClass, "exllamav2": ExllamaV2DeploymentClass}
ENGINE_CLASS_NAMES = {"vllm": "vllm", "exllamav2": "exl2"}

if GLOBAL_CONFIG.enabled_model_classes.llm:
    for model_entry in GLOBAL_CONFIG.models:
        
        # TODO: This will all be deprecated once we switch to ray clusters.
        # Only deploy the default model.
        # if not model_entry.id == GLOBAL_CONFIG.default_models.llm:
        #     continue
        if model_entry.disabled:
            continue
        elif model_entry.deployment_config is None:
            print("CANNOT DEPLOY; No deployment config for enabled model:", model_entry.id)
            continue
        class_choice = ENGINE_CLASSES[model_entry.engine]
        class_choice_decorated : serve.Deployment = serve.deployment(
            _func_or_class=class_choice,
            name=ENGINE_CLASS_NAMES[model_entry.engine]+":"+model_entry.id,
            **model_entry.deployment_config
        )
        LOCAL_MODEL_BINDINGS[model_entry.id] = class_choice_decorated.bind(
            model_config=model_entry,
            max_model_len=model_entry.max_model_len,
        )

embedding_models = {}
if GLOBAL_CONFIG.enabled_model_classes.embedding:
    for embedding_model in GLOBAL_CONFIG.other_local_models.embedding_models:
        class_choice_decorated : serve.Deployment = serve.deployment(
            _func_or_class=EmbeddingDeployment,
            name="embedding"+":"+embedding_model.id,
            **embedding_model.deployment_config
        )
        embedding_models[embedding_model.id] = class_choice_decorated.bind(
            model_card=embedding_model
        )

rerank_models = {}
if GLOBAL_CONFIG.enabled_model_classes.rerank:
    for rerank_model in GLOBAL_CONFIG.other_local_models.rerank_models:
        class_choice_decorated : serve.Deployment = serve.deployment(
            _func_or_class=RerankerDeployment,
            name="rerank"+":"+rerank_model.id,
            **rerank_model.deployment_config
        )
        rerank_models[rerank_model.id] = class_choice_decorated.bind(
            model_card=rerank_model
        )

deployment = UmbrellaClass.bind(
    configuration=GLOBAL_CONFIG,
    toolchains=TOOLCHAINS,
    web_scraper_handle=WebScraperDeployment.bind(),
    llm_handles=LOCAL_MODEL_BINDINGS,
    embedding_handles=embedding_models,
    rerank_handles=rerank_models
)

if __name__ == "__main__":
    serve.run(deployment)
