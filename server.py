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
from QueryLake.operation_classes.ray_vllm_class import VLLMDeploymentClass, format_chat_history
# from QueryLake.operation_classes.ray_exllamav2_class import ExllamaV2DeploymentClass
from QueryLake.operation_classes.ray_embedding_class import EmbeddingDeployment
from QueryLake.operation_classes.ray_reranker_class import RerankerDeployment
from QueryLake.operation_classes.ray_web_scraper import WebScraperDeployment
from QueryLake.misc_functions.function_run_clean import get_function_call_preview, get_function_specs
from QueryLake.typing.function_calling import FunctionCallDefinition

from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware import Middleware
import asyncio

from psycopg2.errors import InFailedSqlTransaction

from fastapi import FastAPI
from fastapi.security import OAuth2PasswordRequestForm
from fastapi import Depends

from QueryLake.api.single_user_auth import global_public_key, global_private_key
from QueryLake.misc_functions.external_providers import external_llm_generator, external_llm_count_tokens
from QueryLake.misc_functions.server_class_functions import stream_results_tokens, find_function_calls, basic_stream_results
from QueryLake.routing.ws_toolchain import toolchain_websocket_handler
from QueryLake.routing.openai_completions import openai_chat_completion, ChatCompletionRequest

from asyncio import gather
from pynvml import nvmlInit, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo

from QueryLake.operation_classes.ray_surya_class import (
    MarkerDeployment
)

from io import BytesIO

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

@serve.deployment(max_ongoing_requests=100)
@serve.ingress(fastapi_app)
class UmbrellaClass:
    def __init__(self,
                 configuration: Config,
                 toolchains: Dict[str, ToolChain],
                 web_scraper_handle: DeploymentHandle,
                 llm_handles: Dict[str, DeploymentHandle] = {},
                 embedding_handles: Dict[str, DeploymentHandle] = None,
                 rerank_handles: Dict[str, DeploymentHandle] = None,
                 surya_handles: Dict[str, DeploymentHandle] = None,
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
        self.surya_handles = surya_handles if surya_handles else {}
        
        self.database, self.engine = initialize_database_engine()
        
        database_admin_operations.add_models_to_database(self.database, self.config.models)
        database_admin_operations.add_toolchains_to_database(self.database, self.toolchain_configs)
        
        self.default_function_arguments = {
            # All model deployments
            "server_llm_handles_no_stream": self.llm_handles_no_stream,
            "server_llm_handles": self.llm_handles,
            "server_embedding_handles": self.embedding_handles,
            "server_rerank_handles": self.rerank_handles,
            "server_web_scraper_handle": self.web_scraper_handle,
            "server_surya_handles": self.surya_handles,
            
            
            "database": self.database,
            "toolchains_available": self.toolchain_configs,
            "public_key": global_public_key,
            "server_private_key": global_private_key,
            "toolchain_function_caller": self.api_function_getter,
            "global_config": self.config,
        }
        
        self.special_function_table = {
            "llm": self.llm_call,
            "llm_count_tokens": self.llm_count_tokens,
            "embedding": self.embedding_call,
            "rerank": self.rerank_call,
            "find_function_calls": find_function_calls,
            "web_scrape": self.web_scrape_call,
            "function_help": self.get_all_function_descriptions
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
        assert model in self.embedding_handles, f"Model choice [{model}] not available for embeddings"
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
        assert model in self.rerank_handles, f"Model choice [{model}] not available for rerankers"
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
    
    async def llm_count_tokens(self, model_id : str, input_string : str):
        model_specified = model_id.split("/")
        if len(model_specified) == 1:
            assert model_id in self.llm_handles, f"Model choice [{model_id}] not available for LLMs (tried to count tokens)"
            return await self.llm_handles_no_stream[model_id].count_tokens.remote(input_string)
        else:
            return external_llm_count_tokens(input_string, model_id)
    
    
    
    @fastapi_app.post("/v1/chat/completions")
    async def openai_chat_completions_endpoint(
        self,
        request: ChatCompletionRequest,
        raw_request : Request
    ):
        return await openai_chat_completion(self, request, raw_request)
    
    async def llm_call(self,
                       auth : AuthType, 
                       question : str = None,
                       model_parameters : dict = {},
                       model : str = None,
                       sources : List[dict] = [],
                       chat_history : List[dict] = None,
                       stream_callables: Dict[str, Awaitable[Callable[[str], None]]] = None,
                       functions_available: List[Union[FunctionCallDefinition, dict]] = None,
                       only_format_prompt: bool = False):
        """
        Call an LLM model, possibly with parameters.
        """
        (_, user_auth, original_auth, auth_type) = api.get_user(self.database, auth, return_auth_type=True)
        
        if not question is None:
            chat_history = [{"role": "user", "content": question}]
        
        if not chat_history is None:
            model_parameters["chat_history"] = chat_history
        
        on_new_token = None
        if not stream_callables is None and "output" in stream_callables:
            on_new_token = stream_callables["output"]
        
        if not model is None:
            model_choice = model
        else:
            model_choice : str = model_parameters.pop("model", self.config.default_models.llm)
        
        model_specified = model_choice.split("/")
        return_stream_response = model_parameters.pop("stream", False)
        
        input_token_count = -1
        def set_input_token_count(input_token_value: int):
            nonlocal input_token_count
            input_token_count = input_token_value
        
        if len(model_specified) > 1:
            # External LLM provider (OpenAI, Anthropic, etc)
            
            new_chat_history = format_chat_history(
                chat_history,
                sources=sources,
                functions_available=functions_available
            )
            model_parameters.update({
                "messages": new_chat_history,
                "chat_history": new_chat_history
            })
            gen = external_llm_generator(self.database, 
                                         auth, 
                                         provider=model_specified[0],
                                         model="/".join(model_specified[1:]),
                                         request_dict=model_parameters,
                                         set_input_token_count=set_input_token_count)
        else:
            model_entry : sql_db_tables.model = self.database.exec(select(sql_db_tables.model)
                                                .where(sql_db_tables.model.id == model_choice)).first()
            
            model_parameters_true = {
                **json.loads(model_entry.default_settings),
            }
            
            model_parameters_true.update(model_parameters)
            
            
            
            stop_sequences = model_parameters_true["stop"] if "stop" in model_parameters_true else []
            
            assert self.config.enabled_model_classes.llm, "LLMs are disabled on this QueryLake Deployment"
            assert model_choice in self.llm_handles, f"Model choice [{model_choice}] not available for LLMs"
            
            llm_handle : DeploymentHandle = self.llm_handles[model_choice]
            gen : DeploymentResponseGenerator = (
                llm_handle.get_result_loop.remote(deepcopy(model_parameters_true), sources=sources, functions_available=functions_available)
            )
            # print("GOT LLM REQUEST GENERATOR WITH %d SOURCES" % len(sources))
            
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
            if len(model_specified) == 1:
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
                return StreamingResponse(basic_stream_results(gen, on_new_token=on_new_token))
        else:
            results = []
            if len(model_specified) == 1:
                async for result in stream_results_tokens(
                    gen, 
                    self.llm_configs[model_choice],
                    on_new_token=on_new_token,
                    increment_token_count=increment_token_count,
                    stop_sequences=stop_sequences
                ):
                    results.append(result)
                on_finish()
            else:
                async for result in basic_stream_results(gen, on_new_token=on_new_token):
                    results.append(result)
        
        
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
    
    @fastapi_app.post("/update_documents")
    @fastapi_app.post("/upload_document")
    async def upload_document_new(self, req : Request, file : UploadFile):
        endpoint = req.scope['path']
        try:
            
            # We have to take the arguments in the header, because the body is the file.
            arguments = json.loads(req.query_params._dict["parameters"])
            file_name = file.filename
            file_ext = file_name.split(".")[-1]
            if endpoint.strip() == "/update_documents":
                target_func, target_func_str = api.update_documents, "update_documents"
            elif file_ext in ["zip", "7z", "rar", "tar"]:
                target_func, target_func_str = api.upload_archive, "upload_archive"
            else:
                target_func, target_func_str = api.upload_document, "upload_document"
            
            true_arguments = clean_function_arguments_for_api({
                **self.default_function_arguments,
                "file": file,
            }, arguments, target_func_str)
            return {"success": True, "result": await target_func(**true_arguments)}
        
        except Exception as e:
            # if isinstance(e, InFailedSqlTransaction):
            self.database.rollback()
            self.database.flush()
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
            if isinstance(e, InFailedSqlTransaction):
                self.database.flush()
                self.database.rollback()
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
            
            self.database.rollback()
            self.database.flush()
            
            error_message = str(e)
            stack_trace = traceback.format_exc()
            return_dict = {"success": False, "error": error_message, "trace": stack_trace}
            print("RETURNING:", json.dumps(return_dict, indent=4))
            return return_dict
            # return {"success": False, "note": str(e)}
    
    @fastapi_app.websocket("/toolchain")
    async def toolchain_websocket_handler(self, ws: WebSocket):
        return await toolchain_websocket_handler(
            self, 
            ws,
            clean_function_arguments_for_api,
            api.fetch_toolchain_session,
            api.create_toolchain_session,
            api.toolchain_file_upload_event_call,
            api.save_toolchain_session,
            api.toolchain_event_call
        )

SERVER_DIR = os.path.dirname(os.path.realpath(__file__))
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

def check_gpus():
    nvmlInit()
    device_count = nvmlDeviceGetCount()
    gpus = []
    for i in range(device_count):
        handle = nvmlDeviceGetHandleByIndex(i)
        memory_info = nvmlDeviceGetMemoryInfo(handle)
        gpus.append({
            "index": i,
            "total_vram": memory_info.total // (1024 ** 2)  # Convert bytes to MB
        })
    return gpus

# Calculate VRAM fractions for each class to adjust serve resource configurations on deployment.

gpus = sorted(check_gpus(), key=lambda x: x["total_vram"], reverse=True)
if len(gpus) > 1:
    print("Multiple GPUs detected. QueryLake isn't optimized for multi-GPU usage yet. Continue at your own risk.")
MAX_GPU_VRAM = gpus[0]["total_vram"]


LOCAL_MODEL_BINDINGS : Dict[str, DeploymentHandle] = {}

# ENGINE_CLASSES = {"vllm": VLLMDeploymentClass, "exllamav2": ExllamaV2DeploymentClass}
ENGINE_CLASSES = {"vllm": VLLMDeploymentClass}
ENGINE_CLASS_NAMES = {"vllm": "vllm", "exllamav2": "exl2"}

if GLOBAL_CONFIG.enabled_model_classes.llm:
    for model_entry in GLOBAL_CONFIG.models:
        
        if model_entry.disabled:
            continue
        elif model_entry.deployment_config is None:
            print("CANNOT DEPLOY; No deployment config for enabled model:", model_entry.id)
            continue
        if not model_entry.engine in ENGINE_CLASSES:
            print("CANNOT DEPLOY; Engine not recognized:", model_entry.engine)
            continue
        assert "vram_required" in model_entry.deployment_config, f"No VRAM requirement specified for {model_entry.id}"
        
        class_choice = ENGINE_CLASSES[model_entry.engine]
        
        vram_fraction = model_entry.deployment_config.pop("vram_required") / MAX_GPU_VRAM
        if model_entry.engine == "vllm":
            model_entry.engine_args["gpu_memory_utilization"] = vram_fraction
        model_entry.deployment_config["ray_actor_options"]["num_gpus"] = vram_fraction 
        
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
        assert "vram_required" in embedding_model.deployment_config, f"No VRAM requirement specified for {embedding_model.id}"
        vram_fraction = embedding_model.deployment_config.pop("vram_required") / MAX_GPU_VRAM
        embedding_model.deployment_config["ray_actor_options"]["num_gpus"] = vram_fraction
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
        assert "vram_required" in rerank_model.deployment_config, f"No VRAM requirement specified for {rerank_model.id}"
        vram_fraction = rerank_model.deployment_config.pop("vram_required") / MAX_GPU_VRAM
        rerank_model.deployment_config["ray_actor_options"]["num_gpus"] = vram_fraction
        class_choice_decorated : serve.Deployment = serve.deployment(
            _func_or_class=RerankerDeployment,
            name="rerank"+":"+rerank_model.id,
            **rerank_model.deployment_config
        )
        
        rerank_models[rerank_model.id] = class_choice_decorated.bind(
            model_card=rerank_model
        )

# Add Surya model bindings
surya_handles = {}
if GLOBAL_CONFIG.enabled_model_classes.surya:
    surya_model_map = {
        "marker": MarkerDeployment,
    }
    
    for surya_model in GLOBAL_CONFIG.other_local_models.surya_models:
        # Map model name to deployment class
        class_to_use = surya_model_map[surya_model.id]
        
        # Configure VRAM fraction
        assert "vram_required" in surya_model.deployment_config, f"No VRAM requirement specified for {surya_model.name}"
        vram_fraction = surya_model.deployment_config.pop("vram_required") / MAX_GPU_VRAM
        surya_model.deployment_config["ray_actor_options"]["num_gpus"] = vram_fraction
        
        # Create deployment
        deployment_class = serve.deployment(
            _func_or_class=class_to_use,
            name=f"surya:{surya_model.id}",
            **surya_model.deployment_config
        )
        
        surya_handles[surya_model.name] = deployment_class.bind(
            model_card=surya_model
        )


deployment = UmbrellaClass.bind(
    configuration=GLOBAL_CONFIG,
    toolchains=TOOLCHAINS,
    web_scraper_handle=WebScraperDeployment.bind(),
    llm_handles=LOCAL_MODEL_BINDINGS,
    embedding_handles=embedding_models,
    rerank_handles=rerank_models,
    surya_handles=surya_handles
)

if __name__ == "__main__":
    serve.run(deployment)
