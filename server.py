import asyncio
import traceback
# import logging

from typing import Annotated, Callable, Any
import json, os
import uvicorn
from fastapi import FastAPI, File, UploadFile, APIRouter, Request, WebSocket, Form
from sse_starlette.sse import EventSourceResponse
from starlette.requests import Request
from fastapi.middleware.cors import CORSMiddleware
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

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid
from vllm.outputs import RequestOutput
from vllm_lmformating_modifed_banned_tokens import build_vllm_token_enforcer_tokenizer_data

from ray.util import ActorPool
from ray import serve, remote
from ray.serve.handle import DeploymentHandle, DeploymentResponseGenerator
from ray import ObjectRef


from lmformatenforcer import JsonSchemaParser
from lmformatenforcer import CharacterLevelParser
from lmformatenforcer.integrations.vllm import VLLMLogitsProcessor
from lmformatenforcer.regexparser import RegexParser

from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel
import torch

from grammar_sampling_functions import get_token_id, get_logits_processor_from_grammar_options
from math import exp
# from QueryLake.api.toolchains import ToolchainSession
from QueryLake.operation_classes.toolchain_session import ToolchainSession
from QueryLake.api.hashing import random_hash
import openai

import pgvector
from pydantic import BaseModel
from QueryLake.typing.config import Config, AuthType, getUserType, Padding, ModelArgs, Model
from QueryLake.models.prompt_construction import construct_chat_history
from QueryLake.typing.toolchains import *

from FlagEmbedding import FlagReranker
import math

from sqlalchemy import util




# from sqlmodel.sql.expression import Select
# from sqlalchemy.engine.result import TupleResult


fastapi_app = FastAPI(
    # lifespan=lifespan
)

# origins = ["*"]

# fastapi_app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
ACTIVE_SESSIONS = {}

# GlobalLLMEnsemble = LLMEnsemble(database, GLOBAL_SETTINGS["default_model"], GLOBAL_SETTINGS)
global_public_key, global_private_key = encryption.ecc_generate_public_private_key()


# print("Model Loaded")

TEMPLATE_FUNCTIONS = [pair[0] for pair in inspect.getmembers(template_functions, inspect.isfunction)]



API_FUNCTIONS = [pair[0] for pair in inspect.getmembers(api, inspect.isfunction)]
# API_FUNCTIONS_ALL = [func for func in API_FUNCTIONS if not re.match(r"__.*?__", func)]
# print("ALL API_FUNCTIONS")
# print(json.dumps(sorted(API_FUNCTIONS_ALL), indent=4))
API_FUNCTIONS = [func for func in API_FUNCTIONS if (not re.match(r"__.*?__", func) and func not in api.excluded_member_function_descriptions)]
# print("API FUNCTIONS INCLUSIVE")
# print(json.dumps(sorted(API_FUNCTIONS), indent=4))

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

# def toolchain_function_caller(function_name):
#     assert function_name in API_FUNCTIONS or function_name in TEMPLATE_FUNCTIONS, "Function not available"
#     if function_name in API_FUNCTIONS:
#         return getattr(api, function_name)
#     return getattr(template_functions, function_name)

@serve.deployment(ray_actor_options={"num_gpus": 0.6}, max_replicas_per_node=1)
class LLMDeploymentClass:
    def __init__(self,
                 model_config : Model,
                 **kwargs):
        """
        Construct a VLLM deployment.

        Refer to https://github.com/vllm-project/vllm/blob/main/vllm/engine/arg_utils.py
        for the full list of arguments.

        Args:
            model: name or path of the huggingface model to use
            download_dir: directory to download and load the weights,
                default to the default cache dir of huggingface.
            use_np_weights: save a numpy copy of model weights for
                faster loading. This can increase the disk usage by up to 2x.
            use_dummy_weights: use dummy values for model weights.
            dtype: data type for model weights and activations.
                The "auto" option will use FP16 precision
                for FP32 and FP16 models, and BF16 precision.
                for BF16 models.
            seed: random seed.
            worker_use_ray: use Ray for distributed serving, will be
                automatically set when using more than 1 GPU
            pipeline_parallel_size: number of pipeline stages.
            tensor_parallel_size: number of tensor parallel replicas.
            block_size: token block size.
            swap_space: CPU swap space size (GiB) per GPU.
            gpu_memory_utilization: the percentage of GPU memory to be used for
                the model executor
            max_num_batched_tokens: maximum number of batched tokens per iteration
            max_num_seqs: maximum number of sequences per iteration.
            disable_log_stats: disable logging statistics.
            engine_use_ray: use Ray to start the LLM engine in a separate
                process as the server process.
            disable_log_requests: disable logging requests.
        """
        self.model_config = model_config
        self.padding : Padding = model_config.padding
        self.default_model_args = self.model_config.default_parameters
        self.minimum_free_token_space = kwargs.pop("minimum_free_token_space", 2000)
        
        args = AsyncEngineArgs(**kwargs, disable_log_requests=True) # Had to mute this thing because it was spamming the logs.
        
        self.context_size = args.max_model_len
        
        self.engine = AsyncLLMEngine.from_engine_args(args)
        
        tokenizer_tmp = self.engine.engine.tokenizer
        
        self.special_token_ids = tokenizer_tmp.all_special_ids
        
        self.space_tokens = [get_token_id(tokenizer_tmp, e) for e in ["\n", "\t", "\r", " \r"]]

        self.tokenizer_data = build_vllm_token_enforcer_tokenizer_data(self.engine.engine.tokenizer)
        
        print("Test count tokens ->", self.count_tokens("Hello, world!"))
    
    def count_tokens(self, input_string : str):
        return len(self.engine.engine.tokenizer(input_string)["input_ids"])
    
    def generator(self, request_dict : dict):
        
        if "prompt" in request_dict:
            prompt = request_dict.pop("prompt")
        else:
            chat_history = request_dict.pop("chat_history")
            prompt = construct_chat_history(self.model_config, self.count_tokens, chat_history, self.minimum_free_token_space)
        
        print("Prompt:", prompt)
            
        request_id = random_uuid()
        # stream = request_dict.pop("stream", False)
        
        grammar_options = request_dict.pop("grammar", None)
        
        logits_processor_local = get_logits_processor_from_grammar_options(
            grammar_options,
            self.tokenizer_data, 
            space_tokens=self.space_tokens, 
            special_ids=self.special_token_ids,
        )
        
        sampling_params = SamplingParams(**request_dict)
        
        if not logits_processor_local is None:
            sampling_params.logits_processors = [logits_processor_local]
        
        results_generator = self.engine.generate(prompt, sampling_params, request_id)
        
        return results_generator

@serve.deployment
class EmbeddingDeployment:
    def __init__(self, model_key: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_key)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = AutoModel.from_pretrained(model_key).to(self.device)

    @serve.batch()
    async def handle_batch(self, inputs: List[List[str]]) -> List[List[List[float]]]:
        print("Running Embedding 3.1")
        # print("Our input array has length:", len(inputs))

        # print("Input array:", inputs)
        
        batch_size = len(inputs)
        # We need to flatten the inputs into a single list of string, then recombine the outputs according to the original structure.
        flat_inputs = [(item, i_1, i_2) for i_1, sublist in enumerate(inputs) for i_2, item in enumerate(sublist)]
        
        flat_inputs_indices = [item[1:] for item in flat_inputs]
        flat_inputs_text = [item[0] for item in flat_inputs]
        encoded_input = self.tokenizer(flat_inputs_text, padding=True, truncation=True, return_tensors='pt').to(self.device)

        with torch.no_grad():
            model_output = self.model(**encoded_input)
            sentence_embeddings = model_output[0][:, 0]
        
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        
        outputs = [[] for _ in range(batch_size)]
        
        for i, output in enumerate(sentence_embeddings.tolist()):
            request_index, input_index = flat_inputs_indices[i]
            outputs[request_index].append(output)
            # outputs[item[1]][item[2]] = sentence_embeddings[i].tolist()
        
        return outputs
        # results = self.model(inputs)
        # return [result[0]["generated_text"] for result in results]

    async def __call__(self, request : Request) -> Response:
        print("Running Embedding 2.1")
        request_dict = await request.json()
        print("Running Embedding 2.2")
        # return await self.handle_batch(request_dict["text"])
        return_tmp = await self.handle_batch(request_dict["text"])
        print("Running Embedding 2.3")
        return Response(content=json.dumps({"output": return_tmp}))
    
    async def run(self, request_dict : dict) -> List[List[float]]:
        return await self.handle_batch(request_dict["text"])
        # __doc_define_servable_end__

def modified_sigmoid(x : Union[torch.Tensor, float]):
    if type(x) == float or type(x) == int:
        return 100 / (1 + exp(-8 * ((x / 100) - 0.5)))
    if type(x) == list:
        x = torch.tensor(x)
    return 100 * torch.sigmoid(-8 * ((x / 100) - 0.5))

def S(x):
    # return 1/(1 + torch.exp(-4 * (x - 0.5))) if isinstance(x, torch.Tensor) else 1 / (1 + math.exp(-4 * (x - 0.5)))
    return torch.sigmoid(4*(x - 0.5)) if isinstance(x, torch.Tensor) else 1 / (1 + math.exp(-4 * (x - 0.5)))

def H(x):
    return 100 * S(x / 100) if isinstance(x, torch.Tensor) else 100 * S(x / 100)

def F(x):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor([x], dtype=torch.float32)

    h_zero = H(0)
    # h_one = H(100)
    delta_h = H(x) - h_zero
    result = delta_h * (100 / (100 - h_zero))

    # If input was a single value, convert back to scalar output
    result = torch.minimum(result, torch.full(result.shape, 100.0))
    
    if len(result.shape) == 0:
        result = float(result)

    return result

@serve.deployment(ray_actor_options={"num_gpus": 0.4}, max_replicas_per_node=1)
class RerankerDeployment:
    def __init__(self, model_key: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_key)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForSequenceClassification.from_pretrained(model_key).to(self.device)
        
        # self.model = FlagReranker(model_key, use_fp16=True)
    
    @serve.batch()
    async def handle_batch(self, inputs: List[List[Tuple[str, str]]]) -> List[List[float]]:
        batch_size = len(inputs)
        flat_inputs = [(item, i_1, i_2) for i_1, sublist in enumerate(inputs) for i_2, item in enumerate(sublist)]
        
        flat_inputs_indices = [item[1:] for item in flat_inputs]
        flat_inputs_text = [item[0] for item in flat_inputs]
        
        with torch.no_grad():
            
            print("Reranker Inputs:", len(flat_inputs_text), batch_size)
            # with open("reranker_inputs.txt", "w") as f:
            #     f.write(json.dumps(flat_inputs_text, indent=4))
            #     f.close()
            
            
            start_time = time.time()
            tokenized_inputs = self.tokenizer(flat_inputs_text, padding=True, truncation=True, return_tensors='pt', max_length=512).to(self.device)
            mid_time = time.time()
            scores = self.model(**tokenized_inputs, return_dict=True).logits.view(-1, ).float().to("cpu") # This takes 8 seconds!!! WTF?!?!
            
            
            # print("Reranker Inputs:", flat_inputs_text)
            # scores = self.model.compute_score(flat_inputs_text)
            end_time = time.time()
            print("RERANKER TIME INTERNAL 2 %7.3fs %7.3fs %7.3fs" % (end_time - start_time, mid_time - start_time, end_time - mid_time))
            
            # start_time = time.time()
            # tokenized_inputs = self.tokenizer(flat_inputs_text, padding=True, truncation=True, return_tensors='pt', max_length=512)
            # mid_time = time.time()
            # scores = self.model(**tokenized_inputs, return_dict=True).logits.view(-1, ).float() # This takes 8 seconds!!! WTF?!?!
            # end_time = time.time()
            # print("RERANKER TIME INTERNAL 2 %7.3fs %7.3fs %7.3fs" % (end_time - start_time, mid_time - start_time, end_time - mid_time))
            
            
        scores = torch.exp(torch.tensor(scores))
        scores = F(scores)
        
        outputs = [[] for _ in range(batch_size)]
        
        for i, output in enumerate(scores.tolist()):
            request_index, input_index = flat_inputs_indices[i]
            outputs[request_index].append(output)
        
        return outputs

    async def __call__(self, request : Request) -> Response:
        request_dict = await request.json()
        # print("Got request:", request_dict)
        return_tmp = await self.handle_batch(request_dict["text"])
        return Response(content=json.dumps({"output": return_tmp}))

    async def run(self, request_dict : dict) -> List[List[float]]:
        start_time = time.time()
        values = await self.handle_batch(request_dict["text"])
        end_time = time.time()
        print("RERANKER INTERNAL TIME %7.3fs" % (end_time - start_time))
        return values

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
        
        text_outputs = "".join(results)
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
        
        assert function_name in API_FUNCTIONS, "Invalid API Function Called"
        assert function_name in API_FUNCTIONS or function_name in TEMPLATE_FUNCTIONS, "Function not available"
        if function_name in API_FUNCTIONS:
            return getattr(api, function_name)
        return getattr(template_functions, function_name)
    
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
                "text_models_callback": self.text_models_callback,
                "file": file,
            }, arguments, "upload_document")

            return await api.upload_document(**true_arguments)
        except Exception as e:
            error_message = str(e)
            stack_trace = traceback.format_exc()
            return_msg = {"success": False, "note": error_message, "trace": stack_trace}
            print(return_msg)
            return return_msg

    @fastapi_app.post("/api/async/upload_document_to_session")
    async def upload_document_to_session(self, req: Request, file : UploadFile):
        # try:
        # arguments = req.query_params._dict
        
        def create_embeddings(text : str) -> List[List[float]]:
            pass
        
        route = req.scope['path']
        route_split = route.split("/")
        print("/".join(route_split[:4]), req.query_params._dict)
        arguments = await req.json()
        # arguments = json.loads(req.query_params._dict["parameters"]) 
        true_arguments = clean_function_arguments_for_api({
            "database": self.database,
            "file": file,
            "create_embeddings": create_embeddings,
            "return_file_hash": True,
            "add_to_vector_db": False
        }, arguments, "upload_document")

        upload_result = api.upload_document(**true_arguments)

        true_args_2 = clean_function_arguments_for_api({
            "database": self.database,
            "public_key": global_public_key,
            "server_private_key": global_private_key,
            "toolchain_function_caller": self.api_function_getter,
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
    
    @fastapi_app.post("/api/{rest_of_path:path}")
    async def api_general_call(self, req: Request, rest_of_path: str):
        """
        This is a wrapper around every api function that is allowed. 
        It will call the function with the arguments provided, after filtering them for security.
        """
        
        try:
            # arguments = req.query_params._dict
            print("Calling:", rest_of_path)
            if "parameters" in req.query_params._dict:
                arguments = json.loads(req.query_params._dict["parameters"])
            else:
                arguments = await req.json()
            print("arguments:", arguments)
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
                
                if type(args_get) is ThreadedGenerator:
                    return EventSourceResponse(args_get)
                elif type(args_get) is StreamingResponse:
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
            print(json.dumps(return_dict, indent=4))
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
                            "toolchain_state": toolchain_session.state,
                        }
                    
                    elif command == "toolchain/file_upload_event_call":
                        true_args = clean_function_arguments_for_api(system_args, arguments, function_object=api.toolchain_file_upload_event_call)
                        result = await api.toolchain_file_upload_event_call(**true_args, session=toolchain_session)
                    
                    elif command == "toolchain/entry":
                        true_args = clean_function_arguments_for_api(system_args, arguments, function_object=api.toolchain_entry_call)
                        result = await api.toolchain_entry_call(**true_args, session=toolchain_session)
                    
                    elif command == "toolchain/event":
                        # print("SYSTEM ARGS KEYS AT EVENT:", list(system_args.keys()))
                        true_args = clean_function_arguments_for_api(system_args, arguments, function_object=api.toolchain_event_call)
                        # print("PASSED KEYS AT EVENT:", list(true_args.keys()))
                        result = await api.toolchain_event_call(**true_args, session=toolchain_session)
                        # print("RESULT AT EVENT:", result)
                    
                    await ws.send_text((json.dumps(result)).encode("utf-8"))
                    await ws.send_text((json.dumps({"ACTION": "END_WS_CALL"})).encode("utf-8"))
                    del result_message
                    generate = {"STOP_GENERATION": False}
                    print("\n\n\n\n\n\n\n\n\n")
                    # await self.llm_call(request_dict, ws)
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
