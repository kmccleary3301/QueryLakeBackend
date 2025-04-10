import json, os, re
import inspect
from copy import deepcopy
from pynvml import (
    nvmlInit, 
    nvmlDeviceGetCount, 
    nvmlDeviceGetHandleByIndex, 
    nvmlDeviceGetMemoryInfo
)

from fastapi import FastAPI, UploadFile, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware import Middleware

from typing import Literal, Tuple, Union, List, Dict, Awaitable, Callable

from ray import serve
from ray.serve.handle import DeploymentHandle

from QueryLake.api import api
from QueryLake.database import database_admin_operations
from QueryLake.database.create_db_session import initialize_database_engine
from QueryLake.typing.config import Config, AuthType, Model
from QueryLake.typing.toolchains import *
from QueryLake.operation_classes.ray_vllm_class import VLLMDeploymentClass
from QueryLake.operation_classes.ray_embedding_class import EmbeddingDeployment
from QueryLake.operation_classes.ray_reranker_class import RerankerDeployment
from QueryLake.operation_classes.ray_web_scraper import WebScraperDeployment
from QueryLake.operation_classes.ray_surya_class import (
    MarkerDeployment,
    SuryaDetectionDeployment,
    SuryaLayoutDeployment,
    SuryaOCRDeployment,
    SuryaOrderDeployment,
    SuryaTableDeployment,
    SuryaTexifyDeployment
)
from QueryLake.misc_functions.function_run_clean import get_function_call_preview, get_function_specs
from QueryLake.typing.function_calling import FunctionCallDefinition


from QueryLake.api.single_user_auth import global_public_key, global_private_key
from QueryLake.misc_functions.external_providers import external_llm_count_tokens
from QueryLake.misc_functions.server_class_functions import find_function_calls
from QueryLake.routing.ws_toolchain import toolchain_websocket_handler
from QueryLake.routing.openai_completions import (
    openai_chat_completion, 
    openai_create_embedding,
    ChatCompletionRequest, 
    EmbeddingRequest
)
from QueryLake.routing.api_call_router import api_general_call
from QueryLake.routing.llm_call import llm_call
from QueryLake.routing.upload_documents import handle_document
from QueryLake.routing.misc_models import (
    embedding_call,
    rerank_call,
    web_scrape_call
)




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

@serve.deployment(max_ongoing_requests=100)
@serve.ingress(fastapi_app)
class UmbrellaClass:
    def __init__(
        self,
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
            "web_scrape": self.web_scrape_call,
            "find_function_calls": find_function_calls,
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
    
    async def embedding_call(self, 
                             auth : AuthType,
                             inputs : List[str],
                             model: str = None):
        return await embedding_call(self, auth, inputs, model)
    
    async def rerank_call(
        self, 
        auth : AuthType,
        inputs : Union[List[Tuple[str, str]], Tuple[str, str]],
        normalize : Union[bool, List[bool]] = True,
        model: str = None
    ):
        return await rerank_call(self, auth, inputs, normalize, model)
    
    async def web_scrape_call(
        self, 
        auth : AuthType,
        inputs : Union[str, List[str]],
        timeout : Union[float, List[float]] = 10,
        markdown : Union[bool, List[bool]] = True,
        load_strategy : Union[Literal["full", "eager", "none"], list] = "full",
        summary: Union[bool, List[bool]] = False
    ):
        return await web_scrape_call(self, auth, inputs, timeout, markdown, load_strategy, summary)
    
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
    
    @fastapi_app.post("/v1/embeddings")
    async def openai_embedding_endpoint(
        self,
        request: EmbeddingRequest, 
        raw_request: Request
    ):
        return await openai_create_embedding(self, request, raw_request)
    
    
    async def llm_call(self,
                       auth : AuthType, 
                       question : str = None,
                       model_parameters : dict = {},
                       model : str = None,
                       lora_id : str = None,
                       sources : List[dict] = [],
                       chat_history : List[dict] = None,
                       stream_callables: Dict[str, Awaitable[Callable[[str], None]]] = None,
                       functions_available: List[Union[FunctionCallDefinition, dict]] = None,
                       only_format_prompt: bool = False):
        return await llm_call(
            self, auth, question,
            model_parameters,
            model, lora_id, sources, chat_history,
            stream_callables,
            functions_available,
            only_format_prompt
        )
    
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
    async def upload_document(self, req : Request, file : UploadFile):
        return await handle_document(self, clean_function_arguments_for_api, req, file)
    
    @fastapi_app.get("/api/ping")
    async def ping_function(self, req: Request):
        print("GOT PING!!!")
        return {"success": True, "note": "Pong"}
    
    # Callable by POST and GET
    @fastapi_app.post("/api/{rest_of_path:path}")
    @fastapi_app.get("/api/{rest_of_path:path}")
    async def api_general_call(self, req: Request, rest_of_path: str, file: UploadFile = None):
        return await api_general_call(
            self,
            clean_function_arguments_for_api,
            API_FUNCTION_HELP_DICTIONARY,
            API_FUNCTION_HELP_GUIDE,
            req, rest_of_path, file
        )
    
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
        "surya_detection": SuryaDetectionDeployment,
        "surya_layout": SuryaLayoutDeployment,
        "surya_recognition": SuryaOCRDeployment,
        "surya_order": SuryaOrderDeployment,
        "surya_table_recognition": SuryaTableDeployment,
        "surya_texify": SuryaTexifyDeployment
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
