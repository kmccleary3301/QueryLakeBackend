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
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from contextlib import asynccontextmanager

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

# This import was added in error and is being removed.
# from QueryLake.log_utils import set_up_logger




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

# This function is the new entrypoint for the application, called by start_querylake.py
def _get_placement_bundle() -> Dict[str, Union[int, float]]:
    """Get the placement group bundle configuration (GPU, VRAM_MB, CPU) for this deployment."""
    try:
        import ray
        import pynvml
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        
        bundle = {"GPU": 1}
        
        # Add VRAM if we can detect it
        if device_count > 0:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            vram_mb = memory_info.total // (1024 * 1024)
            bundle["VRAM_MB"] = vram_mb
        
        # Add CPU - calculate CPU allocation per GPU node based on actual per-node CPU availability
        gpu_nodes = [node for node in ray.nodes() if node.get("Resources", {}).get("GPU", 0) > 0 and node.get("alive", False)]
        if gpu_nodes:
            # Use 90% of CPUs from a single GPU node, leaving 10% for system overhead
            node_cpus = gpu_nodes[0].get("Resources", {}).get("CPU", 48)
            cpus_per_gpu_node = max(2, int(node_cpus * 0.9))  # Use 90% of node CPUs, minimum 2
            bundle["CPU"] = cpus_per_gpu_node
        else:
            bundle["CPU"] = 40  # Conservative fallback
            
        return bundle
    except Exception:
        # Fallback bundle
        return {"GPU": 1, "CPU": 2}

def _get_placement_group_config(strategy: str, num_replicas: int, replica_resources: Dict = None) -> Dict:
    """
    Ray Serve has its own built-in resource management and scheduling mechanisms.
    Placement groups are not directly compatible with Ray Serve deployments.
    This function now returns an empty dict to disable placement group usage.
    
    Ray Serve automatically handles:
    - Resource allocation across available nodes
    - Replica placement and load balancing  
    - Scaling based on traffic and resource utilization
    
    The strategy parameter is noted but not applied since Ray Serve manages placement internally.
    """
    print(f"ℹ️ Ray Serve deployment with {num_replicas} replicas using strategy: {strategy}")
    print(f"   Ray Serve will handle resource allocation and replica placement automatically")
    print(f"   Replica resources: {replica_resources}")
    
    # Ray Serve manages its own scheduling and resource allocation
    # Placement groups are not needed and not supported for Serve deployments
    return {}

def build_and_run_application(
    strategy: str,
    global_config: Config,
    toolchains: Dict[str, ToolChain]
):
    """
    Builds the Ray Serve application graph and runs it.
    This function is called by the new CLI orchestrator.
    """
    LOCAL_MODEL_BINDINGS: Dict[str, DeploymentHandle] = {}
    ENGINE_CLASSES = {"vllm": VLLMDeploymentClass}
    ENGINE_CLASS_NAMES = {"vllm": "vllm"}

    if global_config.enabled_model_classes.llm:
        for model_entry in global_config.models:
            if model_entry.disabled:
                continue
            elif model_entry.deployment_config is None:
                print("CANNOT DEPLOY; No deployment config for enabled model:", model_entry.id)
                continue
            if model_entry.engine not in ENGINE_CLASSES:
                print("CANNOT DEPLOY; Engine not recognized:", model_entry.engine)
                continue
            
            assert "vram_required" in model_entry.deployment_config, f"No VRAM requirement specified for {model_entry.id}"
            
            class_choice = ENGINE_CLASSES[model_entry.engine]
            
            # New resource model: Explicit VRAM request, no more VRAM fraction hack.
            # We request a tiny amount of GPU to ensure it's a GPU actor, and the real
            # constraint is the custom VRAM_MB resource.
            deployment_config = model_entry.deployment_config.copy()
            ray_actor_options = deployment_config.get("ray_actor_options", {})
            ray_actor_options["num_gpus"] = 0.01 
            ray_actor_options["resources"] = ray_actor_options.get("resources", {})
            ray_actor_options["resources"]["VRAM_MB"] = deployment_config.pop("vram_required")
            deployment_config["ray_actor_options"] = ray_actor_options
            
            # Log the strategy being used - Ray Serve will handle placement internally
            replica_count = deployment_config.get("num_replicas", deployment_config.get("min_replicas", 1))
            replica_resources = {
                "GPU": ray_actor_options["num_gpus"],
                "CPU": ray_actor_options.get("num_cpus", 2),
                "VRAM_MB": ray_actor_options["resources"]["VRAM_MB"]
            }
            # Note: Ray Serve handles resource allocation and placement automatically
            _get_placement_group_config(strategy, replica_count, replica_resources)
            
            class_choice_decorated: serve.Deployment = serve.deployment(
                _func_or_class=class_choice,
                name=ENGINE_CLASS_NAMES[model_entry.engine] + ":" + model_entry.id,
                **deployment_config
            )
            LOCAL_MODEL_BINDINGS[model_entry.id] = class_choice_decorated.bind(
                model_config=model_entry,
                max_model_len=model_entry.max_model_len,
            )

    embedding_models = {}
    if global_config.enabled_model_classes.embedding:
        for embedding_model in global_config.other_local_models.embedding_models:
            assert "vram_required" in embedding_model.deployment_config, f"No VRAM requirement specified for {embedding_model.id}"
            
            deployment_config = embedding_model.deployment_config.copy()
            ray_actor_options = deployment_config.get("ray_actor_options", {})
            ray_actor_options["num_gpus"] = 0.01
            ray_actor_options["resources"] = ray_actor_options.get("resources", {})
            ray_actor_options["resources"]["VRAM_MB"] = deployment_config.pop("vram_required", None)
            
            deployment_config["ray_actor_options"] = ray_actor_options
            
            # Log the strategy being used - Ray Serve will handle placement internally
            replica_count = deployment_config.get("num_replicas", deployment_config.get("min_replicas", 1))
            replica_resources = {
                "GPU": ray_actor_options["num_gpus"],
                "CPU": ray_actor_options.get("num_cpus", 2),
                "VRAM_MB": ray_actor_options["resources"]["VRAM_MB"]
            }
            # Note: Ray Serve handles resource allocation and placement automatically
            _get_placement_group_config(strategy, replica_count, replica_resources)
            
            
            emb_pg_bundle = replica_resources.copy()
            p_group_strategy = strategy
            
            class_choice_decorated : serve.Deployment = serve.deployment(
                _func_or_class=EmbeddingDeployment,
                name="embedding" + ":" + embedding_model.id,
                **deployment_config,
                placement_group_bundles=[emb_pg_bundle],
                placement_group_strategy=p_group_strategy
            )
            embedding_models[embedding_model.id] = class_choice_decorated.bind(
                model_card=embedding_model
            )

    rerank_models = {}
    if global_config.enabled_model_classes.rerank:
        for rerank_model in global_config.other_local_models.rerank_models:
            assert "vram_required" in rerank_model.deployment_config, f"No VRAM requirement specified for {rerank_model.id}"
            
            deployment_config = rerank_model.deployment_config.copy()
            ray_actor_options = deployment_config.get("ray_actor_options", {})
            ray_actor_options["num_gpus"] = 0.01
            ray_actor_options["resources"] = ray_actor_options.get("resources", {})
            ray_actor_options["resources"]["VRAM_MB"] = deployment_config.pop("vram_required", None)
            
            deployment_config["ray_actor_options"] = ray_actor_options
            
            # Log the strategy being used - Ray Serve will handle placement internally
            replica_count = deployment_config.get("num_replicas", deployment_config.get("min_replicas", 1))
            replica_resources = {
                "GPU": ray_actor_options["num_gpus"],
                "CPU": ray_actor_options.get("num_cpus", 2),
                "VRAM_MB": ray_actor_options["resources"]["VRAM_MB"]
            }
            # Note: Ray Serve handles resource allocation and placement automatically
            _get_placement_group_config(strategy, replica_count, replica_resources)
            
            class_choice_decorated : serve.Deployment = serve.deployment(
                _func_or_class=RerankerDeployment,
                name="rerank" + ":" + rerank_model.id,
                **deployment_config
            )
            
            rerank_models[rerank_model.id] = class_choice_decorated.bind(
                model_card=rerank_model
            )
    
    surya_handles = {}
    if global_config.enabled_model_classes.surya:
        surya_model_map = {
            "marker": MarkerDeployment,
            "surya_detection": SuryaDetectionDeployment,
            "surya_layout": SuryaLayoutDeployment,
            "surya_recognition": SuryaOCRDeployment,
            "surya_order": SuryaOrderDeployment,
            "surya_table_recognition": SuryaTableDeployment,
            "surya_texify": SuryaTexifyDeployment
        }
        for surya_model in global_config.other_local_models.surya_models:
            class_to_use = surya_model_map[surya_model.id]
            assert "vram_required" in surya_model.deployment_config, f"No VRAM requirement specified for {surya_model.name}"
            
            deployment_config = surya_model.deployment_config.copy()
            ray_actor_options = deployment_config.get("ray_actor_options", {})
            ray_actor_options["num_gpus"] = 0.01
            ray_actor_options["resources"] = ray_actor_options.get("resources", {})
            ray_actor_options["resources"]["VRAM_MB"] = deployment_config.pop("vram_required", None)
            
            deployment_config["ray_actor_options"] = ray_actor_options
            
            # Log the strategy being used - Ray Serve will handle placement internally
            replica_count = deployment_config.get("num_replicas", deployment_config.get("min_replicas", 1))
            replica_resources = {
                "GPU": ray_actor_options["num_gpus"],
                "CPU": ray_actor_options.get("num_cpus", 2),
                "VRAM_MB": ray_actor_options["resources"]["VRAM_MB"]
            }
            # Note: Ray Serve handles resource allocation and placement automatically
            _get_placement_group_config(strategy, replica_count, replica_resources)
            
            deployment_class = serve.deployment(
                _func_or_class=class_to_use,
                name=f"surya:{surya_model.id}",
                **deployment_config
            )
            surya_handles[surya_model.name] = deployment_class.bind(
                model_card=surya_model
            )

    # Bind the main UmbrellaClass with all the model handles
    deployment = UmbrellaClass.bind(
        configuration=global_config,
        toolchains=toolchains,
        web_scraper_handle=WebScraperDeployment.bind(),
        llm_handles=LOCAL_MODEL_BINDINGS,
        embedding_handles=embedding_models,
        rerank_handles=rerank_models,
        surya_handles=surya_handles
    )

    # Run the application
    # The application is now deployed within the existing Ray cluster
    # started by start_querylake.py
    serve.run(deployment)


SERVER_DIR = os.path.dirname(os.path.realpath(__file__))
os.chdir(SERVER_DIR)

# The old startup logic is now removed.
# The following code is no longer needed as start_querylake.py handles it.
# with open("config.json", 'r', encoding='utf-8') as f:
#     GLOBAL_CONFIG = f.read()
#     f.close()
# GLOBAL_CONFIG, TOOLCHAINS = Config.model_validate_json(GLOBAL_CONFIG), {}
# ... and so on ...
# if __name__ == "__main__":
#    serve.run(deployment)
