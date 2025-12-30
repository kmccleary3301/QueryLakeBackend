import json, os, re, math, sys
import inspect
from copy import deepcopy
import logging
from pynvml import (
    nvmlInit, 
    nvmlDeviceGetCount, 
    nvmlDeviceGetHandleByIndex, 
    nvmlDeviceGetMemoryInfo
)

from fastapi import FastAPI, UploadFile, Request, WebSocket, HTTPException, status
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware import Middleware
from sqlalchemy import text
from typing import Literal, Tuple, Union, List, Dict, Awaitable, Callable, Optional, Any

from ray import serve
from ray.serve.handle import DeploymentHandle
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from QueryLake.api import api
from QueryLake.database import database_admin_operations
from QueryLake.database.create_db_session import initialize_database_engine
from QueryLake.typing.config import Config, AuthType, Model
from QueryLake.typing.toolchains import ToolChain, ToolChainV2
from QueryLake.operation_classes.ray_vllm_class import VLLMDeploymentClass
from QueryLake.operation_classes.ray_embedding_class import EmbeddingDeployment
from QueryLake.operation_classes.ray_reranker_class import RerankerDeployment
from QueryLake.operation_classes.ray_web_scraper import WebScraperDeployment
from sse_starlette.sse import EventSourceResponse
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


from QueryLake.api.single_user_auth import (
    global_public_key,
    global_private_key,
    process_input_as_auth_type,
    OAUTH_SECRET_KEY,
)
from QueryLake.api.auth_utils import resolve_bearer_auth_header
from QueryLake.misc_functions.external_providers import external_llm_count_tokens
from QueryLake.misc_functions.server_class_functions import find_function_calls
from QueryLake.routing.ws_toolchain import toolchain_websocket_handler
from QueryLake.routing.openai_completions import (
    openai_chat_completion, 
    openai_create_embedding,
)
from QueryLake.routing.api_call_router import api_general_call
from QueryLake.routing.llm_call import llm_call
from QueryLake.routing.upload_documents import handle_document
from QueryLake.routing.misc_models import (
    embedding_call,
    rerank_call,
    web_scrape_call
)
from QueryLake.runtime.service import ToolchainRuntimeService
from QueryLake.observability import metrics
from QueryLake.database import sql_db_tables as T
from jose import jwt
from QueryLake.files.service import FilesRuntimeService


def _ensure_logging_configured() -> None:
    """Ensure all QueryLake processes have a sane logging configuration.

    Ray Serve replicas often start in fresh interpreter processes without the
    root logger being configured. When that happens, module-level loggers default
    to WARNING and emit nothing, making debugging nearly impossible. We install
    a basic StreamHandler the first time we import the server module so that
    INFO-level logs propagate into Ray's log aggregation by default.
    """
    root_logger = logging.getLogger()
    has_stream_handler = any(
        isinstance(handler, logging.StreamHandler) for handler in root_logger.handlers
    )
    if not has_stream_handler:
        stream_handler = logging.StreamHandler(stream=sys.stdout)
        stream_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
        )
        root_logger.addHandler(stream_handler)
    level_name = os.environ.get("QUERYLAKE_LOG_LEVEL", "INFO").upper()
    try:
        root_logger.setLevel(level_name)
    except (ValueError, TypeError):
        root_logger.setLevel(logging.INFO)
    for handler in root_logger.handlers:
        try:
            handler.setLevel(root_logger.level)
        except Exception:  # pragma: no cover - defensive
            continue
    logging.captureWarnings(True)


_ensure_logging_configured()

logger = logging.getLogger(__name__)

def _recommend_gpu_memory_fraction(
    required_vram: int,
    max_model_len: Optional[int],
    per_node_capacity: Optional[int],
    configured_fraction: Optional[float],
) -> float:
    """Heuristically choose a GPU memory utilization fraction for vLLM models."""
    try:
        configured = float(configured_fraction) if configured_fraction is not None else 0.0
    except (TypeError, ValueError):
        configured = 0.0

    if per_node_capacity and per_node_capacity > 0:
        base_fraction = required_vram / per_node_capacity
        context_bonus = 0.0
        if max_model_len:
            context_bonus = max(0.0, (max_model_len - 8192) / 16384.0) * 0.15
        heuristic = min(0.9, base_fraction + context_bonus)

        if configured <= 0.0:
            candidate = heuristic
        elif configured + 1e-3 < heuristic:
            candidate = heuristic
        else:
            candidate = min(configured, 0.9)

        candidate = max(candidate, base_fraction)
    else:
        candidate = max(configured, 0.4)

    return max(0.3, min(candidate, 0.9))


def _select_gpu_fraction(
    existing_fraction: Optional[float],
    recommended_fraction: float,
) -> float:
    """Reserve a slightly larger Ray GPU fraction to avoid oversubscription."""
    try:
        existing = float(existing_fraction) if existing_fraction is not None else 0.0
    except (TypeError, ValueError):
        existing = 0.0

    target = max(existing, recommended_fraction + 0.05)
    if target >= 0.99:
        return 1.0
    return max(round(target, 3), 0.01)


def _estimate_reserved_vram(
    per_node_capacity: Optional[int],
    required_vram: int,
    util_fraction: float,
    gpu_fraction: float,
) -> int:
    """Derive the VRAM resource reservation to hand to Ray."""
    if not per_node_capacity or per_node_capacity <= 0:
        return required_vram

    projected_fraction = max(util_fraction, gpu_fraction)
    return max(
        required_vram,
        int(math.ceil(per_node_capacity * projected_fraction)),
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
    
    logger.debug("Cleaning args with %s and %s", function_args, keys_get)
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
        toolchains_v1: Dict[str, ToolChain],
        toolchains_v2: Dict[str, ToolChainV2],
        web_scraper_handle: DeploymentHandle,
        llm_handles: Dict[str, DeploymentHandle] = {},
        embedding_handles: Dict[str, DeploymentHandle] = None,
        rerank_handles: Dict[str, DeploymentHandle] = None,
        surya_handles: Dict[str, DeploymentHandle] = None,
    ):
        logger.info("Initializing UmbrellaClass deployment")
        self.config : Config = configuration
        self.toolchain_configs : Dict[str, ToolChain] = toolchains_v1
        self.toolchain_configs_v2 : Dict[str, ToolChainV2] = toolchains_v2
        
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
        
        logger.info("Connecting to database and synchronizing metadata")
        try:
            self.database, self.engine = initialize_database_engine()
            logger.info("Database connection established")
            database_admin_operations.add_models_to_database(self.database, self.config.models)
            database_admin_operations.add_toolchains_to_database(self.database, self.toolchain_configs)
            logger.info("Database model/toolchain sync complete")
        except Exception as exc:
            logger.exception("Failed to initialize database session: %s", exc)
            raise

        self.toolchain_runtime = ToolchainRuntimeService(
            umbrella=self,
            toolchains_v1=self.toolchain_configs,
            toolchains_v2=self.toolchain_configs_v2,
        )
        self.files_runtime = FilesRuntimeService(self.database, umbrella=self)

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
            "toolchain_runtime_service": self.toolchain_runtime,
            "umbrella": self,
            "job_signal_bus": self.toolchain_runtime.signal_bus,
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
        
        logger.info("UmbrellaClass initialization complete")
        logger.debug("Enabled model classes: %s", self.config.enabled_model_classes)

    async def _resolve_auth(self, request: Request, auth_payload: Optional[dict]) -> AuthType:
        if auth_payload is not None:
            return process_input_as_auth_type(auth_payload)
        auth_header = request.headers.get("authorization")
        if auth_header and auth_header.lower().startswith("bearer "):
            return resolve_bearer_auth_header(auth_header)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")

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
        raw_request : Request
    ):
        return await openai_chat_completion(self, raw_request)
    
    @fastapi_app.post("/v1/embeddings")
    async def openai_embedding_endpoint(
        self,
        raw_request: Request
    ):
        return await openai_create_embedding(self, raw_request)
    
    
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
        logger.debug("Ping received from %s", req.client.host if req.client else "unknown")
        return {"success": True, "note": "Pong"}

    # -----------------
    # Files v1 endpoints
    # -----------------
    @fastapi_app.post("/files")
    async def upload_file_endpoint(self, request: Request, file: UploadFile, logical_name: Optional[str] = None, collection_id: Optional[str] = None):
        auth = await self._resolve_auth(request, None)
        checksum = request.headers.get("x-checksum-sha256")
        idem = request.headers.get("idempotency-key")
        result = await self.files_runtime.upload_file(
            auth,
            file,
            logical_name=logical_name,
            collection_id=collection_id,
            checksum_sha256=checksum,
            idempotency_key=idem,
        )
        return {"success": True, **result}

    @fastapi_app.get("/files/{file_id}")
    async def get_file_endpoint(self, file_id: str, request: Request):
        auth = await self._resolve_auth(request, None)
        result = await self.files_runtime.list_file(file_id, auth=auth)
        return {"success": True, **result}

    @fastapi_app.get("/files/{file_id}/versions")
    async def get_file_versions_endpoint(self, file_id: str, request: Request):
        auth = await self._resolve_auth(request, None)
        result = await self.files_runtime.list_versions(file_id, auth=auth)
        return {"success": True, "versions": result}

    @fastapi_app.get("/files/{file_id}/events")
    async def get_file_events_endpoint(self, file_id: str, request: Request, since: Optional[int] = None):
        auth = await self._resolve_auth(request, None)
        result = await self.files_runtime.list_events(file_id, since, auth=auth)
        return {"success": True, "events": result}

    @fastapi_app.get("/files/{file_id}/jobs")
    async def get_file_jobs_endpoint(self, file_id: str, request: Request):
        auth = await self._resolve_auth(request, None)
        result = await self.files_runtime.list_jobs(file_id, auth=auth)
        return {"success": True, "jobs": result}

    @fastapi_app.get("/files/{file_id}/stream")
    async def files_stream_endpoint(self, file_id: str, request: Request):
        auth = await self._resolve_auth(request, None)
        _ = auth
        subscriber = await self.files_runtime.subscribe(file_id)

        last_event_id_str = request.headers.get("last-event-id") or request.query_params.get("last_event_id")
        since_rev = None
        if last_event_id_str is not None:
            try:
                since_rev = int(last_event_id_str)
            except ValueError:
                since_rev = None

        if since_rev is not None:
            history = await self.files_runtime.list_events(file_id, since_rev, auth=auth)
            for event in history:
                await subscriber.push({"event": event.get("kind"), "data": json.dumps(event), "id": event.get("rev")})

        async def event_generator():
            try:
                async for message in subscriber.stream():
                    yield message
            finally:
                await self.files_runtime.unsubscribe(file_id, subscriber)

        return EventSourceResponse(event_generator())

    @fastapi_app.post("/files/{file_id}/versions/{version_id}/process")
    async def process_file_version_endpoint(self, file_id: str, version_id: str, request: Request):
        auth = await self._resolve_auth(request, None)
        result = await self.files_runtime.process_version(file_id, version_id, auth=auth)
        return {"success": True, **result}

    @fastapi_app.get("/files/cas/{sha256}")
    async def get_file_bytes_cas(self, sha256: str, request: Request):
        auth = await self._resolve_auth(request, None)
        # Resolve permission by locating any file_version referencing CAS
        from sqlmodel import select
        matches = self.database.exec(select(T.file_version).where(T.file_version.bytes_cas == sha256)).all()
        for fv in matches:
            try:
                await self.files_runtime.list_file(fv.file_id, auth=auth)
                data = self.files_runtime.store.get_bytes(sha256)
                if data is None:
                    raise HTTPException(status_code=404, detail="Object not found")
                return Response(content=data, media_type="application/octet-stream")
            except HTTPException:
                continue
        raise HTTPException(status_code=404, detail="Object not found")

    @fastapi_app.post("/files/presign")
    async def presign_file_download(self, request: Request):
        body = await request.json()
        auth = await self._resolve_auth(request, body.get("auth"))
        file_id = body.get("file_id")
        version_id = body.get("version_id")
        bytes_cas = body.get("bytes_cas")
        expires_in = int(body.get("expires_in", 300))
        _username = (await self.toolchain_runtime.get_session_state if False else None)  # placeholder to quiet linters
        # Permission gate via file listing
        if file_id:
            await self.files_runtime.list_file(file_id, auth=auth)
        cas = bytes_cas
        if not cas:
            if not (file_id and version_id):
                raise HTTPException(status_code=400, detail="Provide either bytes_cas or (file_id, version_id)")
            fv = self.database.get(T.file_version, version_id)
            if fv is None or fv.file_id != file_id:
                raise HTTPException(status_code=404, detail="version not found")
            cas = fv.bytes_cas
        # Build token
        from datetime import datetime, timedelta, timezone
        username = self.files_runtime._username_from_auth(auth)
        payload = {"cas": cas, "sub": username or "files", "exp": datetime.now(timezone.utc) + timedelta(seconds=expires_in)}
        token = jwt.encode(payload, OAUTH_SECRET_KEY, algorithm="HS256")
        return {"success": True, "token": token, "expires_in": expires_in}

    @fastapi_app.get("/files/download/{token}")
    async def download_presigned(self, token: str):
        try:
            payload = jwt.decode(token, OAUTH_SECRET_KEY, algorithms=["HS256"])
            cas = payload.get("cas")
            if not cas:
                raise HTTPException(status_code=400, detail="Invalid token")
            data = self.files_runtime.store.get_bytes(cas)
            if data is None:
                raise HTTPException(status_code=404, detail="Object not found")
            return Response(content=data, media_type="application/octet-stream")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid or expired token: {e}")

    @fastapi_app.get("/metrics")
    async def metrics_endpoint(self):
        body, content_type = metrics.expose_metrics()
        return Response(content=body, media_type=content_type)

    @fastapi_app.get("/healthz")
    async def healthz(self):
        return {"ok": True}

    @fastapi_app.get("/readyz")
    async def readyz(self):
        try:
            self.database.exec(text("SELECT 1"))
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"db not ready: {e}")
        return {"ok": True, "db": "ok", "models": list(self.llm_configs.keys())}
    
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

    @fastapi_app.post("/sessions")
    async def create_session_endpoint(self, request: Request):
        body = await request.json()
        auth = await self._resolve_auth(request, body.get("auth"))
        toolchain_id = body.get("toolchain_id")
        if not toolchain_id:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="toolchain_id required")
        title = body.get("title")
        initial_inputs = body.get("initial_inputs")
        result = await self.toolchain_runtime.create_session(auth, toolchain_id, title, initial_inputs)
        return {"success": True, **result}

    @fastapi_app.get("/sessions/{session_id}")
    async def get_session_endpoint(self, session_id: str, request: Request):
        auth = await self._resolve_auth(request, None)
        state = await self.toolchain_runtime.get_session_state(session_id, auth)
        return {"success": True, **state}

    @fastapi_app.delete("/sessions/{session_id}")
    async def delete_session_endpoint(self, session_id: str, request: Request):
        auth = await self._resolve_auth(request, None)
        await self.toolchain_runtime.delete_session(session_id, auth)
        return {"success": True}

    @fastapi_app.post("/sessions/{session_id}/event")
    async def post_session_event(self, session_id: str, request: Request):
        body = await request.json()
        auth = await self._resolve_auth(request, body.get("auth"))
        node_id = body.get("node_id")
        if not node_id:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="node_id required")
        inputs = body.get("inputs", {})
        expected_rev = body.get("rev")
        correlation_id = body.get("correlation_id")
        result = await self.toolchain_runtime.post_event(
            session_id,
            node_id,
            inputs,
            expected_rev,
            auth=auth,
            correlation_id=correlation_id,
        )
        return {"success": True, **result}

    @fastapi_app.get("/sessions/{session_id}/events")
    async def list_session_events(self, session_id: str, request: Request, since: Optional[int] = None):
        auth = await self._resolve_auth(request, None)
        events = await self.toolchain_runtime.list_events(session_id, since, auth)
        return {"success": True, "events": [event.model_dump() for event in events]}

    @fastapi_app.get("/sessions/{session_id}/jobs")
    async def list_session_jobs(self, session_id: str, request: Request):
        auth = await self._resolve_auth(request, None)
        jobs = await self.toolchain_runtime.list_jobs(session_id, auth)
        return {"success": True, "jobs": jobs}

    @fastapi_app.post("/sessions/{session_id}/jobs/{job_id}/cancel")
    async def cancel_session_job(self, session_id: str, job_id: str, request: Request):
        auth = await self._resolve_auth(request, None)
        result = await self.toolchain_runtime.cancel_job(session_id, job_id, auth)
        return {"success": True, **result}

    @fastapi_app.get("/sessions/{session_id}/stream")
    async def stream_session(self, session_id: str, request: Request):
        auth = await self._resolve_auth(request, None)
        subscriber = await self.toolchain_runtime.subscribe(session_id, auth)

        last_event_id_str = request.headers.get("last-event-id") or request.query_params.get("last_event_id")
        since_rev = None
        if last_event_id_str is not None:
            try:
                since_rev = int(last_event_id_str)
            except ValueError:
                since_rev = None

        if since_rev is not None:
            history = await self.toolchain_runtime.list_events(session_id, since_rev, auth)
            for event in history:
                await subscriber.push(event.model_dump())

        async def event_generator():
            try:
                async for message in subscriber.stream():
                    payload = {key: value for key, value in message.items() if value is not None}
                    yield payload
            finally:
                await self.toolchain_runtime.unsubscribe(session_id, subscriber)

        return EventSourceResponse(event_generator())

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

def _get_placement_group_config(
    strategy: str,
    num_replicas: int,
    replica_resources: Optional[Dict[str, Union[int, float]]] = None,
) -> Dict[str, Any]:
    """Build the Serve placement-group configuration for a deployment."""
    if not replica_resources:
        logger.debug(
            "No replica resource specification provided; skipping placement group setup."
        )
        return {}

    pg_config: Dict[str, Any] = {
        "placement_group_bundles": [replica_resources],
        "placement_group_strategy": strategy,
    }

    logger.info(
        "Applying placement group strategy=%s replicas=%s resources=%s",
        strategy,
        num_replicas,
        replica_resources,
    )
    return pg_config

def _get_cluster_vram_capacity() -> Optional[int]:
    """Return the total VRAM budget registered with the Ray cluster."""
    try:
        import ray
        resources = ray.cluster_resources()
        if not resources:
            return None
        capacity = resources.get("VRAM_MB")
        if capacity is None:
            return None
        return int(capacity)
    except Exception as exc:  # pragma: no cover - best effort logging
        logger.debug("Unable to determine cluster VRAM capacity: %s", exc)
        return None


def _get_max_vram_per_node() -> Optional[int]:
    """Return the maximum VRAM_MB advertised by any alive Ray node."""
    try:
        import ray
        max_vram = 0
        for node in ray.nodes():
            if not node.get("alive", False):
                continue
            node_vram = int(node.get("Resources", {}).get("VRAM_MB", 0))
            if node_vram > max_vram:
                max_vram = node_vram
        return max_vram or None
    except Exception as exc:  # pragma: no cover - best effort logging
        logger.debug("Unable to determine per-node VRAM capacity: %s", exc)
        return None


def _prioritize_by_id(items: List[Any], preferred_id: Optional[str]) -> List[Any]:
    """Ensure the preferred item appears first while preserving order of others."""
    if not preferred_id:
        return list(items)
    preferred: List[Any] = []
    others: List[Any] = []
    swapped = False
    for item in items:
        if getattr(item, "id", None) == preferred_id and not swapped:
            preferred.append(item)
            swapped = True
        else:
            others.append(item)
    return preferred + others


class _VRAMBudget:
    """Simple tracker for reserving VRAM_MB resources across deployments."""

    def __init__(self, capacity: Optional[int]) -> None:
        self.capacity = capacity if capacity and capacity > 0 else None
        self.reserved = 0

    def track(self, amount: Optional[int], label: str) -> None:
        if amount is None:
            return
        try:
            required = int(amount)
        except (ValueError, TypeError):
            logger.warning("Invalid VRAM requirement for %s: %s", label, amount)
            return
        if required <= 0:
            return

        if self.capacity is None:
            self.reserved += required
            logger.debug("Reserved %s MB VRAM for %s (capacity unknown)", required, label)
            return

        remaining = self.capacity - self.reserved
        if required > remaining:
            logger.warning(
                "VRAM budget nearly exhausted: %s needs %s MB, only %s MB remaining (cluster capacity %s MB)",
                label,
                required,
                max(0, remaining),
                self.capacity,
            )
        self.reserved += required

    def log_summary(self) -> None:
        if self.capacity is not None:
            logger.info(
                "Total VRAM requested by configured deployments: %s MB (cluster budget %s MB)",
                self.reserved,
                self.capacity,
            )


def build_and_run_application(
    strategy: str,
    global_config: Config,
    toolchains_v1: Dict[str, ToolChain],
    toolchains_v2: Dict[str, ToolChainV2],
):
    """
    Builds the Ray Serve application graph and runs it.
    This function is called by the new CLI orchestrator.
    """
    global_config = global_config.model_copy(deep=True)

    cluster_capacity = _get_cluster_vram_capacity()
    per_node_capacity = _get_max_vram_per_node()
    vram_budget = _VRAMBudget(cluster_capacity)
    if cluster_capacity is not None:
        logger.info(
            "Detected %s MB of VRAM resources across the cluster for Serve deployments",
            cluster_capacity,
        )
    if per_node_capacity is not None:
        logger.info(
            "Largest GPU advertises %s MB of VRAM; deployments requiring more will fail to start",
            per_node_capacity,
        )

    LOCAL_MODEL_BINDINGS: Dict[str, DeploymentHandle] = {}
    ENGINE_CLASSES = {"vllm": VLLMDeploymentClass}
    ENGINE_CLASS_NAMES = {"vllm": "vllm"}

    embedding_models: Dict[str, DeploymentHandle] = {}
    if global_config.enabled_model_classes.embedding:
        preferred_embedding = getattr(global_config.default_models, "embedding", None)
        embedding_iterable = _prioritize_by_id(
            list(global_config.other_local_models.embedding_models),
            preferred_embedding,
        )
        for embedding_model in embedding_iterable:
            deployment_config = (embedding_model.deployment_config or {}).copy()
            required_vram = deployment_config.pop("vram_required", None)
            assert (
                required_vram is not None
            ), f"No VRAM requirement specified for {embedding_model.id}"

            if per_node_capacity is not None and required_vram > per_node_capacity:
                logger.warning(
                    "Embedding model %s declares %s MB VRAM requirement, but largest GPU only advertises %s MB. "
                    "Ray may still schedule it if memory fits at runtime.",
                    embedding_model.id,
                    required_vram,
                    per_node_capacity,
                )
            vram_budget.track(required_vram, f"embedding:{embedding_model.id}")

            ray_actor_options = deployment_config.get("ray_actor_options", {})
            ray_actor_options["num_gpus"] = max(ray_actor_options.get("num_gpus", 0.01), 0.01)
            ray_actor_options["resources"] = ray_actor_options.get("resources", {})
            ray_actor_options["resources"]["VRAM_MB"] = required_vram
            deployment_config["ray_actor_options"] = ray_actor_options

            replica_count = deployment_config.get(
                "num_replicas", deployment_config.get("min_replicas", 1)
            )
            replica_resources = {
                "GPU": ray_actor_options["num_gpus"],
                "CPU": ray_actor_options.get("num_cpus", 2),
                "VRAM_MB": ray_actor_options["resources"]["VRAM_MB"],
            }
            pg_config = _get_placement_group_config(strategy, replica_count, replica_resources)
            deployment_kwargs = {**pg_config, **deployment_config}

            class_choice_decorated: serve.Deployment = serve.deployment(
                _func_or_class=EmbeddingDeployment,
                name="embedding" + ":" + embedding_model.id,
                **deployment_kwargs,
            )
            embedding_models[embedding_model.id] = class_choice_decorated.bind(
                model_card=embedding_model
            )

    rerank_models: Dict[str, DeploymentHandle] = {}
    if global_config.enabled_model_classes.rerank:
        preferred_rerank = getattr(global_config.default_models, "rerank", None)
        rerank_iterable = _prioritize_by_id(
            list(global_config.other_local_models.rerank_models),
            preferred_rerank,
        )
        for rerank_model in rerank_iterable:
            deployment_config = (rerank_model.deployment_config or {}).copy()
            required_vram = deployment_config.pop("vram_required", None)
            assert (
                required_vram is not None
            ), f"No VRAM requirement specified for {rerank_model.id}"

            if per_node_capacity is not None and required_vram > per_node_capacity:
                logger.warning(
                    "Rerank model %s declares %s MB VRAM requirement, but largest GPU only advertises %s MB.",
                    rerank_model.id,
                    required_vram,
                    per_node_capacity,
                )
            vram_budget.track(required_vram, f"rerank:{rerank_model.id}")

            ray_actor_options = deployment_config.get("ray_actor_options", {})
            ray_actor_options["num_gpus"] = max(ray_actor_options.get("num_gpus", 0.01), 0.01)
            ray_actor_options["resources"] = ray_actor_options.get("resources", {})
            ray_actor_options["resources"]["VRAM_MB"] = required_vram
            deployment_config["ray_actor_options"] = ray_actor_options

            replica_count = deployment_config.get(
                "num_replicas", deployment_config.get("min_replicas", 1)
            )
            replica_resources = {
                "GPU": ray_actor_options["num_gpus"],
                "CPU": ray_actor_options.get("num_cpus", 2),
                "VRAM_MB": ray_actor_options["resources"]["VRAM_MB"],
            }
            pg_config = _get_placement_group_config(strategy, replica_count, replica_resources)
            deployment_kwargs = {**pg_config, **deployment_config}

            class_choice_decorated: serve.Deployment = serve.deployment(
                _func_or_class=RerankerDeployment,
                name="rerank" + ":" + rerank_model.id,
                **deployment_kwargs,
            )

            rerank_models[rerank_model.id] = class_choice_decorated.bind(
                model_card=rerank_model
            )

    updated_llm_models: Dict[str, Model] = {}
    if global_config.enabled_model_classes.llm:
        preferred_llm = getattr(global_config.default_models, "llm", None)
        llm_iterable = _prioritize_by_id(
            [model for model in global_config.models],
            preferred_llm,
        )
        for model_entry in llm_iterable:
            if model_entry.disabled:
                continue
            if model_entry.deployment_config is None:
                logger.warning(
                    "Skipping deployment for model %s: missing deployment_config",
                    model_entry.id,
                )
                continue
            if model_entry.engine not in ENGINE_CLASSES:
                logger.warning(
                    "Skipping deployment for model %s: engine '%s' not recognized",
                    model_entry.id,
                    model_entry.engine,
                )
                continue

            deployment_config = model_entry.deployment_config.copy()
            required_vram = deployment_config.pop("vram_required", None)
            assert (
                required_vram is not None
            ), f"No VRAM requirement specified for {model_entry.id}"

            if per_node_capacity is not None and required_vram > per_node_capacity:
                logger.warning(
                    "Model %s declares %s MB VRAM requirement, but largest GPU only advertises %s MB.",
                    model_entry.id,
                    required_vram,
                    per_node_capacity,
                )
            class_choice = ENGINE_CLASSES[model_entry.engine]

            engine_args = dict(model_entry.engine_args or {})
            raw_current_util = engine_args.get("gpu_memory_utilization")
            try:
                current_util_numeric = (
                    float(raw_current_util)
                    if raw_current_util is not None
                    else None
                )
            except (TypeError, ValueError):
                current_util_numeric = None
            recommended_util = _recommend_gpu_memory_fraction(
                required_vram=required_vram,
                max_model_len=model_entry.max_model_len,
                per_node_capacity=per_node_capacity,
                configured_fraction=current_util_numeric,
            )
            if (
                current_util_numeric is not None
                and recommended_util > current_util_numeric + 1e-3
            ):
                logger.warning(
                    (
                        "Model %s GPU memory utilization increased from %s to %.3f "
                        "to accommodate context length %s."
                    ),
                    model_entry.id,
                    raw_current_util,
                    recommended_util,
                    model_entry.max_model_len,
                )
            engine_args["gpu_memory_utilization"] = round(recommended_util, 3)

            ray_actor_options = deployment_config.get("ray_actor_options", {})
            target_gpu_fraction = _select_gpu_fraction(
                ray_actor_options.get("num_gpus", 0.0),
                recommended_util,
            )
            ray_actor_options["num_gpus"] = target_gpu_fraction
            ray_actor_options["resources"] = ray_actor_options.get("resources", {})
            estimated_vram = _estimate_reserved_vram(
                per_node_capacity=per_node_capacity,
                required_vram=required_vram,
                util_fraction=recommended_util,
                gpu_fraction=target_gpu_fraction,
            )
            ray_actor_options["resources"]["VRAM_MB"] = estimated_vram
            deployment_config["ray_actor_options"] = ray_actor_options

            replica_count = deployment_config.get(
                "num_replicas", deployment_config.get("min_replicas", 1)
            )
            replica_resources = {
                "GPU": ray_actor_options["num_gpus"],
                "CPU": ray_actor_options.get("num_cpus", 2),
                "VRAM_MB": ray_actor_options["resources"]["VRAM_MB"],
            }
            pg_config = _get_placement_group_config(strategy, replica_count, replica_resources)
            deployment_kwargs = {**pg_config, **deployment_config}

            updated_model_entry = model_entry.model_copy(
                update={
                    "engine_args": engine_args,
                    "deployment_config": {
                        **model_entry.deployment_config,
                        "vram_required": estimated_vram,
                        "ray_actor_options": ray_actor_options,
                    },
                }
            )

            updated_llm_models[model_entry.id] = updated_model_entry
            vram_budget.track(estimated_vram, f"llm:{model_entry.id}")

            class_choice_decorated: serve.Deployment = serve.deployment(
                _func_or_class=class_choice,
                name=ENGINE_CLASS_NAMES[model_entry.engine] + ":" + model_entry.id,
                **deployment_kwargs,
            )
            LOCAL_MODEL_BINDINGS[model_entry.id] = class_choice_decorated.bind(
                model_config=updated_model_entry,
                max_model_len=model_entry.max_model_len,
            )
    if updated_llm_models:
        global_config = global_config.model_copy(
            update={
                "models": [
                    updated_llm_models.get(model.id, model)
                    for model in global_config.models
                ]
            }
        )

    surya_handles: Dict[str, DeploymentHandle] = {}
    if global_config.enabled_model_classes.surya:
        surya_model_map = {
            "marker": MarkerDeployment,
            "surya_detection": SuryaDetectionDeployment,
            "surya_layout": SuryaLayoutDeployment,
            "surya_recognition": SuryaOCRDeployment,
            "surya_order": SuryaOrderDeployment,
            "surya_table_recognition": SuryaTableDeployment,
            "surya_texify": SuryaTexifyDeployment,
        }
        for surya_model in global_config.other_local_models.surya_models:
            class_to_use = surya_model_map[surya_model.id]
            deployment_config = (surya_model.deployment_config or {}).copy()
            required_vram = deployment_config.pop("vram_required", None)
            assert (
                required_vram is not None
            ), f"No VRAM requirement specified for {surya_model.name}"

            if per_node_capacity is not None and required_vram > per_node_capacity:
                logger.warning(
                    "Surya model %s declares %s MB VRAM requirement, but largest GPU only advertises %s MB.",
                    surya_model.id,
                    required_vram,
                    per_node_capacity,
                )
            vram_budget.track(required_vram, f"surya:{surya_model.id}")

            ray_actor_options = deployment_config.get("ray_actor_options", {})
            ray_actor_options["num_gpus"] = max(ray_actor_options.get("num_gpus", 0.01), 0.01)
            ray_actor_options["resources"] = ray_actor_options.get("resources", {})
            ray_actor_options["resources"]["VRAM_MB"] = required_vram
            deployment_config["ray_actor_options"] = ray_actor_options

            replica_count = deployment_config.get(
                "num_replicas", deployment_config.get("min_replicas", 1)
            )
            replica_resources = {
                "GPU": ray_actor_options["num_gpus"],
                "CPU": ray_actor_options.get("num_cpus", 2),
                "VRAM_MB": ray_actor_options["resources"]["VRAM_MB"],
            }
            pg_config = _get_placement_group_config(strategy, replica_count, replica_resources)
            deployment_kwargs = {**pg_config, **deployment_config}

            deployment_class = serve.deployment(
                _func_or_class=class_to_use,
                name=f"surya:{surya_model.id}",
                **deployment_kwargs,
            )
            surya_handles[surya_model.name] = deployment_class.bind(
                model_card=surya_model
            )

    vram_budget.log_summary()

    deployment = UmbrellaClass.bind(
        configuration=global_config,
        toolchains_v1=toolchains_v1,
        toolchains_v2=toolchains_v2,
        web_scraper_handle=WebScraperDeployment.bind(),
        llm_handles=LOCAL_MODEL_BINDINGS,
        embedding_handles=embedding_models,
        rerank_handles=rerank_models,
        surya_handles=surya_handles,
    )

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
