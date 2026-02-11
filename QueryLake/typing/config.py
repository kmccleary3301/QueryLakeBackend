from typing import List, Dict, Optional, Union, Tuple, Literal, Annotated, Any
from pydantic import BaseModel
from ..database.sql_db_tables import user
from fastapi.security import OAuth2PasswordRequestForm
from fastapi import Depends
from fastapi.security import OAuth2PasswordBearer
from .toolchains import ToolChain

class ModelArgs(BaseModel):
    stream: Optional[bool] = False
    verbose: Optional[bool] = False
    temperature: float
    top_k: float
    top_p: float
    max_tokens: int
    repetition_penalty: float
    stop: Optional[List[str]]

class Padding(BaseModel):
    system_instruction_wrap: str
    context_wrap: str
    question_wrap: str
    response_wrap: str


class ModelLoras(BaseModel):
    id : str
    system_path: Optional[str] = None

class Model(BaseModel):
    name: str
    id : str
    source : str
    engine: Optional[Literal["vllm", "exllamav2"]] = "vllm"
    modelcard: str
    system_path: Optional[str] = None
    default_parameters: Dict[str, Any]
    max_model_len: int
    loras: Optional[List[ModelLoras]] = []
    padding: Padding
    default_system_instruction: str
    disabled: Optional[bool] = False
    engine_args: Optional[dict] = {}
    deployment_config: Optional[dict] = None

class ExternalModelProviders(BaseModel):
    name: str
    id: str
    context: int
    modelcard: str

class LoaderParameters(BaseModel):
    temperature: Dict[str, float]
    top_k: Dict
    top_p: Dict
    min_p: Dict
    typical: Dict
    token_repetition_penalty_max: Dict
    token_repetition_penalty_sustain: Dict
    token_repetition_penalty_decay: Dict
    beams: Dict
    beam_length: Dict

class LocalModel(BaseModel):
    name: str
    id: str
    source: Union[str, Dict[str, str]]
    system_path: Optional[Union[str, Dict[str, str]]] = None
    deployment_config: Optional[dict] = {}

class OtherLocalModelsField(BaseModel):
    rerank_models: Optional[List[LocalModel]] = []
    embedding_models: Optional[List[LocalModel]] = []
    surya_models: Optional[List[LocalModel]] = []
    chandra_models: Optional[List[LocalModel]] = []

class ConfigDefaultModels(BaseModel):
    llm: str
    rerank: str
    embedding: str

class ConfigEnabledModelClasses(BaseModel):
    llm: bool
    rerank: bool
    embedding: bool
    surya: bool
    chandra: bool = False

class RayClusterConfig(BaseModel):
    head_port: int
    dashboard_port: int
    worker_port_base: int
    worker_port_step: int
    default_gpu_strategy: str
    # Optional physical GPU indices (as reported by nvidia-smi / pynvml) to exclude
    # from Ray worker node startup. This is useful when some GPUs are reserved for
    # external services (e.g., Chandra external vLLM servers) or other workloads.
    worker_gpu_exclude_indices: Optional[List[int]] = None

class Config(BaseModel):
    default_toolchain: str
    default_models: ConfigDefaultModels
    enabled_model_classes: ConfigEnabledModelClasses
    models: List[Model]
    external_model_providers: Dict[str, List[ExternalModelProviders]]
    providers: Optional[List[str]] = []
    other_local_models: OtherLocalModelsField
    ray_cluster: RayClusterConfig
    vllm_upstream_base_url: Optional[str] = None
    vllm_upstream_model_map: Optional[Dict[str, str]] = None
    chandra_vllm_server_base_url: Optional[str] = None
    chandra_vllm_server_base_urls: Optional[List[str]] = None
    chandra_vllm_server_model: Optional[str] = None
    # Optional convenience + lifecycle knobs for Chandra external vLLM server mode.
    # These are intentionally coarse; advanced tuning lives in env vars and
    # scripts/chandra_vllm_server.sh.
    chandra_vllm_server_topology: Optional[Literal["single", "striped"]] = None
    chandra_vllm_server_autostart: bool = False
    # Default to a single-GPU Chandra server to keep one GPU available for other local models.
    # Use "striped" explicitly when you want to dedicate 2 GPUs to OCR throughput.
    chandra_vllm_server_autostart_topology: Literal["single", "striped", "auto"] = "single"
    chandra_vllm_server_autostart_port_base: int = 8022
    chandra_vllm_server_autostart_gpu_memory_utilization: float = 0.90
    chandra_vllm_server_autostart_api_key: Optional[str] = None
    
class ChatHistoryEntry(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str



# Authentication

# We don't want AuthType1 used for session persistence. 
# It basically makes password_prehash another password,
# and one that clients store in session/cookies, which is bad practice.
# password_prehash's only purpose should be for encryption.

class AuthType1(BaseModel):
    username: str
    password_prehash: str

class AuthType2(BaseModel):
    api_key: str
    
class AuthType3(BaseModel):
    username: str
    password: str

class AuthType4(BaseModel):
    oauth2: str

class getUserAuthType(BaseModel):
    username: str
    password_prehash: str

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

AuthType = Union[AuthType1, AuthType2, AuthType3, AuthType4, str]

getUserType = Tuple[user, getUserAuthType]


AuthInputType = Union[AuthType, dict, str]
