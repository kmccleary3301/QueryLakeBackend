from typing import List, Dict, Optional, Union, Tuple, Literal
from pydantic import BaseModel
from ..database.sql_db_tables import user

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

class Model(BaseModel):
    name: str
    id : str
    quantization: Optional[Literal["awq", "gptq", "squeezellm"]]
    modelcard: str
    system_path: str
    default_parameters: ModelArgs
    max_model_len: int
    padding: Padding
    default_system_instructions: str

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

class Config(BaseModel):
    default_model: str
    models: List[Model]
    external_model_providers: Dict[str, List[ExternalModelProviders]]
    
    

class ChatHistoryEntry(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str




class AuthType1(BaseModel):
    username: str
    password_prehash: str

class AuthType2(BaseModel):
    api_key: str

class getUserAuthType(BaseModel):
    username: str
    password_prehash: str

AuthType = Union[AuthType1, AuthType2]

getUserType = Tuple[user, getUserAuthType]

