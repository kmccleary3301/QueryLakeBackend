from typing import Optional, List, Literal
from sqlmodel import Field, SQLModel, ARRAY, String, Integer, Float, JSON, LargeBinary

from sqlalchemy.sql.schema import Column
from sqlmodel import Session, create_engine
from pgvector.sqlalchemy import Vector
from sqlalchemy import Column
from ..api.hashing import random_hash
import pgvector

def data_dict(db_entry : SQLModel):
    return {i:db_entry.__dict__[i] for i in db_entry.__dict__ if i != "_sa_instance_state"}


# COLLECTION_TYPES = Literal["user", "organization", "global", "toolchain", "website"]

class DocumentEmbedding(SQLModel, table=True):
    id: Optional[str] = Field(default_factory=random_hash, primary_key=True)
    collection_type: Optional[str] = Field(index=True, default=None)
    document_id: Optional[str] = Field(foreign_key="document_raw.hash_id", default=None)
    document_integrity: Optional[str] = Field(default=None)
    parent_collection_hash_id: Optional[str] = Field(index=True, default=None)
    document_name: str = Field()
    website_url : Optional[str] = Field(default=None)
    embedding: List[float] = Field(sa_column=Column(Vector(1024)))
    text: str = Field()
    private: bool = Field(default=False)

class toolchain_session(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    hash_id: str = Field(index=True, unique=True)
    title: Optional[str] = Field(default=None)
    hidden: Optional[bool] = Field(default=False)
    creation_timestamp: float
    toolchain_id: str = Field(foreign_key="toolchain.toolchain_id", index=True)
    author: str = Field(foreign_key="user.name", index=True)
    state_arguments: Optional[str] = Field(default=None)
    queue_inputs: Optional[str] = Field(default="")
    firing_queue: Optional[str] = Field(default="")

class toolchain(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    toolchain_id: str = Field(index=True, unique=True)
    title: str
    category: str
    content: str #JSON loads this portion.

class document_access_token(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    hash_id: str = Field(index=True, unique=True)
    expiration_timestamp: float


class model(SQLModel, table=True):
    id: str = Field(primary_key=True, unique=True)
    name: str = Field(index=True, unique=True)
    path_on_server: str
    quantization: Optional[str] = Field(default=None) # Only qunatization supported by vLLM, "awq" | "gptq" | "squeezellm"
    default_settings: str #This will be a stringified JSON that will be parsed out kwargs on load.

    # Let's store wrappers as something like prepend_string+"\<system_instruction}\>"+append_string,
    # Where the prepend and append strings are html-encoded.
    # Wrappers will obviously be a single string, but this way we can split into 3 parts.

    system_instruction_wrapper: str 
    context_wrapper: str
    user_question_wrapper: str
    bot_response_wrapper: str

    default_system_instruction: str

class chat_session_new(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)

    title: Optional[str] = Field(default=None)
    hash_id: str = Field(index=True, unique=True)
    model: str = Field(foreign_key="model.name")

    system_instruction: Optional[str] = Field(default=None)

    author_user_name: str = Field(foreign_key="user.name", index=True)
    access_token_id: int = Field(foreign_key="access_token.id", index=True)
    creation_timestamp: float
    tool_used: Optional[str] = Field(default="chat")
    currently_generating: Optional[bool] = Field(default=False)
    hidden: Optional[bool] = Field(default=False)

class chat_entry_user_question(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    chat_session_id: int = Field(foreign_key="chat_session_new.id", index=True)
    timestamp: float
    content: str

class chat_entry_model_response(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    chat_entry_response_to: int = Field(foreign_key="chat_entry_user_question.id")
    chat_session_id: int = Field(foreign_key="chat_session_new.id", index=True)
    timestamp: float
    content: str
    sources: Optional[str] = Field(default=None)

class web_search(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    author_user_name: str = Field(foreign_key="user.name", index=True)
    query: str
    timestamp: float
    organization_hash_id: Optional[str] = Field(foreign_key="organization.hash_id", index=True, default=None)
    result: str

# Decided on

class user(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True, unique=True)
    email: Optional[str] = Field(default="", index=True)
    password_hash: str
    password_salt: str
    creation_timestamp: float
    is_admin: Optional[bool] = Field(default=False)
    public_key: str
    private_key_encryption_salt: str
    private_key_secured: str # Encrypted with salt(password_prehash, encryption_salt)
    serp_api_key_encrypted: Optional[str] = Field(default=None)         # Encrypted with user's public key
    openai_api_key_encrypted: Optional[str] = Field(default=None)       # Encrypted with user's public key

class organization(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    hash_id: str = Field(index=True, unique=True)
    name: str
    creation_timestamp: float
    public_key: str
    serp_api_key_encrypted: Optional[str] = Field(default=None)             # Encrypted with organization's public key
    openai_organization_id_encrypted: Optional[str] = Field(default=None)   # Encrypted with organization's public key

class chat_session(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    author_user_name: str = Field(foreign_key="user.name", index=True)
    access_token_id: int = Field(foreign_key="access_token.id", index=True)
    creation_timestamp: float
    tool_used: Optional[str] = Field(default="chat")
    currently_generating: Optional[bool] = Field(default=False)

class access_token(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    hash_id: str = Field(index=True)
    type: str
    creation_timestamp: float
    author_user_name: str = Field(foreign_key="user.name", index=True)
    tokens_used: Optional[int] = Field(default=0)
    tokens_per_day_limit: Optional[int] = Field(default=500000) # I think 500,000 tokens per day is reasonable. We can revisit this later.

class model_query_raw(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    prompt: str
    response: str
    prompt_size_tokens: int
    response_size_tokens: int
    model: str
    settings: str
    timestamp: float #UnixEpoch
    time_taken: float # In milliseconds
    access_token_id: Optional[int] = Field(foreign_key="access_token.id", index=True, default=None)
    organization_id: Optional[int] = Field(default=None, foreign_key="organization.id", index=True)

class document_raw(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    hash_id: str = Field(index=True, unique=True)
    file_name: str
    server_zip_archive_path: str = Field(unique=True)
    creation_timestamp: float
    integrity_sha256: str = Field(index=True)
    size_bytes: int
    encryption_key_secure: Optional[str] = Field(default=None)
    organization_document_collection_hash_id: Optional[str] = Field(default=None, foreign_key="organization_document_collection.hash_id", index=True)
    user_document_collection_hash_id: Optional[str] = Field(default=None, foreign_key="user_document_collection.hash_id", index=True)
    global_document_collection_hash_id: Optional[str] = Field(default=None, foreign_key="global_document_collection.hash_id", index=True)
    toolchain_session_hash_id: Optional[str] = Field(default=None, foreign_key="toolchain_session.hash_id", index=True)
    
    file_data: bytes = Field(sa_column=Column(LargeBinary))


class organization_document_collection(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    hash_id: str = Field(index=True, unique=True)
    name: str
    author_organization_id: int = Field(foreign_key="organization.id", index=True)
    creation_timestamp: float
    public: Optional[bool] = Field(default=False)
    description: Optional[str] = Field(default="")
    document_count: int = Field(default=0)

class user_document_collection(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    hash_id: str = Field(index=True, unique=True)
    name: str
    author_user_name: str = Field(foreign_key="user.name", index=True)
    creation_timestamp: float
    public: Optional[bool] = Field(default=False)
    description: Optional[str] = Field(default="")
    document_count: int = Field(default=0)

class global_document_collection(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    hash_id: str = Field(index=True, unique=True)
    name: str
    description: Optional[str] = Field(default="")
    document_count: int = Field(default=0)

class organization_membership(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    role: str # "owner" | "admin" | "member" | "viewer"
    organization_id: int = Field(foreign_key="organization.id", index=True)
    user_name: str = Field(foreign_key="user.name", index=True)
    invite_sender_user_name: Optional[str] = Field(default=None, foreign_key="user.name")
    
    # This is the organization private key encrypted with the user's public key
    # Although the private key is unique to the organization, it cannot be stored in plaintext
    # This way, it is exchanged securely between users, and only decrypted for file retrieval
    # Or when an invite is extended
    organization_private_key_secure: str 
    invite_still_open: Optional[bool] = Field(default=True)

class view_priviledge(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_document_collection_id: int = Field(foreign_key="user_document_collection.id", index=True)
    added_user_name: str = Field(foreign_key="user.name", index=True)

class collaboration(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    organization_document_collection_id: int = Field(foreign_key="organization_document_collection.id", index=True)
    added_organization_id: int = Field(foreign_key="organization.id", index=True)
    write_priviledge: Optional[bool] = Field(default=False)



        