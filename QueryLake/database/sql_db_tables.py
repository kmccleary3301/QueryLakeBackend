from typing import Optional, List, Literal, Tuple, Union
from sqlmodel import Field, SQLModel, ARRAY, String, Integer, Float, JSON, LargeBinary

from sqlalchemy.sql.schema import Column
from sqlalchemy.dialects.postgresql import TSVECTOR, JSONB
from sqlalchemy import event, text

from sqlalchemy.sql import func

from sqlmodel import Session, create_engine
from pgvector.sqlalchemy import Vector
from sqlalchemy import Column
from ..api.hashing import random_hash
import pgvector
from sqlalchemy import Column, DDL, event, text
from pydantic import BaseModel
import re
from psycopg2.errors import InFailedSqlTransaction
from functools import partial
from time import time
import inspect

def data_dict(db_entry : SQLModel):
    return {i:db_entry.__dict__[i] for i in db_entry.__dict__ if i != "_sa_instance_state"}

# COLLECTION_TYPES = Literal["user", "organization", "global", "toolchain", "website"]

class UsageTally(SQLModel, table=True):
    id: Optional[str] = Field(default_factory=random_hash, primary_key=True, index=True, unique=True)
    start_timestamp: float = Field(index=True)
    window: str # "hour" | "day" | "week" | "month"
    organization_id: Optional[str] = Field(foreign_key="organization.id", index=True)
    api_key_id: Optional[str] = Field(foreign_key="apikey.id", index=True)
    user_id: Optional[str] = Field(foreign_key="user.id", index=True)
    value : str # JSON of tallies.

class ApiKey(SQLModel, table=True):
    id: Optional[str] = Field(default_factory=random_hash, primary_key=True, index=True, unique=True)
    key_hash: str = Field(index=True, unique=True)
    creation_timestamp: float
    last_used: Optional[float] = Field(default=None)
    author: str = Field(foreign_key="user.name", index=True)
    
    title: Optional[str] = Field(default="API Key")
    key_preview: str # This will be of the form `sk-****abcd`
    
    # This will be the password prehash encrypted with the api key.
    user_password_prehash_encrypted: str

class DocumentChunk(SQLModel, table=True):
    id: Optional[str] = Field(default_factory=random_hash, primary_key=True, index=True, unique=True)
    creation_timestamp: Optional[float] = Field(default_factory=time)
    collection_type: Optional[str] = Field(index=True, default=None)
    document_id: Optional[str] = Field(foreign_key="document_raw.hash_id", default=None, index=True)
    document_chunk_number: Optional[int] = Field(default=None, index=True)
    document_integrity: Optional[str] = Field(default=None, index=True)
    parent_collection_hash_id: Optional[str] = Field(index=True, default=None)
    document_name: str = Field()
    website_url : Optional[str] = Field(default=None, index=True)
    embedding: List[float] = Field(sa_column=Column(Vector(1024)))
    private: bool = Field(default=False)
    
    md: dict = Field(sa_column=Column(JSONB), default={})
    text: str = Field()
    # ts_content : TSVECTOR = Field(sa_column=Column(TSVECTOR))
    ts_content: str = Field(sa_column=Column(TSVECTOR))

CHUNK_CLASS_NAME = DocumentChunk.__name__.lower()

CREATE_BM25_INDEX_SQL = """
CALL paradedb.create_bm25(
	index_name => 'search_&CHUNK_CLASS_NAME&_idx',
	table_name => '&CHUNK_CLASS_NAME&',
	key_field => 'id',
	text_fields => '{
		text: {tokenizer: {type: "en_stem"}},
        document_id: {},
        website_url: {},
        parent_collection_hash_id: {}
	}',
    json_fields => '{"md": {}}'
);
""".replace("&CHUNK_CLASS_NAME&", CHUNK_CLASS_NAME)

# You never need to pass ts_content to DocumentEmbedding, as this will automatically
# derive it from `text` using a trigger.
# It also adds an index on ts_content for faster searching.
trigger = DDL(f"""
CREATE OR REPLACE FUNCTION update_ts_content()
RETURNS TRIGGER AS $$
BEGIN
  NEW.ts_content := to_tsvector('english', NEW.text);
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS update_ts_content_trigger ON {CHUNK_CLASS_NAME};

CREATE TRIGGER update_ts_content_trigger
BEFORE INSERT OR UPDATE ON {CHUNK_CLASS_NAME}
FOR EACH ROW EXECUTE FUNCTION update_ts_content();

CREATE INDEX IF NOT EXISTS ts_content_gin ON {CHUNK_CLASS_NAME} USING gin(ts_content);
""")
event.listen(DocumentChunk.__table__, 'after_create', trigger.execute_if(dialect='postgresql'))


class DocumentEmbeddingDictionary(BaseModel):
    id: str
    collection_type: Union[str, Literal[None]]
    document_id: Union[str, Literal[None]]
    document_chunk_number: Optional[int]
    document_integrity: Union[str, Literal[None]]
    parent_collection_hash_id: Union[str, Literal[None]]
    document_name : str
    website_url : Optional[Union[str, Literal[None]]]
    private : bool
    text : str
    headline : Optional[str] = None
    cover_density_rank : Optional[float] = None
    creation_timestamp : float

def search_embeddings_lexical(database: Session,
                              collection_ids: List[str],
                              search_string: Union[str, List[str]], 
                              limit: int = 10,
                              return_statement : bool = False,
                              web_search : bool = False) -> List[Tuple[
                                    DocumentEmbeddingDictionary,
                                    float,  # rank
                                    str,    # headline
                              ]]:
    print("Collection IDs:", collection_ids)
    
    try:                     
        if isinstance(search_string, list):
            search_string = " ".join(search_string)
        
        # Safeguards for the input, preventing SQL injection
        assert isinstance(limit, int), "Limit must be an integer."
        collection_ids = [re.sub(r'[^a-zA-Z0-9]', '', e) for e in collection_ids]
        
        # Replace any characters other than letters, numbers, or spaces with a space
        search_string = re.sub(r'[^a-zA-Z0-9]', ' ', search_string)
        search_string = re.sub(r'\s+', ' ', search_string)
        # search_string = [e for e in search_string.split(" ") if e.strip() != ""]
        search_string = search_string.split(" ")
        
        # Replace spaces with the OR operator and limit to 64 terms
        search_array = [e for e in search_string[:min(64, len(search_string))] if e.strip() != ""]
        search_string = " | ".join(search_array).strip(" | ")
        
        if web_search:
            where_clause = where_clause = "WHERE ts_content @@ query AND (parent_collection_hash_id = ANY(:collection_ids) OR website_url IS NOT NULL)"
        else:
            where_clause = "WHERE ts_content @@ query AND parent_collection_hash_id = ANY(:collection_ids)"
        
        print("Arguments for query:", [search_string, collection_ids, limit, where_clause])
        
        stmt = text(f"""
            SELECT id,
                collection_type,
                document_id,
                document_chunk_number,
                document_integrity,
                parent_collection_hash_id,
                document_name,
                website_url,
                private,
                text,
                ts_rank_cd(ts_content, query) AS rank,
                ts_headline(text, query) AS headline,
                creation_timestamp
            FROM {CHUNK_CLASS_NAME},
                to_tsquery(:search_string) query
            {where_clause}
            ORDER BY rank DESC
            LIMIT :limit
        """).bindparams(search_string=search_string, collection_ids=collection_ids, limit=limit)
        
        def result_tuple_to_dict(value_in : tuple) -> Tuple[dict, float, str]:
            return (DocumentEmbeddingDictionary(**{
                "id": value_in[0],
                "collection_type": value_in[1],
                "document_id": value_in[2],
                "document_chunk_number": value_in[3],
                "document_integrity": value_in[4],
                "parent_collection_hash_id": value_in[5],
                "document_name": value_in[6],
                "website_url": value_in[7],
                "private": value_in[8],
                "text": value_in[9],
                "cover_density_rank": value_in[10],
                "headline": value_in[11],
                "creation_timestamp": value_in[12]
            }), value_in[10], value_in[11])
        
        if return_statement:
            return stmt
        
        results = database.exec(stmt)
        return list(map(result_tuple_to_dict, results))
    except InFailedSqlTransaction:
        database.flush()
        raise Exception("Database transaction failed. Please try again.")

class ToolchainSessionFileOutput(SQLModel, table=True):
    id: Optional[str] = Field(default_factory=random_hash, primary_key=True, index=True, unique=True)
    creation_timestamp: float
    file_name: Optional[str] = Field(default=None)
    file_data: bytes = Field(sa_column=Column(LargeBinary))

class toolchain_session(SQLModel, table=True):
    id: Optional[str] = Field(default_factory=partial(random_hash, 26, 62), primary_key=True, index=True, unique=True)
    title: Optional[str] = Field(default=None)
    hidden: Optional[bool] = Field(default=False)
    creation_timestamp: float
    toolchain_id: str = Field(foreign_key="toolchain.toolchain_id", index=True)
    author: str = Field(foreign_key="user.name", index=True)
    state: Optional[str] = Field(default=None)
    file_state : Optional[str] = Field(default=None)
    queue_inputs: Optional[str] = Field(default="")
    firing_queue: Optional[str] = Field(default="")
    first_event_fired: Optional[bool] = Field(default=False, index=True)

class toolchain(SQLModel, table=True):
    id: Optional[str] = Field(default_factory=random_hash, primary_key=True, index=True, unique=True)
    toolchain_id: str = Field(index=True, unique=True)
    title: str
    category: str
    content: str #JSON loads this portion.

class document_access_token(SQLModel, table=True):
    id: Optional[str] = Field(default_factory=random_hash, primary_key=True, index=True, unique=True)
    hash_id: str = Field(index=True, unique=True)
    expiration_timestamp: float

class model(SQLModel, table=True):
    id: Optional[str] = Field(default_factory=random_hash, primary_key=True, index=True, unique=True)
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
    """
    The user account table.
    This relies on salted password hashing.
    The hash process is as follows:
    
    1. password_salt = random_hash()
    2. password_prehash = hash(password)
    3. password_hash = hash(password_prehash + password_salt)'
    
    The password prehash is the unsalted password hash. It is never stored on disk,
    but it is critical for decrypting user data since it is used to encrypt the user's private key.
    """
    
    id: Optional[str] = Field(default_factory=random_hash, primary_key=True, index=True, unique=True)
    name: str = Field(index=True, unique=True)
    email: Optional[str] = Field(default="", index=True)
    password_hash: str
    password_salt: str
    creation_timestamp: float
    is_admin: Optional[bool] = Field(default=False)
    public_key: str                                                     # Randomly generated public key
    private_key_encryption_salt: str
    private_key_secured: str                                            # Encrypted with salt(password_prehash, encryption_salt)
    external_providers_encrypted: Optional[str] = Field(default=None)   # Encrypted with salt(password_prehash, encryption_salt), is a JSON string.

class organization(SQLModel, table=True):
    id: Optional[str] = Field(default_factory=random_hash, primary_key=True, index=True, unique=True)
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
    organization_id: Optional[str] = Field(default=None, foreign_key="organization.id", index=True)

class document_raw(SQLModel, table=True):
    id: Optional[str] = Field(default_factory=random_hash, primary_key=True, index=True, unique=True)
    hash_id: str = Field(index=True, unique=True)
    file_name: str
    creation_timestamp: float
    integrity_sha256: str = Field(index=True)
    size_bytes: int
    encryption_key_secure: Optional[str] = Field(default=None)
    organization_document_collection_hash_id: Optional[str] = Field(default=None, foreign_key="organization_document_collection.hash_id", index=True)
    user_document_collection_hash_id: Optional[str] = Field(default=None, foreign_key="user_document_collection.hash_id", index=True)
    global_document_collection_hash_id: Optional[str] = Field(default=None, foreign_key="global_document_collection.hash_id", index=True)
    toolchain_session_id: Optional[str] = Field(default=None, foreign_key="toolchain_session.id", index=True)
    
    website_url: Optional[str] = Field(default=None)
    
    file_data: bytes = Field(sa_column=Column(LargeBinary))
    finished_processing: Optional[bool] = Field(default=False)
    


class organization_document_collection(SQLModel, table=True):
    id: Optional[str] = Field(default_factory=random_hash, primary_key=True, index=True, unique=True)
    hash_id: str = Field(index=True, unique=True)
    name: str
    author_organization_id: str = Field(foreign_key="organization.id", index=True)
    creation_timestamp: float
    public: Optional[bool] = Field(default=False)
    description: Optional[str] = Field(default="")
    document_count: int = Field(default=0)

class user_document_collection(SQLModel, table=True):
    id: Optional[str] = Field(default_factory=random_hash, primary_key=True, index=True, unique=True)
    hash_id: str = Field(index=True, unique=True)
    name: str
    author_user_name: str = Field(foreign_key="user.name", index=True)
    creation_timestamp: float
    public: Optional[bool] = Field(default=False)
    description: Optional[str] = Field(default="")
    document_count: int = Field(default=0)

class global_document_collection(SQLModel, table=True):
    id: Optional[str] = Field(default_factory=random_hash, primary_key=True, index=True, unique=True)
    hash_id: str = Field(index=True, unique=True)
    name: str
    description: Optional[str] = Field(default="")
    document_count: int = Field(default=0)

class organization_membership(SQLModel, table=True):
    id: Optional[str] = Field(default_factory=random_hash, primary_key=True, index=True, unique=True)
    role: str # "owner" | "admin" | "member" | "viewer"
    organization_id: str = Field(foreign_key="organization.id", index=True)
    user_name: str = Field(foreign_key="user.name", index=True)
    invite_sender_user_name: Optional[str] = Field(default=None, foreign_key="user.name")
    
    # This is the organization private key encrypted with the user's public key
    # Although the private key is unique to the organization, it cannot be stored in plaintext
    # This way, it is exchanged securely between users, and only decrypted for file retrieval
    # Or when an invite is extended
    organization_private_key_secure: str 
    invite_still_open: Optional[bool] = Field(default=True)

class view_priviledge(SQLModel, table=True):
    id: Optional[str] = Field(default_factory=random_hash, primary_key=True, index=True, unique=True)
    user_document_collection_id: str = Field(foreign_key="user_document_collection.id", index=True)
    added_user_name: str = Field(foreign_key="user.name", index=True)

class collaboration(SQLModel, table=True):
    id: Optional[str] = Field(default_factory=random_hash, primary_key=True, index=True, unique=True)
    organization_document_collection_id: str = Field(foreign_key="organization_document_collection.id", index=True)
    added_organization_id: str = Field(foreign_key="organization.id", index=True)
    write_priviledge: Optional[bool] = Field(default=False)
