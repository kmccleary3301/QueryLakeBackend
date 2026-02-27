from typing import Optional, List, Literal, Tuple, Union
from sqlmodel import Field, SQLModel, ARRAY, String, Integer, Float, JSON, LargeBinary

from sqlalchemy.sql.schema import Column
from sqlalchemy.dialects.postgresql import TSVECTOR, JSONB
from sqlalchemy import UniqueConstraint, event, text

from sqlalchemy.sql import func

from sqlmodel import Session, create_engine, UUID
from pgvector.sqlalchemy import Vector, HALFVEC, SPARSEVEC
from sqlalchemy import Column
from ..api.hashing import random_hash
from sqlalchemy import Column, DDL, event, text
from pydantic import BaseModel
import re
from psycopg2.errors import InFailedSqlTransaction
from functools import partial
from time import time
import inspect
import uuid as uuid_pkg

def data_dict(db_entry : SQLModel):
    return {i:db_entry.__dict__[i] for i in db_entry.__dict__ if i != "_sa_instance_state"}

def random_hash_32():
    return random_hash(32, 62)


generator_1 = random_hash_32
id_type_1 = str


id_factories = {
    "document_chunk": generator_1,
    "collections": generator_1,
    "document_raw": generator_1,
    "document_version": generator_1,
    "document_artifact": generator_1,
    "document_segment": generator_1,
    "segmentation_run": generator_1,
    "embedding_record": generator_1,
    "segment_edge": generator_1,
    # Files v2
    "file": generator_1,
    "file_version": generator_1,
    "file_page": generator_1,
    "file_chunk": generator_1,
    # Retrieval observability
    "retrieval_run": generator_1,
    "retrieval_pipeline_config": generator_1,
    "retrieval_experiment": generator_1,
    "retrieval_pipeline_binding": generator_1,
    # Generic pipeline queue
    "pipeline_work_item": generator_1,
}

id_types = {
    "document_chunk": id_type_1,
    "collections": id_type_1,
    "document_raw": id_type_1,
    "document_version": id_type_1,
    "document_artifact": id_type_1,
    "document_segment": id_type_1,
    "segmentation_run": id_type_1,
    "embedding_record": id_type_1,
    "segment_edge": id_type_1,
    # Files v2
    "file": id_type_1,
    "file_version": id_type_1,
    "file_page": id_type_1,
    "file_chunk": id_type_1,
    # Retrieval observability
    "retrieval_run": id_type_1,
    "retrieval_pipeline_config": id_type_1,
    "retrieval_experiment": id_type_1,
    "retrieval_pipeline_binding": id_type_1,
    # Generic pipeline queue
    "pipeline_work_item": id_type_1,
}

# COLLECTION_TYPES = Literal["user", "organization", "global", "toolchain", "website"]

class UsageTally(SQLModel, table=True):
    id: Optional[str] = Field(default_factory=random_hash_32, primary_key=True, index=True, unique=True)
    start_timestamp: int = Field(index=True)
    window: str = Field(index=True) # "hour" | "day" | "month"
    organization_id: Optional[str] = Field(foreign_key="organization.id", index=True, default=None)
    api_key_id: Optional[str] = Field(foreign_key="apikey.id", index=True, default=None)
    user_id: Optional[str] = Field(foreign_key="user.id", index=True)
    value : dict = Field(sa_column=Column(JSONB), default={}) # JSON of tallies.

class ApiKey(SQLModel, table=True):
    id: Optional[str] = Field(default_factory=random_hash_32, primary_key=True, index=True, unique=True)
    key_hash: str = Field(index=True, unique=True)
    creation_timestamp: float
    last_used: Optional[float] = Field(default=None)
    author: str = Field(foreign_key="user.name", index=True)
    
    title: Optional[str] = Field(default="API Key")
    key_preview: str # This will be of the form `sk-****abcd`
    
    # This will be the password prehash encrypted with the api key.
    user_password_prehash_encrypted: str


class ToolchainSessionFileOutput(SQLModel, table=True):
    id: Optional[str] = Field(default_factory=random_hash_32, primary_key=True, index=True, unique=True)
    creation_timestamp: float
    file_name: Optional[str] = Field(default=None)
    file_data: bytes = Field(sa_column=Column(LargeBinary))



class toolchain(SQLModel, table=True):
    id: Optional[str] = Field(default_factory=random_hash_32, primary_key=True, index=True, unique=True)
    toolchain_id: str = Field(index=True, unique=True)
    title: str
    category: str
    content: str #JSON loads this portion.
    
class toolchain_session(SQLModel, table=True):
    id: Optional[str] = Field(default_factory=partial(random_hash, 26, 62), primary_key=True, index=True, unique=True)
    title: Optional[str] = Field(default=None)
    hidden: Optional[bool] = Field(default=False)
    creation_timestamp: float
    toolchain_id: str = Field(foreign_key=f"{toolchain.__tablename__}.toolchain_id", index=True)
    author: str = Field(foreign_key="user.name", index=True)
    state: Optional[str] = Field(default=None)
    misc_data: Optional[str] = Field(default=None)
    file_state : Optional[str] = Field(default=None)
    queue_inputs: Optional[str] = Field(default="")
    firing_queue: Optional[str] = Field(default="")
    first_event_fired: Optional[bool] = Field(default=False, index=True)


class ToolchainSessionEvent(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: str = Field(foreign_key=f"{toolchain_session.__tablename__}.id", index=True)
    rev: int = Field(index=True)
    ts: float = Field(default_factory=time)
    kind: str
    payload: dict = Field(sa_column=Column(JSONB))
    actor: Optional[str] = None
    correlation_id: Optional[str] = None


class ToolchainSessionSnapshot(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: str = Field(foreign_key=f"{toolchain_session.__tablename__}.id", index=True)
    rev: int = Field(index=True)
    ts: float = Field(default_factory=time)
    state: dict = Field(sa_column=Column(JSONB))
    files: Optional[dict] = Field(default=None, sa_column=Column(JSONB))


class ToolchainJob(SQLModel, table=True):
    job_id: str = Field(default_factory=random_hash_32, primary_key=True, index=True, unique=True)
    session_id: str = Field(foreign_key=f"{toolchain_session.__tablename__}.id", index=True)
    node_id: str
    status: str
    request_id: Optional[str] = None
    progress: Optional[dict] = Field(default=None, sa_column=Column(JSONB))
    result_meta: Optional[dict] = Field(default=None, sa_column=Column(JSONB))
    created_at: float = Field(default_factory=time)
    updated_at: float = Field(default_factory=time, index=True)


class ToolchainDeadLetter(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: str = Field(foreign_key=f"{toolchain_session.__tablename__}.id", index=True)
    rev: Optional[int] = Field(default=None)
    ts: float = Field(default_factory=time)
    event: dict = Field(sa_column=Column(JSONB))
    error: Optional[str] = None

# -----------------------------
# Files subsystem (Phase 1–2)
# -----------------------------

class file(SQLModel, table=True):
    id: Optional[str] = Field(default_factory=id_factories["file"], primary_key=True, index=True, unique=True)
    logical_name: str
    created_at: float = Field(default_factory=time)
    created_by: Optional[str] = Field(default=None, index=True)
    # Foreign key to document_collection (optional); avoid forward ref for simplicity
    collection_id: Optional[str] = Field(default=None, index=True)


class file_version(SQLModel, table=True):
    id: Optional[str] = Field(default_factory=id_factories["file_version"], primary_key=True, index=True, unique=True)
    file_id: str = Field(foreign_key=f"{file.__tablename__}.id", index=True)
    version_no: int
    bytes_cas: str = Field(index=True)
    size_bytes: int
    mime_type: Optional[str] = Field(default=None)
    processing_fingerprint: Optional[dict] = Field(default=None, sa_column=Column(JSONB))
    created_at: float = Field(default_factory=time)


class file_page(SQLModel, table=True):
    id: Optional[str] = Field(default_factory=id_factories["file_page"], primary_key=True, index=True, unique=True)
    file_version_id: str = Field(foreign_key=f"{file_version.__tablename__}.id", index=True)
    page_num: int
    width_px: Optional[int] = Field(default=None)
    height_px: Optional[int] = Field(default=None)
    ocr_json_cas: Optional[str] = Field(default=None)
    image_cas: Optional[str] = Field(default=None)
    text_span_offsets: Optional[str] = Field(default=None)


class file_chunk(SQLModel, table=True):
    id: Optional[str] = Field(default_factory=id_factories["file_chunk"], primary_key=True, index=True, unique=True)
    file_version_id: str = Field(foreign_key=f"{file_version.__tablename__}.id", index=True)
    page_start: Optional[int] = Field(default=None)
    page_end: Optional[int] = Field(default=None)
    byte_start: Optional[int] = Field(default=None)
    byte_end: Optional[int] = Field(default=None)
    text: str
    md: dict = Field(sa_column=Column(JSONB), default={})
    anchors: list = Field(sa_column=Column(JSONB), default=[])
    embedding: Optional[List[float]] = Field(sa_column=Column(HALFVEC(1024)), default=None)
    created_at: float = Field(default_factory=time)

# Fields supported by ParadeDB BM25 for file_chunk
FILE_CHUNK_INDEXED_COLUMNS = [
    "id",
    "text",
    "md",
    "created_at",
]


class file_event(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    file_id: str = Field(foreign_key=f"{file.__tablename__}.id", index=True)
    version_id: str = Field(foreign_key=f"{file_version.__tablename__}.id", index=True)
    rev: int = Field(index=True)
    ts: float = Field(default_factory=time)
    kind: str
    payload: dict = Field(sa_column=Column(JSONB))


class file_job(SQLModel, table=True):
    job_id: str = Field(primary_key=True, index=True, unique=True)
    file_id: str = Field(foreign_key=f"{file.__tablename__}.id", index=True)
    version_id: str = Field(foreign_key=f"{file_version.__tablename__}.id", index=True)
    status: str
    progress: Optional[dict] = Field(default=None, sa_column=Column(JSONB))
    result_meta: Optional[dict] = Field(default=None, sa_column=Column(JSONB))
    created_at: float = Field(default_factory=time)
    updated_at: float = Field(default_factory=time, index=True)


class file_dead_letter(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    file_id: str = Field(foreign_key=f"{file.__tablename__}.id", index=True)
    version_id: Optional[str] = Field(default=None, foreign_key=f"{file_version.__tablename__}.id", index=True)
    ts: float = Field(default_factory=time)
    event: dict = Field(sa_column=Column(JSONB))
    error: Optional[str] = None


# ---------------------------------
# Generic pipeline work queue
# ---------------------------------

class PipelineWorkItem(SQLModel, table=True):
    __tablename__ = "pipeline_work_item"

    id: Optional[str] = Field(default_factory=id_factories["pipeline_work_item"], primary_key=True, index=True, unique=True)
    pipeline_key: str = Field(index=True)  # e.g. "documentchunk_embedding_backfill"
    stage: str = Field(index=True)  # e.g. "embed_missing"
    entity_table: str = Field(index=True)  # e.g. "documentchunk"
    entity_id: str = Field(index=True)
    collection_id: Optional[str] = Field(default=None, index=True)
    priority: int = Field(default=100, index=True)

    status: str = Field(default="pending", index=True)  # pending | leased | done | failed
    attempts: int = Field(default=0, index=True)
    lease_owner: Optional[str] = Field(default=None, index=True)
    lease_expires_at: Optional[float] = Field(default=None, index=True)
    available_at: float = Field(default_factory=time, index=True)

    payload: dict = Field(sa_column=Column(JSONB), default={})
    last_error: Optional[str] = Field(default=None)
    created_at: float = Field(default_factory=time, index=True)
    updated_at: float = Field(default_factory=time, index=True)

    __table_args__ = (
        UniqueConstraint(
            "pipeline_key",
            "stage",
            "entity_table",
            "entity_id",
            name="uq_pipeline_work_item_scope_entity",
        ),
    )


# ---------------------------------
# Retrieval observability (Phase 0)
# ---------------------------------

class retrieval_run(SQLModel, table=True):
    run_id: Optional[str] = Field(default_factory=id_factories["retrieval_run"], primary_key=True, index=True, unique=True)
    created_at: float = Field(default_factory=time, index=True)
    completed_at: Optional[float] = Field(default=None, index=True)
    status: str = Field(default="ok", index=True)  # "ok" | "error"

    route: str = Field(index=True)  # e.g. search_hybrid, search_bm25
    actor_user: Optional[str] = Field(default=None, index=True)
    tenant_scope: Optional[str] = Field(default=None, index=True)

    pipeline_id: Optional[str] = Field(default=None, index=True)
    pipeline_version: Optional[str] = Field(default=None, index=True)

    query_text: str = Field(default="")
    query_hash: Optional[str] = Field(default=None, index=True)

    filters: dict = Field(sa_column=Column(JSONB), default={})
    budgets: dict = Field(sa_column=Column(JSONB), default={})
    timings: dict = Field(sa_column=Column(JSONB), default={})
    counters: dict = Field(sa_column=Column(JSONB), default={})
    costs: dict = Field(sa_column=Column(JSONB), default={})
    index_snapshots_used: dict = Field(sa_column=Column(JSONB), default={})
    result_ids: list = Field(sa_column=Column(JSONB), default=[])
    md: dict = Field(sa_column=Column(JSONB), default={})

    error: Optional[str] = Field(default=None)


class retrieval_candidate(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    run_id: str = Field(foreign_key=f"{retrieval_run.__tablename__}.run_id", index=True)
    content_id: str = Field(index=True)
    final_selected: bool = Field(default=True, index=True)

    stage_scores: dict = Field(sa_column=Column(JSONB), default={})
    stage_ranks: dict = Field(sa_column=Column(JSONB), default={})
    provenance: list = Field(sa_column=Column(JSONB), default=[])
    md: dict = Field(sa_column=Column(JSONB), default={})


class retrieval_pipeline_config(SQLModel, table=True):
    id: Optional[str] = Field(default_factory=id_factories["retrieval_pipeline_config"], primary_key=True, index=True, unique=True)
    pipeline_id: str = Field(index=True)
    version: str = Field(index=True)
    immutable_hash: str = Field(index=True)
    spec_json: dict = Field(sa_column=Column(JSONB), default={})
    created_at: float = Field(default_factory=time, index=True)
    created_by: Optional[str] = Field(default=None, index=True)
    status: str = Field(default="active", index=True)
    md: dict = Field(sa_column=Column(JSONB), default={})

    __table_args__ = (
        UniqueConstraint("pipeline_id", "version", name="uq_retrieval_pipeline_config_pipeline_version"),
    )


class retrieval_experiment(SQLModel, table=True):
    experiment_id: Optional[str] = Field(default_factory=id_factories["retrieval_experiment"], primary_key=True, index=True, unique=True)
    title: str = Field(index=True)
    owner: Optional[str] = Field(default=None, index=True)
    status: str = Field(default="draft", index=True)  # draft | running | paused | completed | archived

    baseline_pipeline_id: str = Field(index=True)
    baseline_pipeline_version: str = Field(index=True)
    candidate_pipeline_id: str = Field(index=True)
    candidate_pipeline_version: str = Field(index=True)

    created_at: float = Field(default_factory=time, index=True)
    updated_at: float = Field(default_factory=time, index=True)
    md: dict = Field(sa_column=Column(JSONB), default={})


class retrieval_experiment_run(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    experiment_id: str = Field(foreign_key=f"{retrieval_experiment.__tablename__}.experiment_id", index=True)
    query_hash: Optional[str] = Field(default=None, index=True)
    query_text: str = Field(default="")

    baseline_run_id: Optional[str] = Field(default=None, foreign_key=f"{retrieval_run.__tablename__}.run_id", index=True)
    candidate_run_id: Optional[str] = Field(default=None, foreign_key=f"{retrieval_run.__tablename__}.run_id", index=True)

    baseline_metrics: dict = Field(sa_column=Column(JSONB), default={})
    candidate_metrics: dict = Field(sa_column=Column(JSONB), default={})
    delta_metrics: dict = Field(sa_column=Column(JSONB), default={})
    publish_mode: str = Field(default="baseline", index=True)
    published_pipeline_id: Optional[str] = Field(default=None, index=True)
    published_pipeline_version: Optional[str] = Field(default=None, index=True)
    created_at: float = Field(default_factory=time, index=True)


class retrieval_pipeline_binding(SQLModel, table=True):
    id: Optional[str] = Field(default_factory=id_factories["retrieval_pipeline_binding"], primary_key=True, index=True, unique=True)
    route: str = Field(index=True)
    tenant_scope: Optional[str] = Field(default=None, index=True)
    active_pipeline_id: str = Field(index=True)
    active_pipeline_version: str = Field(index=True)
    previous_pipeline_id: Optional[str] = Field(default=None, index=True)
    previous_pipeline_version: Optional[str] = Field(default=None, index=True)
    updated_by: Optional[str] = Field(default=None, index=True)
    updated_at: float = Field(default_factory=time, index=True)
    md: dict = Field(sa_column=Column(JSONB), default={})

    __table_args__ = (
        UniqueConstraint("route", "tenant_scope", name="uq_retrieval_pipeline_binding_route_tenant"),
    )

class document_access_token(SQLModel, table=True):
    id: Optional[str] = Field(default_factory=random_hash_32, primary_key=True, index=True, unique=True)
    hash_id: str = Field(index=True, unique=True)
    expiration_timestamp: float

class model(SQLModel, table=True):
    id: Optional[str] = Field(default_factory=random_hash_32, primary_key=True, index=True, unique=True)
    name: str = Field(index=True)
    path_on_server: str
    default_settings: str #This will be a stringified JSON that will be parsed out kwargs on load.

    # Let's store wrappers as something like prepend_string+"\<system_instruction}\>"+append_string,
    # Where the prepend and append strings are html-encoded.
    # Wrappers will obviously be a single string, but this way we can split into 3 parts.

    system_instruction_wrapper: str 
    context_wrapper: str
    user_question_wrapper: str
    bot_response_wrapper: str

    default_system_instruction: str


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
    
    id: Optional[str] = Field(default_factory=random_hash_32, primary_key=True, index=True, unique=True)
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
    id: Optional[str] = Field(default_factory=random_hash_32, primary_key=True, index=True, unique=True)
    hash_id: str = Field(index=True, unique=True)
    name: str
    creation_timestamp: float
    public_key: str
    

DOCUMENT_INDEXED_COLUMNS = [
    "id", 
    "file_name",
    "creation_timestamp",
    "integrity_sha256", 
    "size_bytes",
    "website_url",
    "md",
    "finished_processing",
    "document_collection_id",
]


class organization_membership(SQLModel, table=True):
    id: Optional[str] = Field(default_factory=random_hash_32, primary_key=True, index=True, unique=True)
    role: str # "owner" | "admin" | "member" | "viewer"
    organization_id: str = Field(foreign_key=f"{organization.__tablename__}.id", index=True)
    user_name: str = Field(foreign_key=f"{user.__tablename__}.name", index=True)
    invite_sender_user_name: Optional[str] = Field(default=None, foreign_key=f"{user.__tablename__}.name")
    
    # This is the organization private key encrypted with the user's public key
    # Although the private key is unique to the organization, it cannot be stored in plaintext
    # This way, it is exchanged securely between users, and only decrypted for file retrieval
    # Or when an invite is extended
    organization_private_key_secure: str 
    invite_still_open: Optional[bool] = Field(default=True)


# TODO: Maybe update this with new schemas and use it one day.
# class view_priviledge(SQLModel, table=True):
#     id: Optional[str] = Field(default_factory=random_hash_32, primary_key=True, index=True, unique=True)
#     user_document_collection_id: str = Field(foreign_key="user_document_collection.id", index=True)
#     added_user_name: str = Field(foreign_key="user.name", index=True)

class document_collection(SQLModel, table=True):
    id: Optional[str] = Field(default_factory=id_factories["collections"], primary_key=True, index=True, unique=True)
    name: str
    creation_timestamp: float = Field(default_factory=time)
    public: Optional[bool] = Field(default=False)
    description: Optional[str] = Field(default="")
    document_count: int = Field(default=0)
    
    collection_type: str = Field(index=True) # "organization" | "user" | "global" | "toolchain_session"
    
    
    author_user_name: str | None = Field(foreign_key=f"{user.__tablename__}.name", index=True, default=None, nullable=True)
    author_organization: str | None = Field(foreign_key=f"{organization.__tablename__}.id", index=True, default=None, nullable=True)
    toolchain_session_id: str | None = Field(foreign_key=f"{toolchain_session.__tablename__}.id", index=True, default=None, nullable=True)
    
    
    # If public or global, is plaintext unencrypted.
    # user or toolchain collection (if private): Encrypted with user's public key
    # organization collection (if private): Encrypted with organization's public key
    encryption_key_secure: Optional[str] = Field(default=None)
    
    
    # Document unlock key is encrypted with `encryption_key_secure`
    # It should never be delivered to the client, only used for internal operations
    # The decrypted value will remain the same always, but encryption may change
    # With shifting collections.
    document_unlock_key: Optional[str] = Field(default=None)

class document_zip_blob(SQLModel, table=True):
    id: Optional[str] = Field(default_factory=random_hash_32, primary_key=True, index=True, unique=True)
    file_count: int
    size_bytes: int
    
    encryption_key_secure: Optional[str] = Field(default=None)
    document_collection_id: Optional[str] = Field(foreign_key=f"{document_collection.__tablename__}.id", index=True)
    
    file_data: bytes = Field(sa_column=Column(LargeBinary))


class document_raw(SQLModel, table=True):
    id: Optional[str] = Field(default_factory=id_factories["document_raw"], primary_key=True, index=True, unique=True)
    # hash_id: str = Field(index=True, unique=True)
    file_name: str
    creation_timestamp: float
    integrity_sha256: str = Field(index=True)
    size_bytes: int = Field(index=True)
    encryption_key_secure: Optional[str] = Field(default=None)
    document_collection_id: Optional[str] = Field(foreign_key=f"{document_collection.__tablename__}.id", index=True)
    
    website_url: Optional[str] = Field(default=None)
    
    file_data: bytes = Field(sa_column=Column(LargeBinary))
    
    blob_id: Optional[str] = Field(default=None, foreign_key=f"{document_zip_blob.__tablename__}.id", index=True)
    blob_dir: Optional[str] = Field(default=None)
    
    # Finished processing has the following states:
    # 0 - Processing not attempted
    # 1 - Processing initiated
    # 2 - Processing failed
    # 3 - Processing done (no embeddings)
    # 4 - Processing done (embeddings)
    finished_processing: int = Field(default=0, index=True) 
    md: Optional[dict] = Field(sa_column=Column(JSONB), default={})


class document_version(SQLModel, table=True):
    id: Optional[str] = Field(default_factory=id_factories["document_version"], primary_key=True, index=True, unique=True)
    document_id: str = Field(foreign_key=f"{document_raw.__tablename__}.id", index=True)
    version_no: int = Field(index=True)
    content_hash: Optional[str] = Field(default=None, index=True)
    status: str = Field(default="ready", index=True)
    created_at: float = Field(default_factory=time, index=True)
    created_by: Optional[str] = Field(default=None, index=True)
    md: dict = Field(sa_column=Column(JSONB), default={})

    __table_args__ = (
        UniqueConstraint("document_id", "version_no", name="uq_document_version_doc_version_no"),
    )


class document_artifact(SQLModel, table=True):
    id: Optional[str] = Field(default_factory=id_factories["document_artifact"], primary_key=True, index=True, unique=True)
    document_version_id: str = Field(foreign_key=f"{document_version.__tablename__}.id", index=True)
    artifact_type: str = Field(index=True)  # raw_text | markdown | ocr_layout | normalized_text
    storage_ref: Optional[str] = Field(default=None, index=True)
    text: Optional[str] = Field(default=None)
    md: dict = Field(sa_column=Column(JSONB), default={})
    created_at: float = Field(default_factory=time, index=True)


class document_segment(SQLModel, table=True):
    id: Optional[str] = Field(default_factory=id_factories["document_segment"], primary_key=True, index=True, unique=True)
    document_version_id: str = Field(foreign_key=f"{document_version.__tablename__}.id", index=True)
    artifact_id: Optional[str] = Field(default=None, foreign_key=f"{document_artifact.__tablename__}.id", index=True)
    segment_type: str = Field(default="chunk", index=True)  # chunk | sentence | section | table | figure
    segment_index: int = Field(default=0, index=True)
    parent_segment_id: Optional[str] = Field(default=None, foreign_key=f"{'document_segment'}.id", index=True)
    text: str = Field(default="")
    md: dict = Field(sa_column=Column(JSONB), default={})
    embedding: Optional[List[float]] = Field(sa_column=Column(HALFVEC(1024)), default=None)
    embedding_sparse: Optional[dict] = Field(sa_column=Column(SPARSEVEC()), default=None)
    created_at: float = Field(default_factory=time, index=True)

    __table_args__ = (
        UniqueConstraint("document_version_id", "segment_type", "segment_index", name="uq_document_segment_version_type_index"),
    )


class segmentation_run(SQLModel, table=True):
    run_id: Optional[str] = Field(default_factory=id_factories["segmentation_run"], primary_key=True, index=True, unique=True)
    document_version_id: str = Field(foreign_key=f"{document_version.__tablename__}.id", index=True)
    segment_type: str = Field(default="chunk", index=True)
    status: str = Field(default="ok", index=True)
    config: dict = Field(sa_column=Column(JSONB), default={})
    stats: dict = Field(sa_column=Column(JSONB), default={})
    created_at: float = Field(default_factory=time, index=True)
    completed_at: Optional[float] = Field(default=None, index=True)


class embedding_record(SQLModel, table=True):
    id: Optional[str] = Field(default_factory=id_factories["embedding_record"], primary_key=True, index=True, unique=True)
    segment_id: str = Field(foreign_key=f"{document_segment.__tablename__}.id", index=True)
    model_id: str = Field(index=True)
    input_hash: str = Field(index=True)
    embedding: Optional[List[float]] = Field(sa_column=Column(HALFVEC(1024)), default=None)
    embedding_sparse: Optional[dict] = Field(sa_column=Column(SPARSEVEC()), default=None)
    created_at: float = Field(default_factory=time, index=True)
    md: dict = Field(sa_column=Column(JSONB), default={})

    __table_args__ = (
        UniqueConstraint("segment_id", "model_id", "input_hash", name="uq_embedding_record_segment_model_input_hash"),
    )


class segment_edge(SQLModel, table=True):
    id: Optional[str] = Field(default_factory=id_factories["segment_edge"], primary_key=True, index=True, unique=True)
    from_segment_id: str = Field(foreign_key=f"{document_segment.__tablename__}.id", index=True)
    to_segment_id: str = Field(foreign_key=f"{document_segment.__tablename__}.id", index=True)
    edge_type: str = Field(default="adjacent", index=True)  # adjacent | parent_child | citation | reference
    weight: float = Field(default=1.0, index=True)
    md: dict = Field(sa_column=Column(JSONB), default={})
    created_at: float = Field(default_factory=time, index=True)

    __table_args__ = (
        UniqueConstraint("from_segment_id", "to_segment_id", "edge_type", name="uq_segment_edge_from_to_type"),
    )

SEGMENT_INDEXED_COLUMNS = [
    "id",
    "segment_type",
    "segment_index",
    "text",
    "md",
    "created_at",
]


class DocumentChunk(SQLModel, table=True):
    id: Optional[str] = Field(default_factory=id_factories["document_chunk"], primary_key=True, index=True, unique=True)
    creation_timestamp: Optional[float] = Field(default_factory=time)
    collection_type: Optional[str] = Field(index=True, default=None)
    document_id: Optional[str] = Field(foreign_key=f"{document_raw.__tablename__}.id", default=None, index=True)
    document_chunk_number: Optional[int] = Field(default=None, index=True)
    document_integrity: Optional[str] = Field(default=None, index=True)
    collection_id: Optional[str] = Field(index=True, default=None)
    document_name: str = Field()
    website_url : Optional[str] = Field(default=None, index=True)
    # embedding: Optional[List[float]] = Field(sa_column=Column(Vector(1024)), default=None) # Full 32-bit Precision
    embedding: Optional[List[float]] = Field(sa_column=Column(HALFVEC(1024)), default=None) # Half Precision (16 bit)
    embedding_sparse: Optional[dict] = Field(sa_column=Column(SPARSEVEC()), default=None)
    private: bool = Field(default=False)
    
    document_md: dict = Field(sa_column=Column(JSONB), default={})
    md: dict = Field(sa_column=Column(JSONB), default={})
    text: str = Field()
    # ts_content : TSVECTOR = Field(sa_column=Column(TSVECTOR))
    ts_content: str = Field(sa_column=Column(TSVECTOR))

CHUNK_INDEXED_COLUMNS = ["id", "text", "document_id", "document_name", "website_url", "collection_id", "md", "document_md", "creation_timestamp", "document_chunk_number"]

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

DROP TRIGGER IF EXISTS update_ts_content_trigger ON {DocumentChunk.__tablename__};

CREATE TRIGGER update_ts_content_trigger
BEFORE INSERT OR UPDATE ON {DocumentChunk.__tablename__}
FOR EACH ROW EXECUTE FUNCTION update_ts_content();

CREATE INDEX IF NOT EXISTS ts_content_gin ON {DocumentChunk.__tablename__} USING gin(ts_content);
""")
event.listen(DocumentChunk.__table__, 'after_create', trigger.execute_if(dialect='postgresql'))



# # Add metadata trigger to document chunks that duplicates from parent document.
# trigger_md = DDL(f"""
# CREATE OR REPLACE FUNCTION set_document_chunk_md()
# RETURNS TRIGGER AS $$
# BEGIN
#     NEW.document_md := (SELECT md FROM {document_raw.__tablename__} WHERE id = NEW.document_id);
#     RETURN NEW;
# END;
# $$ LANGUAGE plpgsql;

# DROP TRIGGER IF EXISTS set_document_chunk_md_trigger ON {DocumentChunk.__tablename__};

# CREATE TRIGGER set_document_chunk_md_trigger
# BEFORE INSERT OR UPDATE ON {DocumentChunk.__tablename__}
# FOR EACH ROW EXECUTE FUNCTION set_document_chunk_md();
# """)

# # Attach the trigger to the DocumentChunk table
# event.listen(DocumentChunk.__table__, 'after_create', trigger_md.execute_if(dialect='postgresql'))


class DocumentEmbeddingDictionary(BaseModel):
    id: str
    collection_type: Union[str, Literal[None]]
    document_id: Union[str, Literal[None]]
    document_chunk_number: Optional[int]
    document_integrity: Union[str, Literal[None]]
    collection_id: Union[str, Literal[None]]
    document_name : str
    website_url : Optional[Union[str, Literal[None]]]
    private : bool
    text : str
    headline : Optional[str] = None
    cover_density_rank : Optional[float] = None
    creation_timestamp : float
