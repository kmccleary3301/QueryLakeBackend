import time
import random
from typing import Optional
from sqlmodel import Field, SQLModel, Session, create_engine
from sqlalchemy import Column, DDL, event, text, Index
from sqlalchemy.dialects.postgresql import TSVECTOR
from typing import List, Tuple
from pgvector.sqlalchemy import Vector

from .sql_db_tables import *


CHECK_INDEX_EXISTS_SQL = """
SELECT EXISTS (
    SELECT 1
    FROM   pg_class c
    JOIN   pg_namespace n ON n.oid = c.relnamespace
    WHERE  c.relname = '&CHUNK_CLASS_NAME&_vector_cos_idx'
    AND    n.nspname = 'public'  -- or your schema name here
);
""".replace("&CHUNK_CLASS_NAME&", DocumentChunk_backup.__tablename__)

CREATE_VECTOR_INDEX_SQL = """
DO $$
BEGIN
	EXECUTE 'CREATE INDEX &CHUNK_CLASS_NAME&_vector_cos_idx ON &CHUNK_CLASS_NAME&
				USING hnsw (embedding vector_cosine_ops)
				WITH (m = 16, ef_construction = 64);';
END
$$;
""".replace("&CHUNK_CLASS_NAME&", DocumentChunk_backup.__tablename__)

CREATE_BM25_CHUNK_INDEX_SQL = """
CALL paradedb.create_bm25(
  index_name => 'search_&CHUNK_CLASS_NAME&_idx',
  table_name => '&CHUNK_CLASS_NAME&',
  key_field => 'id',
  text_fields => paradedb.field('text') || paradedb.field('document_id') || paradedb.field('document_name') || paradedb.field('website_url') || paradedb.field('collection_id'),
  json_fields => paradedb.field('md') || paradedb.field('document_md')
);
""".replace("&CHUNK_CLASS_NAME&", DocumentChunk_backup.__tablename__)

DELETE_BM25_CHUNK_INDEX_SQL = """
CALL paradedb.drop_bm25(
  index_name => 'search_&CHUNK_CLASS_NAME&_idx',
  schema_name => 'public'
);
""".replace("&CHUNK_CLASS_NAME&", DocumentChunk_backup.__tablename__)

CREATE_BM25_DOC_INDEX_SQL = """
CALL paradedb.create_bm25(
  index_name => 'search_&CHUNK_CLASS_NAME&_idx',
  table_name => '&CHUNK_CLASS_NAME&',
  key_field => 'id',
  numeric_fields => paradedb.field('creation_timestamp') || paradedb.field('size_bytes'),
  text_fields => paradedb.field('file_name') || paradedb.field('website_url') ||  paradedb.field('integrity_sha256') || paradedb.field('document_collection_id'),
  json_fields => paradedb.field('md')
);
""".replace("&CHUNK_CLASS_NAME&", document_raw_backup.__tablename__)

DELETE_BM25_DOC_INDEX_SQL = """
CALL paradedb.drop_bm25(
  index_name => 'search_&CHUNK_CLASS_NAME&_idx',
  schema_name => 'public'
);
""".replace("&CHUNK_CLASS_NAME&", document_raw_backup.__tablename__)

def check_index_created(database: Session):
    result = database.exec(text(CHECK_INDEX_EXISTS_SQL))
    index_exists = list(result)[0][0]
    return index_exists

def initialize_database_engine() -> Session:
    engine = create_engine("postgresql://querylake_access:querylake_access_password@localhost:5444/querylake_database")
    
    SQLModel.metadata.create_all(engine)
    database = Session(engine)
    
    RESTART_DB = False
    
    index_exists = check_index_created(database)
    print("CHECKING IF SEARCH INDICES CREATED:", index_exists)
    
    if RESTART_DB:
        database.exec(text(DELETE_BM25_CHUNK_INDEX_SQL))
        database.commit()
        database.exec(text(CREATE_BM25_CHUNK_INDEX_SQL))
        database.commit()
        database.exec(text(DELETE_BM25_DOC_INDEX_SQL))
        database.commit()
        database.exec(text(CREATE_BM25_DOC_INDEX_SQL))
        database.commit()
    
    if not index_exists:
	    # Your SQL commands to execute if the index does not exist
        print("ATTEMPTING TO ADD INDICES...")
        database.exec(text(CREATE_VECTOR_INDEX_SQL)) 
        database.exec(text(CREATE_BM25_CHUNK_INDEX_SQL)) 
        database.exec(text(CREATE_BM25_DOC_INDEX_SQL)) 
        database.commit()
    
    index_exists = check_index_created(database)
    print("CHECKING IF SEARCH INDICES CREATED:", index_exists)
    
    # For some reason in testing, the raw SQL adds the fastest speedup despite adding the same index with the same ID.
    
    return database, engine