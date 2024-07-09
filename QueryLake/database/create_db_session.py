import time
import random
from faker import Faker
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
    WHERE  c.relname = 'documentembedding_vector_cos_idx'
    AND    n.nspname = 'public'  -- or your schema name here
);
"""

CREATE_VECTOR_INDEX_SQL = """
DO $$
BEGIN
	EXECUTE 'CREATE INDEX documentembedding_vector_cos_idx ON documentembedding
				USING hnsw (embedding vector_cosine_ops)
				WITH (m = 16, ef_construction = 64);';
END
$$;
"""

CREATE_BM25_INDEX_SQL = """
CALL paradedb.create_bm25(
	index_name => 'ql_search_idx',
	table_name => 'documentembedding',
	key_field => 'id',
	text_fields => '{
		text: {tokenizer: {type: "en_stem"}},
        document_id: {},
        website_url: {},
        parent_collection_hash_id: {}
	}'
);
"""

def check_index_created(database: Session):
    result = database.exec(text(CHECK_INDEX_EXISTS_SQL))
    index_exists = list(result)[0][0]
    return index_exists

def initialize_database_engine() -> Session:
    engine = create_engine("postgresql://admin:admin@localhost:5432/server_database")
    
    SQLModel.metadata.create_all(engine)
    
    database = Session(engine)
    
    index_exists = check_index_created(database)
    print("CHECKING IF SEARCH INDICES CREATED:", index_exists)
    
    if not index_exists:
	    # Your SQL commands to execute if the index does not exist
        print("ATTEMPTING TO ADD INDICES...")
        database.exec(text(CREATE_VECTOR_INDEX_SQL)) 
        database.exec(text(CREATE_BM25_INDEX_SQL)) 
        database.commit()
    
    index_exists = check_index_created(database)
    print("CHECKING IF SEARCH INDICES CREATED:", index_exists)
    
    # For some reason in testing, the raw SQL adds the fastest speedup despite adding the same index with the same ID.
    
    return database