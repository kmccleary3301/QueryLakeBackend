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
""".replace("&CHUNK_CLASS_NAME&", ORIGINAL_CHUNK_CLASS_NAME)

CREATE_VECTOR_INDEX_SQL = """
DO $$
BEGIN
	EXECUTE 'CREATE INDEX &CHUNK_CLASS_NAME&_vector_cos_idx ON &CHUNK_CLASS_NAME&
				USING hnsw (embedding vector_cosine_ops)
				WITH (m = 16, ef_construction = 64);';
END
$$;
""".replace("&CHUNK_CLASS_NAME&", ORIGINAL_CHUNK_CLASS_NAME)

def check_index_created(database: Session):
    result = database.exec(text(CHECK_INDEX_EXISTS_SQL))
    index_exists = list(result)[0][0]
    return index_exists

def initialize_database_engine() -> Session:
    # engine = create_engine("postgresql://admin:admin@localhost:5432/server_database")
    engine = create_engine("postgresql://myuser:mypassword@localhost:5432/mydatabase")
    
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
    
    return database, engine