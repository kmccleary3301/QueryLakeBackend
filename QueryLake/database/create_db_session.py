import time
import random
from typing import Optional
from sqlmodel import Field, SQLModel, Session, create_engine
from sqlalchemy import Column, DDL, event, text, Index
from sqlalchemy.dialects.postgresql import TSVECTOR
from typing import List, Tuple
from pgvector.sqlalchemy import Vector

from .sql_db_tables import *


CHECK_INDEX_EXISTS_SQL = f"""
SELECT EXISTS (
    SELECT 1
    FROM   pg_class c
    JOIN   pg_namespace n ON n.oid = c.relnamespace
    WHERE  c.relname = '{DocumentChunk.__tablename__}_vector_cos_idx'
    AND    n.nspname = 'public'  -- or your schema name here
);
"""

CREATE_VECTOR_INDEX_SQL = f"""
DO $$
BEGIN
	EXECUTE 'CREATE INDEX {DocumentChunk.__tablename__}_vector_cos_idx ON {DocumentChunk.__tablename__}
				USING hnsw (embedding halfvec_cosine_ops)
				WITH (m = 16, ef_construction = 64);';
END
$$;
"""


DELETE_VECTOR_INDEX_SQL = f"""
DROP INDEX {DocumentChunk.__tablename__}_vector_cos_idx;
"""

# CREATE_BM25_CHUNK_INDEX_SQL = """
# CALL paradedb.create_bm25(
#   index_name => 'search_&CHUNK_CLASS_NAME&_idx',
#   table_name => '&CHUNK_CLASS_NAME&',
#   key_field => 'id',
#   text_fields => paradedb.field('text') || paradedb.field('document_id') || paradedb.field('document_name') || paradedb.field('website_url') || paradedb.field('collection_id'),
#   numeric_fields => paradedb.field('creation_timestamp') || paradedb.field('document_chunk_number'),
#   json_fields => paradedb.field('md') || paradedb.field('document_md')
# );
# """.replace("&CHUNK_CLASS_NAME&", DocumentChunk_backup.__tablename__)

CREATE_BM25_CHUNK_INDEX_SQL = f"""
CREATE INDEX search_{DocumentChunk.__tablename__}_idx ON {DocumentChunk.__tablename__}
USING bm25 (id, text, document_id, document_name, website_url, collection_id, creation_timestamp, document_chunk_number, md, document_md)
WITH (key_field = 'id');
"""

DELETE_BM25_CHUNK_INDEX_SQL = f"""
DROP INDEX search_{DocumentChunk.__tablename__}_idx;
"""

CREATE_BM25_DOC_INDEX_SQL = f"""
CREATE INDEX search_{document_raw.__tablename__}_idx ON {document_raw.__tablename__}
USING bm25 (id, creation_timestamp, size_bytes, file_name, website_url, integrity_sha256, document_collection_id, md, finished_processing)
WITH (key_field = 'id');
"""

DELETE_BM25_DOC_INDEX_SQL = f"""
DROP INDEX search_{document_raw.__tablename__}_idx;
"""

def check_index_created(database: Session):
    result = database.exec(text(CHECK_INDEX_EXISTS_SQL))
    index_exists = list(result)[0][0]
    return index_exists

def initialize_database_engine() -> Session:
    url = "postgresql://querylake_access:querylake_access_password@localhost:5444/querylake_database"
    engine = create_engine(url)
    print("PG URL:", url)
    
    REBUILD_INDEX = True
    
    # Create tables
    SQLModel.metadata.create_all(engine)
    
    # Create initial session
    database = Session(engine)
    
    if REBUILD_INDEX:
        print("Deleting existing indices...")
        database.exec(text(DELETE_VECTOR_INDEX_SQL))
        database.exec(text(DELETE_BM25_CHUNK_INDEX_SQL))
        database.exec(text(DELETE_BM25_DOC_INDEX_SQL))
        database.commit()
        print("Deleting existing indices...")
    
    
    # Check if indices exist
    index_exists = check_index_created(database)
    print("CHECKING IF SEARCH INDICES CREATED:", index_exists)
    
    if not index_exists:
        # Close current session and create a temporary one just for index creation
        
        try:
            print("ATTEMPTING TO ADD INDICES...")
            database.exec(text(CREATE_VECTOR_INDEX_SQL))
            database.exec(text(CREATE_BM25_CHUNK_INDEX_SQL))
            database.exec(text(CREATE_BM25_DOC_INDEX_SQL))
            database.commit()
            
            # Verify indices were created
            index_exists = check_index_created(database)
            print("VERIFYING INDICES CREATED:", index_exists)
        finally:
            database.close()
        
        # Completely close the session and make a new one to return.
        # This prevents a baffling machine-dependent bug (only occurs on one of my machines)
        # Essentially, on the first time running the server,
        # a rollback+flush would delete the BM25 indices
        # *despite* them being committed first.
        
        # I tried very hard to figure out why this happened, but could not.
        # However, running the server for the first time and instantly
        # restarting it prevented the issue. Thus, I am using this workaround.
        del database, engine
        
        # Create fresh session after index creation
        engine_2 = create_engine(url)
        database_2 = Session(engine_2)
        
        # Return the *fresh* sessions to prevent the aforementioned bug
        return database_2, engine_2
        
    
    return database, engine