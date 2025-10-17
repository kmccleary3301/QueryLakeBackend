import os
import time
import random
import logging
from typing import Optional
from sqlmodel import Field, SQLModel, Session, create_engine
from sqlalchemy import Column, DDL, event, text, Index
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.exc import OperationalError
from typing import List, Tuple
from pgvector.sqlalchemy import Vector

from .sql_db_tables import *

logger = logging.getLogger(__name__)


CHECK_INDEX_EXISTS_SQL = f"""
SELECT EXISTS (
    SELECT 1
    FROM   pg_class c
    JOIN   pg_namespace n ON n.oid = c.relnamespace
    WHERE  c.relname = '{DocumentChunk.__tablename__}_vector_cos_idx'
    AND    n.nspname = 'public'  -- or your schema name here
);
"""

CHECK_FILE_INDEX_EXISTS_SQL = f"""
SELECT EXISTS (
    SELECT 1
    FROM   pg_class c
    JOIN   pg_namespace n ON n.oid = c.relnamespace
    WHERE  c.relname = '{file_chunk.__tablename__}_vector_cos_idx'
    AND    n.nspname = 'public'
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

CREATE_FILE_VECTOR_INDEX_SQL = f"""
DO $$
BEGIN
    EXECUTE 'CREATE INDEX {file_chunk.__tablename__}_vector_cos_idx ON {file_chunk.__tablename__}
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

CREATE_BM25_FILE_CHUNK_INDEX_SQL = f"""
CREATE INDEX search_{file_chunk.__tablename__}_idx ON {file_chunk.__tablename__}
USING bm25 (id, text, md, created_at)
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

def check_file_index_created(database: Session):
    result = database.exec(text(CHECK_FILE_INDEX_EXISTS_SQL))
    index_exists = list(result)[0][0]
    return index_exists

def initialize_database_engine() -> Session:
    url = "postgresql://querylake_access:querylake_access_password@localhost:5444/querylake_database"
    connect_timeout = int(os.environ.get("QUERYLAKE_DB_CONNECT_TIMEOUT", "5"))
    engine = create_engine(
        url,
        pool_pre_ping=True,
        connect_args={"connect_timeout": connect_timeout},
    )
    logger.info("Connecting to ParadeDB/Postgres at %s (timeout=%ss)", url, connect_timeout)
    
    REBUILD_INDEX = False
    
    # Create tables
    try:
        SQLModel.metadata.create_all(engine)
    except OperationalError as exc:
        logger.error("Unable to reach ParadeDB/Postgres at %s: %s", url, exc)
        logger.error("Is the database container running and listening on port 5444?")
        raise
    
    # Create initial session
    database = Session(engine)
    try:
        database.exec(text("SELECT 1"))
    except OperationalError as exc:
        logger.error("Database session validation failed: %s", exc)
        raise
    
    if REBUILD_INDEX:
        logger.info("Dropping existing vector and BM25 indices per configuration")
        database.exec(text(DELETE_VECTOR_INDEX_SQL))
        database.exec(text(DELETE_BM25_CHUNK_INDEX_SQL))
        database.exec(text(DELETE_BM25_DOC_INDEX_SQL))
        database.commit()
        logger.info("Existing indices dropped successfully")
    
    
    # Check if indices exist
    index_exists = check_index_created(database)
    logger.debug("Vector/BM25 indices present: %s", index_exists)
    
    if not index_exists:
        # Close current session and create a temporary one just for index creation
        
        try:
            logger.info("Creating vector and BM25 indices for DocumentChunk/document_raw")
            database.exec(text(CREATE_VECTOR_INDEX_SQL))
            database.exec(text(CREATE_BM25_CHUNK_INDEX_SQL))
            database.exec(text(CREATE_BM25_DOC_INDEX_SQL))
            database.commit()
            
            # Verify indices were created
            index_exists = check_index_created(database)
            logger.debug("Vector/BM25 indices created: %s", index_exists)
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
        engine_2 = create_engine(
            url,
            pool_pre_ping=True,
            connect_args={"connect_timeout": connect_timeout},
        )
        database_2 = Session(engine_2)

        # Ensure new Files indices are present as well
        try:
            file_idx_exists = check_file_index_created(database_2)
            logger.debug("File chunk indices present: %s", file_idx_exists)
            if not file_idx_exists:
                logger.info("Creating file chunk vector/BM25 indices")
                database_2.exec(text(CREATE_FILE_VECTOR_INDEX_SQL))
                database_2.exec(text(CREATE_BM25_FILE_CHUNK_INDEX_SQL))
                database_2.commit()
                logger.info("File chunk indices created successfully")
        except Exception as e:
            logger.warning("Failed to create file chunk indices: %s", e)

        # Return the *fresh* sessions to prevent the aforementioned bug
        return database_2, engine_2
        
    # Ensure Files indices are present
    try:
        file_idx_exists = check_file_index_created(database)
        logger.debug("File chunk indices present: %s", file_idx_exists)
        if not file_idx_exists:
            logger.info("Creating file chunk vector/BM25 indices")
            database.exec(text(CREATE_FILE_VECTOR_INDEX_SQL))
            database.exec(text(CREATE_BM25_FILE_CHUNK_INDEX_SQL))
            database.commit()
            logger.info("File chunk indices created successfully")
    except Exception as e:
        logger.warning("Failed to create file chunk indices: %s", e)

    return database, engine
