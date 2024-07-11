import time
import random
from typing import Optional
from sqlmodel import Field, SQLModel, Session, create_engine
from sqlalchemy import Column, DDL, event, text, func, Index, JSON, MetaData, Table
from sqlalchemy.dialects.postgresql import TSVECTOR, JSONB
from typing import List, Tuple, Union, Literal
from pgvector.sqlalchemy import Vector
from pydantic import BaseModel
import random
import time
import re
from ..misc_functions.paradedb_query_parser import parse_search
from ..database.sql_db_tables import CHUNK_CLASS_NAME

class DocumentChunkDictionary(BaseModel):
    id: str
    creation_timestamp: float
    collection_type: Optional[str]
    document_id: Optional[str]
    document_chunk_number: Optional[int]
    document_integrity: Optional[str]
    parent_collection_hash_id: Optional[str]
    document_name: str
    website_url : Optional[str]
    embedding: List[float]
    private: bool
    md: dict
    text: str

def convert_query_result(query_results: tuple):
    return DocumentChunkDictionary(*query_results)

def search_hybrid(database: Session,
                  search_query: Union[str, List[str]],
                  embedding: List[float],
                  collection_ids: List[str] = [],
                  limit_bm25: int = 10,
                  limit_similarity: int = 10,
                  similarity_weight: float = 0.1,
                  bm25_weight: float = 0.9,
                  return_statement : bool = False,
                  web_search : bool = False,
                  ) -> List[DocumentChunkDictionary]:
    
    assert (len(collection_ids) > 0 or web_search), \
        "Either web search must be enabled or at least one collection must be specified"
    
    assert (isinstance(similarity_weight, (float, int)) and isinstance(bm25_weight, (float, int))), \
        "similarity_weight and bm25_weight must be floats"
    
    assert (isinstance(limit_bm25, int) and limit_bm25 >= 0 and limit_bm25 <= 200), \
        "limit_bm25 must be an int between 0 and 200"
    
    assert (isinstance(limit_similarity, int) and limit_similarity >= 0 and limit_similarity <= 200), \
        "limit_similarity must be an int between 0 and 200"
    
    # Prevent SQL injection with the collection ids.
    collection_ids = list(map(lambda x: re.sub(r"(^[a-zA-Z0-9]+)", "", x), collection_ids))
    
    if web_search:
        collection_ids.append(["WEB"])
    
    formatted_query = parse_search(search_query)
    
    STMT = text(f"""
	SELECT m.*
	FROM {CHUNK_CLASS_NAME} m
	RIGHT JOIN (
		SELECT * FROM search_chunk_collection_idx.rank_hybrid(
			bm25_query => paradedb.parse('parent_collection:IN {str(collection_ids).replace("'", "")} AND ({formatted_query})'),
			similarity_query => '':embedding_in' <=> embedding',
			bm25_weight => :bm25_weight,
			similarity_weight => :similarity_weight,
			bm25_limit_n => :limit_bm25,
			similarity_limit_n => :limit_similarity
		)
	) s
	ON m.id = s.id;
	""").bindparams(
        embedding_in=str(embedding), 
        limit_bm25=limit_bm25,
        limit_similarity=limit_similarity,
        bm25_weight=bm25_weight,
        similarity_weight=similarity_weight,
    )
    
    if return_statement:
        return STMT.compile(compile_kwargs={"literal_binds": True})

    try:
        results = database.exec(STMT)
        results = list(map(lambda x: convert_query_result(x[1:]), list(results)))
        database.rollback()
        return results
    except Exception as e:
        database.rollback()
        raise e