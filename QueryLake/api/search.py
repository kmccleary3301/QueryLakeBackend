import time
import random
from typing import Optional, Dict
from sqlmodel import Field, Session, select, func
from sqlalchemy import Column, DDL, event, text, Index, JSON, MetaData, Table
from sqlalchemy.dialects.postgresql import TSVECTOR, JSONB
from typing import List, Tuple, Union, Literal, Callable, Awaitable, Any
from pgvector.sqlalchemy import Vector
from pydantic import BaseModel
import random
import time
import re
from ..misc_functions.paradedb_query_parser import parse_search
from ..database.sql_db_tables import CHUNK_CLASS_NAME, DocumentChunk
from ..typing.config import AuthType
from .single_user_auth import get_user

class DocumentChunkDictionary(BaseModel):
    id: Union[str, int, List[str], List[int]]
    creation_timestamp: float
    collection_type: Optional[Union[str, None]]
    document_id: Optional[Union[str, None]]
    document_chunk_number: Optional[Union[int, Tuple[int, int], None]]
    # document_integrity: Optional[Union[str, None]]
    collection_id: Optional[Union[str, None]]
    document_name: str
    # website_url : Optional[Union[str, None]]
    # private: bool
    md: dict
    document_md: dict
    text: str
    
class DocumentChunkDictionaryReranked(DocumentChunkDictionary):
    rerank_score: float

chunk_dict_arguments = [
    "id", 
    "creation_timestamp", 
    "collection_type", 
    "document_id", 
    "document_chunk_number", 
    "document_integrity", 
    "collection_id", 
    "document_name", 
    "website_url", 
    "private", 
    "md", 
    "document_md", 
    "text", 
    "rerank_score"
]

field_strings_no_rerank = [e for e in chunk_dict_arguments if e not in ["rerank_score"]]
column_attributes = [getattr(DocumentChunk, e) for e in field_strings_no_rerank]
retrieved_fields_string = ", ".join(["m."+e for e in field_strings_no_rerank])
retrieved_fields_string_bm25 = ", ".join(field_strings_no_rerank)


def convert_query_result(query_results: tuple, rerank: bool = False, return_wrapped : bool = False):
    wrapped_args =  {chunk_dict_arguments[i]: query_results[i] for i in range(min(len(query_results), len(chunk_dict_arguments)))}
    if return_wrapped:
        return wrapped_args
    try:
        return DocumentChunkDictionary(**wrapped_args) if not rerank else DocumentChunkDictionaryReranked(**wrapped_args)
    except Exception as e:
        print("Error with result tuple:", query_results)
        print("Error with wrapped args:", wrapped_args)
        raise e


def find_overlap(string_a: str, string_b: str) -> int:
    max_overlap = min(len(string_a), len(string_b))
    for i in range(max_overlap, 0, -1):
        if string_a[-i:] == string_b[:i]:
            return i
    return 0

def group_adjacent_chunks(chunks: List[DocumentChunkDictionary]) -> List[DocumentChunkDictionary]:
    document_bin : Dict[str, List[DocumentChunkDictionary]] = {}
    for chunk in chunks:
        if chunk.document_id in document_bin:
            document_bin[chunk.document_id].append(chunk)
        else:
            document_bin[chunk.document_id] = [chunk]
    new_results = []
    
    for bin in document_bin:
        if len(document_bin[bin]) == 0:
            continue
        document_bin[bin] = sorted(document_bin[bin], key=lambda x: x.document_chunk_number[0] if isinstance(x.document_chunk_number, tuple) else x.document_chunk_number)
        current_chunk = document_bin[bin][0]
        most_recent_chunk_added = False
        
        for chunk in document_bin[bin][1:]:
            current_chunk_bottom_index = current_chunk.document_chunk_number[0] if isinstance(current_chunk.document_chunk_number, tuple) else current_chunk.document_chunk_number
            current_chunk_top_index = current_chunk.document_chunk_number[1] if isinstance(current_chunk.document_chunk_number, tuple) else current_chunk.document_chunk_number
            chunk_bottom_index = chunk.document_chunk_number[0] if isinstance(chunk.document_chunk_number, tuple) else chunk.document_chunk_number
            chunk_top_index = chunk.document_chunk_number[1] if isinstance(chunk.document_chunk_number, tuple) else chunk.document_chunk_number
            
            
            
            most_recent_chunk_added = False
            if chunk_bottom_index == current_chunk_top_index + 1:
                
                overlap = find_overlap(current_chunk.text, chunk.text)
                
                if overlap > 100:
                    current_chunk.text += chunk.text[overlap:]
                else:
                    current_chunk.text += "\n\n" + chunk.text
                
                current_chunk.document_chunk_number = (current_chunk_bottom_index, chunk_top_index)
                if isinstance(current_chunk.id, (int, str)):
                    current_chunk.id = [current_chunk.id]
                current_chunk.id.append(chunk.id)
                
            else:
                most_recent_chunk_added = True
                new_results.append(current_chunk)
                current_chunk = chunk
        
        if not most_recent_chunk_added:
            new_results.append(current_chunk)
        # if not most_recent_chunk_added:
    return new_results
          

async def search_hybrid(database: Session,
                        toolchain_function_caller: Callable[[Any], Union[Callable, Awaitable[Callable]]],
                        auth : AuthType,
                        query: Union[str, dict[str, str]],
                        embedding: List[float] = None,
                        collection_ids: List[str] = [],
                        limit_bm25: int = 10,
                        limit_similarity: int = 10,
                        similarity_weight: float = 0.1,
                        bm25_weight: float = 0.9,
                        return_statement : bool = False,
                        web_search : bool = False,
                        rerank : bool = False,
                        ) -> List[DocumentChunkDictionary]:
    # TODO: Check permissions on specified collections.
    
    (_, _) = get_user(database, auth)
    
    assert (len(collection_ids) > 0 or web_search), \
        "Either web search must be enabled or at least one collection must be specified"
    
    assert (isinstance(similarity_weight, (float, int)) and isinstance(bm25_weight, (float, int))), \
        "similarity_weight and bm25_weight must be floats"
    
    assert (isinstance(limit_bm25, int) and limit_bm25 >= 0 and limit_bm25 <= 200), \
        "limit_bm25 must be an int between 0 and 200"
    
    assert (isinstance(limit_similarity, int) and limit_similarity >= 0 and limit_similarity <= 200), \
        "limit_similarity must be an int between 0 and 200"
    
    # assert all(list(map(lambda x: isinstance(x, int), collection_ids))), \
    #     "All collection ids must be integers"
    
    # Prevent SQL injection with embedding
    if not embedding is None:
        assert len(embedding) == 1024 and all(list(map(lambda x: isinstance(x, (int, float)), embedding))), \
            "Embedding must be a list of 1024 floats"
    
    # Prevent SQL injection with the collection ids.
    collection_ids = list(map(lambda x: re.sub(r'[^a-zA-Z0-9]', '', x), collection_ids))
    
    if web_search:
        collection_ids.append(["WEB"])
    
    if isinstance(query, str):
        query = {key: query for key in ["bm25", "embedding"]}
        if rerank:
            query["rerank"] = query["bm25"]
    
    if (similarity_weight > 0):
        assert "embedding" in query or not (embedding is None), \
            "If similarity_weight > 0, 'embedding' must be a key in the query dictionary or must be passed as an argument"
        embedding_call : Awaitable[Callable] = toolchain_function_caller("embedding")
        embedding = (await embedding_call(auth, [query["embedding"]]))[0] if embedding is None else embedding
    else:
        embedding = [0.0]*1024
    
    formatted_query, id_exclusions = parse_search(query["bm25"], catch_all_fields=["text"], return_id_exclusions=True)
    
    print("Formatted query:", formatted_query)
    
    collection_spec = f"""collection_id:IN {str(collection_ids).replace("'", "")}"""
    
    STMT = text(f"""
	SELECT {retrieved_fields_string}
	FROM {CHUNK_CLASS_NAME} m
	RIGHT JOIN (
		SELECT * FROM search_{CHUNK_CLASS_NAME}_idx.score_hybrid(
			bm25_query => paradedb.parse('({collection_spec}) AND ({formatted_query})'),
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
        limit_bm25=limit_bm25 + len(id_exclusions),
        limit_similarity=limit_similarity + len(id_exclusions),
        bm25_weight=bm25_weight,
        similarity_weight=similarity_weight,
    )
    
    id_exclusions = set(id_exclusions)
    
    if return_statement:
        return str(STMT.compile(compile_kwargs={"literal_binds": True}))

    try:
        results = database.exec(STMT)
        results = list(results)
        results = list(filter(lambda x: not x[0] is None, results))
        
        results = list(filter(lambda x: not x[0] in id_exclusions, results))
        
        results : List[dict] = list(map(lambda x: convert_query_result(x, return_wrapped=True), results))
        database.rollback()
        
        
        if "rerank" in query:
            rerank_call : Awaitable[Callable] = toolchain_function_caller("rerank")
    
            rerank_scores = await rerank_call(auth, [
                (
                    query["rerank"], 
                    doc["text"]
                ) if len(doc["text"]) > 0 else 0 for doc in results 
            ])
            
            results : List[DocumentChunkDictionaryReranked] = list(map(lambda x: DocumentChunkDictionaryReranked(
                **results[x], 
                rerank_score=rerank_scores[x]
            ), list(range(len(results)))))
            
            results = sorted(results, key=lambda x: x.rerank_score, reverse=True)
        else:
            results = list(map(lambda x: DocumentChunkDictionary(**x), results))
        
        results = group_adjacent_chunks(results)
        
        return results
    except Exception as e:
        database.rollback()
        raise e

def search_bm25(database: Session,
                auth : AuthType,
                query: str,
                collection_ids: List[str] = [],
                limit: int = 10,
                offset: int = 0,
                web_search : bool = False,
                ) -> List[DocumentChunkDictionary]:
    
    (_, _) = get_user(database, auth)
    
    assert (len(collection_ids) > 0 or web_search), \
        "Either web search must be enabled or at least one collection must be specified"
    
    assert (isinstance(limit, int) and limit >= 0 and limit <= 200), \
        "limit must be an int between 0 and 200"
    
    assert (isinstance(offset, int) and offset >= 0), \
        "offset must be an int greater than 0"
    
    # Prevent SQL injection with the collection ids.
    collection_ids = list(map(lambda x: re.sub(r'[^a-zA-Z0-9]', '', x), collection_ids))
    
    if web_search:
        collection_ids.append(["WEB"])
    
    collection_string = str(collection_ids).replace("'", "")
    
    formatted_query = parse_search(query, catch_all_fields=["text"])
    unique_alias = "temp_alias"
    
    print("Formatted query:", formatted_query)
    # print(f'collection_id:IN {collection_string} AND {formatted_query}')
    collection_spec = f"""collection_id:IN {str(collection_ids).replace("'", "")}"""
    
    
    # # OLD QUERY; ParadeDB deprecated the highlight function.
    # STMT = text(f"""
	# SELECT {retrieved_fields_string_bm25}, 
    #        paradedb.highlight(id, field => 'text', alias => '{unique_alias}'), paradedb.rank_bm25(id, alias => '{unique_alias}')
	# FROM search_documentchunk_idx.search(
    #  	query => paradedb.parse('({collection_spec}) AND {formatted_query}'),
	# 	offset_rows => :offset,
	# 	limit_rows => :limit,
	# 	alias => '{unique_alias}'
    # )
 	# LIMIT :limit;
	# """).bindparams(
    #     limit=limit,
    #     offset=offset,
    # )
    
    STMT = text(f"""
	SELECT {retrieved_fields_string_bm25}
	FROM search_documentchunk_idx.search(
     	query => paradedb.parse('({collection_spec}) AND {formatted_query}'),
		offset_rows => :offset,
		limit_rows => :limit,
		alias => '{unique_alias}'
    )
 	LIMIT :limit;
	""").bindparams(
        limit=limit,
        offset=offset,
    )
    
    
    
    try:
        results = database.exec(STMT)
        results = list(results)
        # results = list(map(lambda x: convert_query_result(x[:-2], return_wrapped=True), results))
        # results = list(map(lambda x: convert_query_result(x[:-2]), results))
        results = list(map(lambda x: convert_query_result(x), results))
        results = group_adjacent_chunks(results)
        database.rollback()
        return results
    except Exception as e:
        database.rollback()
        raise e
    
def get_random_chunks(database: Session,
                      auth : AuthType,
                      collection_ids: List[str],
                      limit : int = 10) -> List[DocumentChunkDictionary]:
    
    (_, _) = get_user(database, auth)
    
    # assert (isinstance(offset, int) and offset >= 0), \
    #     "offset must be an int greater than 0"
    
    assert (isinstance(limit, int) and limit >= 0 and limit <= 2000), \
        "limit must be an int between 0 and 2000"
    
    # Prevent SQL injection with the collection ids.
    collection_ids = list(map(lambda x: re.sub(r'[^a-zA-Z0-9]', '', x), collection_ids))
    
    results = database.exec(
        select(*column_attributes)
        .where(
            DocumentChunk.collection_id.in_(collection_ids)
        )
        .order_by(func.random())
        # .offset(offset)
        .limit(limit)
    ).all()
    results = list(results)
    results : List[dict] = list(map(lambda x: convert_query_result(x, return_wrapped=True), results))
    
    return results
    
def count_chunks(database: Session,
                 auth : AuthType,
                 collection_ids: List[str]) -> int:
    
    (_, _) = get_user(database, auth)
    
    # Prevent SQL injection with the collection ids.
    collection_ids = list(map(lambda x: re.sub(r'[^a-zA-Z0-9]', '', x), collection_ids))
    
    count = database.exec(
        select(func.count())
        .where(DocumentChunk.collection_id.in_(collection_ids))
    ).first()
    
    return count

def expand_document_segment(database: Session,
                            auth : AuthType,
                            document_id: str,
                            chunks_to_get: List[int] = []) -> DocumentChunkDictionary:
    
    (_, _) = get_user(database, auth)
    
    chunks = database.exec(
        select(*column_attributes)
        # select(DocumentChunk)
        .where(DocumentChunk.document_id == document_id)
        .where(DocumentChunk.document_chunk_number.in_(chunks_to_get))
    ).all()
    
    chunks = list(map(lambda x: convert_query_result(x), list(chunks)))
    
    chunks = group_adjacent_chunks(chunks)
    
    return chunks