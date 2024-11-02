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
from ..database.sql_db_tables import CHUNK_CLASS_NAME, DocumentChunk, document_raw, CHUNK_INDEXED_COLUMNS, DOCUMENT_INDEXED_COLUMNS
from ..typing.config import AuthType
from .single_user_auth import get_user
from .collections import assert_collections_priviledge

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
    embedding: Optional[List[float]] = None
    
    hybrid_score: Optional[float] = None
    bm25_score: Optional[float] = None
    similarity_score: Optional[float] = None
    
class DocumentChunkDictionaryReranked(DocumentChunkDictionary):
    rerank_score: float
    
class DocumentRawDictionary(BaseModel):
    id: Optional[str]
    file_name: str
    creation_timestamp: float
    integrity_sha256: str
    size_bytes: int
    encryption_key_secure: Optional[str]
    organization_document_collection_hash_id: Optional[str] = None
    user_document_collection_hash_id: Optional[str] = None
    global_document_collection_hash_id: Optional[str] = None
    toolchain_session_id: Optional[str] = None
    website_url: Optional[str]
    blob_id: Optional[str]
    blob_dir: Optional[str]
    finished_processing: bool
    md: dict
    bm25_score: Optional[float] = None

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

document_dict_arguments = [
    "id",
    "file_name",
    "creation_timestamp",
    "integrity_sha256",
    "size_bytes",
    "encryption_key_secure",
    "organization_document_collection_hash_id",
    "user_document_collection_hash_id",
    "global_document_collection_hash_id",
    "toolchain_session_id",
    "website_url",
    "blob_id",
    "blob_dir",
    "finished_processing",
    "md"
]

document_collection_attrs = [
    "organization_document_collection_hash_id", 
    "user_document_collection_hash_id", 
    "global_document_collection_hash_id",
    "toolchain_session_id"
]
    

field_strings_no_rerank = [e for e in chunk_dict_arguments if e not in ["rerank_score"]]
column_attributes = [getattr(DocumentChunk, e) for e in field_strings_no_rerank]
retrieved_fields_string = ", ".join([f"{DocumentChunk.__tablename__}."+e for e in field_strings_no_rerank])
retrieved_fields_string_bm25 = ", ".join(field_strings_no_rerank)

document_field_strings = [e for e in document_dict_arguments]
retrieved_document_fields_string = ", ".join([f"{document_raw.__tablename__}."+e for e in document_field_strings])



def convert_chunk_query_result(query_results: tuple, rerank: bool = False, return_wrapped : bool = False):
    wrapped_args =  {chunk_dict_arguments[i]: query_results[i] for i in range(min(len(query_results), len(chunk_dict_arguments)))}
    if return_wrapped:
        return wrapped_args
    try:
        return DocumentChunkDictionary(**wrapped_args) if not rerank else DocumentChunkDictionaryReranked(**wrapped_args)
    except Exception as e:
        print("Error with result tuple:", query_results)
        print("Error with wrapped args:", wrapped_args)
        raise e
    
def convert_doc_query_result(query_results: tuple, return_wrapped : bool = False):
    wrapped_args =  {document_dict_arguments[i]: query_results[i] for i in range(min(len(query_results), len(document_dict_arguments)))}
    if return_wrapped:
        return wrapped_args
    try:
        return DocumentRawDictionary(**wrapped_args)
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
                keys_set = ["bm25_score", "similarity_score", "hybrid_score"] + (["rerank_score"] \
                        if isinstance(current_chunk, DocumentChunkDictionaryReranked) and \
                            isinstance(chunk, DocumentChunkDictionaryReranked)
                        else []
                )
                for key in keys_set:
                    if not any([getattr(chunk, key) is None, getattr(current_chunk, key) is None]):
                        max_value = max([0 if e is None else e for e in [getattr(chunk, key), getattr(current_chunk, key)]])
                        setattr(current_chunk, key, max_value)
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
                        group_chunks : bool = True
                        ) -> List[DocumentChunkDictionary]:
    # TODO: Check permissions on specified collections.
    
    (_, _) = get_user(database, auth)
    
    assert (len(collection_ids) > 0 or web_search), \
        "Either web search must be enabled or at least one collection must be specified"
    
    assert (isinstance(similarity_weight, (float, int)) and isinstance(bm25_weight, (float, int))), \
        "`similarity_weight` and bm25_weight must be floats"
    
    assert (isinstance(limit_bm25, int) and limit_bm25 >= 0 and limit_bm25 <= 200), \
        "`limit_bm25` must be an int between 0 and 200"
    
    assert (isinstance(limit_similarity, int) and limit_similarity >= 0 and limit_similarity <= 200), \
        "`limit_similarity` must be an int between 0 and 200"
    
    
    
    # Prevent SQL injection with embedding
    if not embedding is None:
        assert len(embedding) == 1024 and all(list(map(lambda x: isinstance(x, (int, float)), embedding))), \
            "Embedding must be a list of 1024 floats"
    
    # Prevent SQL injection with the collection ids.
    collection_ids = list(map(lambda x: re.sub(r'[^a-zA-Z0-9]', '', x), collection_ids))
    
    assert_collections_priviledge(database, auth, collection_ids)
    
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
    
    formatted_query, strong_where_clause = parse_search(query["bm25"], CHUNK_INDEXED_COLUMNS, catch_all_fields=["text"])
    
    print("Formatted query:", formatted_query)
    
    collection_spec = f"""collection_id:IN {str(collection_ids).replace("'", "")}"""
    collection_spec_new = f"""collection_id IN ({str(collection_ids)[1:-1]})"""
    similarity_constraint = f"WHERE id @@@ paradedb.parse('({collection_spec}) AND ({strong_where_clause})')" \
        if not strong_where_clause is None else \
        f"WHERE {collection_spec_new}"
    
    
    STMT = text(f"""
	WITH semantic_search AS (
        SELECT id, RANK () OVER (ORDER BY embedding <=> :embedding_in) AS rank
        FROM {DocumentChunk.__tablename__}
        {similarity_constraint}
        ORDER BY embedding <=> :embedding_in 
        LIMIT :limit_similarity
    ),
    bm25_search AS (
        SELECT id, RANK () OVER (ORDER BY paradedb.score(id) DESC) as rank
        FROM {DocumentChunk.__tablename__} 
        WHERE id @@@ paradedb.parse('({collection_spec}) AND ({formatted_query})')
        LIMIT :limit_bm25
    )
    SELECT
        COALESCE(semantic_search.id, bm25_search.id) AS id,
        COALESCE(1.0 / (60 + semantic_search.rank), 0.0) AS semantic_score,
        COALESCE(1.0 / (60 + bm25_search.rank), 0.0) AS bm25_score,
        COALESCE(1.0 / (60 + semantic_search.rank), 0.0) +
        COALESCE(1.0 / (60 + bm25_search.rank), 0.0) AS score,
        {retrieved_fields_string}
    FROM semantic_search
    FULL OUTER JOIN bm25_search ON semantic_search.id = bm25_search.id
    JOIN {DocumentChunk.__tablename__} ON {DocumentChunk.__tablename__}.id = COALESCE(semantic_search.id, bm25_search.id)
    ORDER BY score DESC, text;
	""").bindparams(
        embedding_in=str(embedding), 
        limit_bm25=limit_bm25,
        limit_similarity=limit_similarity,
    )
    
    if return_statement:
        return str(STMT.compile(compile_kwargs={"literal_binds": True}))

    try:
        results = database.exec(STMT)
        results = list(results)
        results = list(filter(lambda x: not x[0] is None, results))
        
        # results = list(filter(lambda x: not x[0] in id_exclusions, results))
        
        results_made : List[DocumentChunkDictionary] = list(map(lambda x: convert_chunk_query_result(x[4:]), results))
        
        for i, chunk in enumerate(results):
            results_made[i].similarity_score = float(chunk[1])
            results_made[i].bm25_score = float(chunk[2])
            results_made[i].hybrid_score = float(chunk[3])
        
        
        if group_chunks:
            results_made = group_adjacent_chunks(results_made)
        
        results = sorted(results_made, key=lambda x: x.hybrid_score, reverse=True)
        
        if "rerank" in query:
            rerank_call : Awaitable[Callable] = toolchain_function_caller("rerank")
    
            rerank_scores = await rerank_call(auth, [
                (
                    query["rerank"], 
                    doc.text
                ) if len(doc.text) > 0 else 0 for doc in results 
            ])
            
            results : List[DocumentChunkDictionaryReranked] = list(map(lambda x: DocumentChunkDictionaryReranked(
                **results[x].model_dump(), 
                rerank_score=rerank_scores[x]
            ), list(range(len(results)))))
            results = sorted(results, key=lambda x: x.rerank_score, reverse=True)
        
        database.rollback()
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
                return_statement : bool = False,
                group_chunks : bool = True,
                table : Literal["document_chunk", "document"] = "document_chunk",
                ) -> List[DocumentChunkDictionary]:
    
    (_, _) = get_user(database, auth)
    
    assert (len(collection_ids) > 0 or web_search), \
        "Either web search must be enabled or at least one collection must be specified"
    
    assert (isinstance(limit, int) and limit >= 0 and limit <= 200), \
        "limit must be an int between 0 and 200"
    
    assert (isinstance(offset, int) and offset >= 0), \
        "offset must be an int greater than or equal to 0"
    
    assert isinstance(query, str), \
        "query must be a string"
    
    assert table in ["document_chunk", "document"], \
        "`table` must be either 'document_chunk' or 'document'"
    
    group_chunks = group_chunks and (table == "document_chunk")
    
    valid_fields, chosen_table, chosen_attributes, chosen_catch_alls = {
        "document_chunk": (CHUNK_INDEXED_COLUMNS, DocumentChunk, retrieved_fields_string, ["text"]),
        "document": (DOCUMENT_INDEXED_COLUMNS, document_raw, retrieved_document_fields_string, ["file_name"])
    }[table]
    
    # Prevent SQL injection with the collection ids.
    collection_ids = list(map(lambda x: re.sub(r'[^a-zA-Z0-9]', '', x), collection_ids))
    
    assert_collections_priviledge(database, auth, collection_ids)
    
    if web_search:
        collection_ids.append(["WEB"])
    
    collection_string = str(collection_ids).replace("'", "")
    
    formatted_query, strong_where_clause = parse_search(query, valid_fields, catch_all_fields=chosen_catch_alls)
    
    print("Formatted query:", formatted_query)
    collection_spec = f"""collection_id:IN {str(collection_ids).replace("'", "")}""" if (table == "document_chunk") else \
            " OR ".join([
                    f"""({collection_attr}:IN {str(collection_ids).replace("'", "")})"""
                    for collection_attr in document_collection_attrs
                ])
    
    score_field = "paradedb.score(id) AS score, " if formatted_query != "()" else ""
    order_by_field = "ORDER BY score DESC" if formatted_query != "()" else ""
    parse_field = f"({collection_spec}) AND ({formatted_query})" if formatted_query != "()" else \
                    f"{collection_spec}"
    
    STMT = text(f"""
    SELECT id, {score_field}{chosen_attributes}
    FROM {chosen_table.__tablename__}
    WHERE id @@@ paradedb.parse('{parse_field}')
    {order_by_field}
    LIMIT :limit
    OFFSET :offset;
    """).bindparams(
        limit=limit,
        offset=offset,
    )
    
    
    if return_statement:
        return str(STMT.compile(compile_kwargs={"literal_binds": True}))
    
    try:
        subset_start = 1 if formatted_query == "()" else 2
        results = database.exec(STMT)
        results = list(results)
        # results = list(map(lambda x: convert_query_result(x[:-2], return_wrapped=True), results))
        # results = list(map(lambda x: convert_query_result(x[:-2]), results))
        if table == "document_chunk":
            results_made : List[DocumentChunkDictionary] = list(map(lambda x: convert_chunk_query_result(x[subset_start:]), results))
        else:
            results_made : List[DocumentRawDictionary] = list(map(lambda x: convert_doc_query_result(x[subset_start:]), results))
        
        if formatted_query != "()":
            for i, chunk in enumerate(results):
                results_made[i].bm25_score = float(chunk[1])
        
        if group_chunks:
            results_made = group_adjacent_chunks(results_made)
        results_made = sorted(results_made, key=lambda x: 0 if x.bm25_score is None else x.bm25_score, reverse=True)
        database.rollback()
        return results_made
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
    results : List[dict] = list(map(lambda x: convert_chunk_query_result(x, return_wrapped=True), results))
    
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
                            chunks_to_get: List[int] = [],
                            group_chunks: bool = True,
                            return_embeddings: bool = False) -> DocumentChunkDictionary:
    """
    Get the document chunks for a specific document, in the order specified by the chunks_to_get list.
    i.e. chunks_to_get = [0, 1, 2, 3] will return the first four chunks of the document.
    
    *   group chunks: If True, will group adjacent chunks together into single results, with their chunk numbers as a range, i.e. [0, 3].
    *   return_embeddings: If True, will return the embeddings as a list of floats. 
            doesn't stack perfectly with group chunks, but will still work.
    """
    
    (_, _) = get_user(database, auth)
    
    chunks = list(database.exec(
        # select(*(column_attributes))
        # select(*(column_attributes + [getattr(DocumentChunk, "embedding")]))
        select(DocumentChunk)
        .where(DocumentChunk.document_id == document_id)
        .where(DocumentChunk.document_chunk_number.in_(chunks_to_get))
    ).all())
    
    chunk_tuples = list(map(lambda c: tuple([getattr(c, e) for e in field_strings_no_rerank]), chunks))
    chunks_return = list(map(lambda x: convert_chunk_query_result(x), chunk_tuples))
    # chunks_return = list(map(lambda x: convert_query_result(x[:-1]), chunks))
    
    if return_embeddings:
        for i, chunk in enumerate(chunks):
            chunks_return[i].embedding = chunk.embedding.tolist()
    
    if group_chunks:
        chunks_return = group_adjacent_chunks(chunks_return)
    
    return chunks_return