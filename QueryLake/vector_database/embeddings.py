# from sentence_transformers import SentenceTransformer
# from sentence_transformers import CrossEncoder
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.docstore.document import Document
from .text_chunking.character import RecursiveCharacterTextSplitter
from .text_chunking.markdown import MarkdownTextSplitter
from .text_chunking.document_class import Document
# from chromadb.api import ClientAPI
from ..database.sql_db_tables import document_raw, DocumentChunk, DocumentEmbeddingDictionary, document_collection
from ..api.hashing import random_hash, hash_function
from ..api.single_user_auth import get_user
from typing import List, Callable, Any, Union, Awaitable, Tuple, Literal
from numpy import ndarray
import numpy as np
from re import sub, split
from .document_parsing import parse_PDFs
from sqlmodel import Session, select, SQLModel, create_engine, and_, or_, func, not_, update
from ..typing.config import AuthType
import pgvector # Not sure exactly how the import works here, but it's necessary for sqlmodel queries.
from time import time
from io import BytesIO
import re
from itertools import chain
import json
import bisect
from ..database.create_db_session import initialize_database_engine
from ..typing.api_inputs import TextChunks

def binary_search(sorted_list, target):
	index = bisect.bisect_right(sorted_list, target)
	index = min(max(0, index), len(sorted_list)-1)
	value = sorted_list[index]
	while value > target and index > 0:
		if value > target:
			index = max(0, index-1)
		value = sorted_list[index]
	return index
# from sqlmodel import Session, SQLModel, create_engine, select


# model = SentenceTransformer('BAAI/bge-large-en-v1.5')
# reranker_ce = CrossEncoder('BAAI/bge-reranker-base')
# model_collection_name = "bge-large-en-v1-5"

def split_list(input_list : list, n : int) -> List[list]:
    """
    Evenly split a list into `n` sublists of approximately equal length.
    """
    if n <= 0:
        return [input_list]
    k, m = divmod(len(input_list), n)
    return [input_list[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

async def chunk_documents(toolchain_function_caller: Callable[[Any], Union[Callable, Awaitable[Callable]]],
                          database : Session,
                          auth : AuthType,
                          document_db_entries: List[Union[str, document_raw]],
                          document_names : List[str],
                          document_bytes_list : List[bytes] = None, 
                          document_texts : List[Union[str, List[TextChunks]]] = None,
                          document_metadata : List[Union[dict, Literal[None]]] = None,
                          create_embeddings : bool = True):
    """
    Add document batch to postgres vector database using embeddings.
    
    Returns: A dictionary breaking down time spent on different tasks.
    
    Named keys:
        - "chunking"
        - "embedding"
        - "database_add"
        - "total"
    """
    start_time = time()
    
    assert not document_bytes_list is None or not document_texts is None, "Either document_bytes_list or document_texts must be provided."
    
    if not document_bytes_list is None:
        assert len(document_db_entries) == len(document_bytes_list) and len(document_db_entries) == len(document_names),\
            "Length of document_bytes_list, document_db_entries, and document_names must be the same."
    else:
        assert len(document_db_entries) == len(document_texts) and len(document_db_entries) == len(document_names),\
            "Length of document_texts, document_db_entries, and document_names must be the same."

    text_segment_collections = []
    real_db_entries = []
    
    # db_2, engine_2 = initialize_database_engine()
    
    for i in range(len(document_db_entries)):
        
        document_db_entry = document_db_entries[i]
        document_name = document_names[i]
        
        if isinstance(document_db_entry, str):
            document_db_entry : document_raw = database.exec(select(document_raw).where(document_raw.id == document_db_entry)).first()
            assert not document_db_entry is None, "Document not found in database."
        
        assert isinstance(document_db_entry, document_raw), "Document returned is not type `document_raw`."
        real_db_entries.append(document_db_entry)
        
        file_extension = document_name.split(".")[-1].lower()
        
        if document_texts is None:
            document_bytes = document_bytes_list[i]
            assert isinstance(document_bytes, bytes), "Document bytes must be a bytes object."
            
            if file_extension in ["pdf"]:
                text_chunks = parse_PDFs(document_bytes)
            elif file_extension in ["txt", "md", "json"]:
                text = document_bytes.decode("utf-8").split("\n")
                text_chunks = list(map(lambda i: (text[i], {"line": i}), list(range(len(text)))))
            else:
                raise ValueError(f"File extension `{file_extension}` not supported for scanning, only [pdf, txt, md, json] are supported at this time.")
        else:
            if isinstance(document_texts[i], str):
                text = document_texts[i].split("\n")
                text_chunks : List[Tuple[str, dict]] = list(map(lambda i: (
                    text[i], {"line": i}
                ), list(range(len(text)))))
            elif isinstance(document_texts[i], list) and all([isinstance(e, TextChunks) for e in document_texts[i]]):
                text_chunks : List[Tuple[str, dict]] = list(map(lambda x: (
                    x.text, x.metadata if not x.metadata is None else {}
                ), document_texts[i]))
            else:
                raise ValueError(f"Document text must be a string or a list of TextChunks.")

        text_segment_collections.append(text_chunks)
    
    if document_metadata is None:
        document_metadata = [None]*len(document_db_entries)
    # try:
    time_dict = await create_text_chunks(database, auth, toolchain_function_caller, text_segment_collections, 
                            real_db_entries, document_names, document_metadata=document_metadata,
                            create_embeddings=create_embeddings)
    
    time_dict["total"] = time() - start_time
    
    return time_dict


async def create_text_chunks(database : Session,
                             auth : AuthType,
                             toolchain_function_caller: Callable[[Any], Union[Callable, Awaitable[Callable]]],
                             text_segment_collections : List[List[Tuple[str, dict]]],
                             document_sql_entries : List[document_raw], 
                             document_names: List[str],
                             document_metadata : List[Union[dict, Literal[None]]],
                             create_embeddings : bool = True):
    """
    Given a set of text chunks, possibly pairs with metadata, create embeddings for the
    entries. Craft an entry in the vector db, using the collection relevant to the
    model used. Each entry into the vector db will have the following metadata:

    collection_type - whether the parent collection is an org, user, or global collection.
    public - bool for if this is public or not.
    parent_collection_id - sql db hash id for the parent collection
    document_id - sql db hash id for the parent document.
    page - what page of the original document the chunk is from.
    document_name - name of the original document.
    
    Returns: A dictionary breaking down time spent on different tasks.
    
    Named keys:
        - "chunking"
        - "embedding"
        - "database_add"
        - "total"
    """

    time_dict = {
        "chunking": 0,
        "embedding": 0,
        "database_add": 0,
        "misc": 0
    }
    start_time = time()
    
    assert len(text_segment_collections) == len(document_sql_entries) and len(text_segment_collections) == len(document_names),\
        "Length of text_segment_collections, document_sql_entries, and document_names must be the same."

    chunk_size = 1200
    chunk_overlap = 100
    text_splitter = MarkdownTextSplitter( # Same implemetation as LangChain
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        add_start_index=True
    )
    
    current_chunks_queued = 0
    embedding_call : Awaitable[Callable] = toolchain_function_caller("embedding")
    
    collection = database.exec(select(document_collection).where(document_collection.id == document_sql_entries[0].document_collection_id)).first()
    assert not collection is None, "Collection not found."
    collection_type = collection.collection_type
    parent_collection_id = collection.id
    
    split_collections = []
    
    chunking_start_time = time()
    
    for i in range(len(text_segment_collections)):
        document_sql_entry = document_sql_entries[i]
        text_segments = text_segment_collections[i]
        document_name = document_names[i]
        
        # We are assuming an input of tuples with text and metadata.
        # We do this to keep track of location/ordering metadata such as page number.
        # This code concatenates the text and allows us to recover the minimum location index
        # After splitting for embedding.
        
        created_document = "\n".join([chunk[0] for chunk in text_segments])
        
        m_1 = time()
        doc_markers, current_sum = [], 0
        for i in range(len(text_segments)):
            doc_markers.append(current_sum)
            current_sum += len(text_segments[i][0]) + 1
        
        splits = text_splitter.split_documents([Document(page_content=created_document, metadata={})])
        splits = list(map(lambda x: Document(
            page_content=x.page_content,
            metadata={**text_segments[binary_search(doc_markers, x.metadata["start_index"])][1]}
        ), splits))
        m_2 = time() - m_1
        if m_2 > 1:
            print("Split %16d segments in %5.2fs" % (len(text_segments), m_2))
        split_collections.append(splits)
    
    
    embeddings_all, embeddings_iterable = None, 0
    
    flattened_splits = list([e.page_content for e in chain.from_iterable(split_collections)])
    
    time_dict["chunking"] = time() - chunking_start_time
    embedding_start_time = time()
    
    if create_embeddings:
        embeddings_all = await embedding_call(auth, flattened_splits)
    
    time_dict["embedding"] = time() - embedding_start_time
    
    db_additions, document_md_lookup = [], {}
    
    db_add_start_time = time()
    
    for i, doc in enumerate(document_sql_entries):
        document_md_lookup[doc.id] = document_metadata[i]
    
    db_additions = []
    docs_to_finish_3, docs_to_finish_4 = [], []
    for seg_i in range(len(text_segment_collections)):
        document_sql_entry = document_sql_entries[seg_i]
        text_segments = text_segment_collections[seg_i]
        document_name = document_names[seg_i]
        splits : List[Document] = split_collections[seg_i]
        current_split_size = len(splits)
        document_metadata = document_md_lookup[document_sql_entry.id]
        
        embeddings = None
        if create_embeddings:
            embeddings = embeddings_all[embeddings_iterable:embeddings_iterable+current_split_size]
        
        
        for i, chunk in enumerate(splits):
            chunk_md = chunk.metadata
            assert isinstance(chunk_md, dict), "Metadata must be a dictionary."
            parent_doc_md = document_sql_entry.md
            assert isinstance(parent_doc_md, dict), "Parent document metadata must be a dictionary."
            document_md_make = {
                **(document_metadata if not document_metadata is None else {}),
                **parent_doc_md,
            }
            
            embedding_db_entry = DocumentChunk(
                # collection_type=collection_type,
                document_id=document_sql_entry.id,
                document_chunk_number=i,
                collection_id=parent_collection_id,
                collection_type=collection_type,
                document_name=document_name,
                document_integrity=document_sql_entry.integrity_sha256,
                md=chunk_md,
                document_md=document_md_make,
                text=chunk.page_content,
                **({"website_url": document_sql_entry.website_url} if not document_sql_entry.website_url is None else {}),
                **({"embedding": embeddings[i]} if not embeddings is None else {}),
            )
            db_additions.append(embedding_db_entry)
        
        if embeddings is None:
            docs_to_finish_3.append(document_sql_entry.id)
        else:
            docs_to_finish_4.append(document_sql_entry.id)
        embeddings_iterable += current_split_size
    
    if len(db_additions) > 0:
        database.add_all(db_additions)

    if len(docs_to_finish_3) > 0:
        stmt = update(document_raw).where(document_raw.id.in_(docs_to_finish_3)).values(finished_processing=3)
        database.exec(stmt)
    
    if len(docs_to_finish_4) > 0:
        stmt = update(document_raw).where(document_raw.id.in_(docs_to_finish_4)).values(finished_processing=4)
        database.exec(stmt)
    
    database.commit()
    
    time_dict["database_add"] = time() - db_add_start_time
    time_dict["total"] = time() - start_time
    
    return time_dict

async def create_website_embeddings(database : Session, 
                                    auth : AuthType,
                                    toolchain_function_caller: Callable[[Any], Union[Callable, Awaitable[Callable]]],
                                    url_content_pairs : list):
    """
    Given a set of text chunks, possibly pairs with metadata, create embeddings for the
    entries. Craft an entry in the chroma db, using the collection relevant to the
    model used. Each entry into the chroma db will have the following metadata:

    collection_type - whether the parent collection is an org, user, or global collection.
    public - bool for if this is public or not.
    parent_collection_id - sql db hash id for the parent collection
    document_id - sql db hash id for the parent document.
    page - what page of the original document the chunk is from.
    document_name - name of the original document.
    """

    chunk_size = 600
    chunk_overlap = 120
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    
    # Split
    urls_shortened = [sub(r"^www\.", "", sub(r"https\:\/\/|http\:\/\/", "", url)).split("/")[0] for (url, _) in url_content_pairs]
    

    splits = text_splitter.split_documents([Document(page_content=url_content, metadata={"type": "website", "url": url, "website_short": urls_shortened[i]}) for i, (url, url_content) in enumerate(url_content_pairs)])

    splits_text = [doc.page_content for doc in splits]
    splits_metadata = [doc.metadata for doc in splits]
    
    embedding_call : Awaitable[Callable] = toolchain_function_caller("embedding")
    
    embeddings = await embedding_call(auth, splits_text)

    documents = [random_hash() for _ in splits]

    for i in range(len(embeddings)):
        
        document_db_entry = DocumentChunk(
            document_name=splits_metadata[i]["website_short"],
            website_url=splits_metadata[i]["url"],
            embedding=embeddings[i],
            text=splits_text[i],
        )
        database.add(document_db_entry)

    database.commit()
    
    pass

async def keyword_query(database : Session ,
                        auth : AuthType,
                        query : str, 
                        collection_hash_ids : List[str], 
                        k : int = 10):
    (_, _) = get_user(database, auth)
    
    if len(collection_hash_ids) > 1:
        lookup_sql_condition = or_(
            *(DocumentChunk.collection_id == collection_hash_id for collection_hash_id in collection_hash_ids)
        )
    else:
        
        lookup_sql_condition = (DocumentChunk.collection_id == collection_hash_ids[0])
    
    selection = select(DocumentChunk).where(lookup_sql_condition).limit(k)
    
    first_pass_results : List[DocumentChunk] = database.exec(selection)
    
    new_docs_dict = {} # Remove duplicates
    for i, doc in enumerate(first_pass_results):
        # print(doc)
        content_hash = hash_function(hash_function(doc.text)+hash_function(doc.document_integrity))
        new_docs_dict[content_hash] = {
            key : v for key, v in doc.__dict__.items() if key not in ["_sa_instance_state", "embedding"]
        }
    
    new_documents = list(new_docs_dict.values())
    
    return new_documents[:min(len(new_documents), k)]

def concat_without_overlap(strings):
    result = strings[0]
    for s in strings[1:]:
        overlap = min(len(result), len(s))
        while overlap > 0 and result[-overlap:] != s[:overlap]:
            overlap -= 1
        result += s[overlap:]
    return result

def expand_source(database : Session,
                  auth : AuthType,
                  chunk_id : str,
                  range : Tuple[int, int]):
    """
    Given the id of an embedding chunk from the vector database, get the surrounding chunks to provide context.
    """
    (_, _) = get_user(database, auth)
    
    assert isinstance(range, tuple) and len(range) == 2, "Range must be a tuple of two integers."
    assert range[0] < 0, "Range start must be less than 0."
    assert range[1] > 0, "Range end must be greater than 0."
    
    main_chunk = database.exec(select(DocumentChunk).where(DocumentChunk.id == chunk_id)).first()
    
    assert not main_chunk is None, "Chunk not found"
    main_chunk_index = main_chunk.document_chunk_number
    
    chunk_range : List[DocumentChunk] = database.exec(select(DocumentChunk).where(and_(
        DocumentChunk.document_id == main_chunk.document_id,
        DocumentChunk.document_chunk_number >= main_chunk_index + range[0],
        DocumentChunk.document_chunk_number <= main_chunk_index + range[1],
    ))).all()
    
    chunk_range : List[DocumentChunk] = sorted(chunk_range, key=lambda x: x.document_chunk_number)
    
    chunks_of_text = [chunk.text for chunk in chunk_range]
    
    # return concat_without_overlap(chunks_of_text)
    
    return DocumentEmbeddingDictionary(
        id=chunk_range[0].id,
        creation_timestamp=chunk_range[0].creation_timestamp,
        document_id=chunk_range[0].document_id,
        document_chunk_number=chunk_range[0].document_chunk_number,
        document_integrity=chunk_range[0].document_integrity,
        collection_id=chunk_range[0].collection_id,
        document_name=chunk_range[0].document_name,
        website_url=chunk_range[0].website_url,
        private=chunk_range[0].private,
        text=concat_without_overlap(chunks_of_text),
    ).model_dump()