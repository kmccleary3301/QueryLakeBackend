# from sentence_transformers import SentenceTransformer
# from sentence_transformers import CrossEncoder
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.docstore.document import Document
from .text_chunking.character import RecursiveCharacterTextSplitter
from .text_chunking.markdown import MarkdownTextSplitter
from .text_chunking.document_class import Document
# from chromadb.api import ClientAPI
from ..database.sql_db_tables import document_raw, DocumentChunk, search_embeddings_lexical, DocumentEmbeddingDictionary
from ..api.hashing import random_hash, hash_function
from ..api.single_user_auth import get_user
from typing import List, Callable, Any, Union, Awaitable, Tuple
from numpy import ndarray
import numpy as np
from re import sub, split
from .document_parsing import parse_PDFs
from sqlmodel import Session, select, SQLModel, create_engine, and_, or_, func, not_
from ..typing.config import AuthType
import pgvector # Not sure exactly how the import works here, but it's necessary for sqlmodel queries.
from time import time
from io import BytesIO
import re
from itertools import chain
import json
import bisect

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

async def create_document_chunks(toolchain_function_caller: Callable[[Any], Union[Callable, Awaitable[Callable]]],
                                 auth : AuthType,
                                 document_bytes : bytes, 
                                 document_db_entry_id : str,
                                 document_name : str,
                                 create_embeddings : bool = True):
    """
    Add document to postgres vector database using embeddings.
    """
    
    engine = create_engine("postgresql://admin:admin@localhost:5432/server_database")
    
    SQLModel.metadata.create_all(engine)
    database = Session(engine)
    
    document_db_entry : document_raw = database.exec(select(document_raw).where(document_raw.id == document_db_entry_id)).first()
    assert not document_db_entry is None, "Document not found in database."
    assert isinstance(document_db_entry, document_raw), "Document returned is not type `document_raw`."
    
    file_extension = document_name.split(".")[-1]
    
    if file_extension in ["pdf"]:
        text_chunks = parse_PDFs(document_bytes)
    elif file_extension in ["txt", "md", "json"]:
        text = document_bytes.decode("utf-8").split("\n")
        text_chunks = list(map(lambda i: (text[i], {"line": i}), list(range(len(text)))))
    else:
        raise ValueError(f"File extension `{file_extension}` not supported, only [pdf, txt, md, json] are supported at this time.")
    
    await create_text_chunks(database, auth, toolchain_function_caller, text_chunks, 
                             document_db_entry, document_name, create_embeddings=create_embeddings)
    pass

async def create_text_chunks(database : Session,
                             auth : AuthType,
                             toolchain_function_caller: Callable[[Any], Union[Callable, Awaitable[Callable]]],
                             text_segments : List[Tuple[str, dict]],
                             document_sql_entry : document_raw, 
                             document_name: str,
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
    """

    chunk_size = 1200
    chunk_overlap = 100
    # text_splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=chunk_size, chunk_overlap=chunk_overlap
    # )
    text_splitter = MarkdownTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        add_start_index=True
    )

    if not document_sql_entry.global_document_collection_hash_id is None:
        collection_type = "global"
        parent_collection_id = document_sql_entry.global_document_collection_hash_id
    elif not document_sql_entry.user_document_collection_hash_id is None:
        collection_type = "user"
        parent_collection_id = document_sql_entry.user_document_collection_hash_id
    elif not document_sql_entry.organization_document_collection_hash_id is None:
        collection_type = "organization"
        parent_collection_id = document_sql_entry.organization_document_collection_hash_id
    elif not document_sql_entry.website_url is None:
        collection_type = "website"
        parent_collection_id = None
    else:
        return
    
    
    
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
    m_2 = time()
    print("Split %10d segments in %5.2fs" % (len(text_segments), m_2 - m_1))
 
    # print("Concatenating text with %d segments..." % (len(text_segments)))
    # for chunk_i, chunk in enumerate(text_segments):
        
    #     if (chunk_i % max(1, int(len(text_segments) / 100))) == 0:
    #         progress = (chunk_i+1)/len(text_segments)
    #         progress_bar = f"\r[{'='*int(20 * progress)}>{' '*(20 - int(20 * progress))}] " + "%3.2f%%" % (progress*100)
    #         print(progress_bar, end="\r")
        
    #     chunk_combined_strip = sub(r"[\s]+", "", chunk[0]).lower()
    #     text_combined_strip += chunk_combined_strip
    #     text_combined += chunk[0] + " "
    #     text_combined_chunk_assignments = np.concatenate((text_combined_chunk_assignments, np.full((len(chunk_combined_strip)), chunk_i, dtype=np.int32)), axis=None)
    
    
    
    
    # splits_text = [doc.page_content for doc in splits]
    # splits_metadata = []
    # text_size_iterator = 0
    
    # m_1 = time()
    # print("Finding minimum metadata for each chunk with %d splits..." % (len(splits)))
    # for doc in splits:
    #     text_stripped = sub(r"[\s]+", "", doc.page_content).lower()
    #     index = text_combined_strip.find(text_stripped)
    #     try:
    #         if index != -1:
    #             metadata = text_segments[text_combined_chunk_assignments[index]][1]
    #             text_size_iterator = index
    #         else:
    #             metadata = text_segments[text_combined_chunk_assignments[text_size_iterator+len(text_stripped)]][1]
    #         # text_size_iterator += len(text_stripped)
    #         splits_metadata.append(metadata)
    #     except:
    #         splits_metadata.append(metadata[-1])
    # m_2 = time()
    # print("Done in %3.2fs" % (m_2 - m_1))

    # del text_combined_chunk_assignments
    
    # print("Running embeddings")
    
    embeddings = None
    
    if create_embeddings:
        embedding_call : Awaitable[Callable] = toolchain_function_caller("embedding")
        embeddings = await embedding_call(auth, list(map(lambda x: x.page_content, splits)))

    for i, chunk in enumerate(splits):
        embedding_db_entry = DocumentChunk(
            collection_type=collection_type,
            document_id=document_sql_entry.id,
            document_chunk_number=i,
            collection_id=parent_collection_id,
            document_name=document_name,
            document_integrity=document_sql_entry.integrity_sha256,
            md=chunk.metadata,
            text=chunk.page_content,
            **({"website_url": document_sql_entry.website_url} if not document_sql_entry.website_url is None else {}),
            **({"embedding": embeddings[i]} if not embeddings is None else {}),
        )
        database.add(embedding_db_entry)
    document_sql_entry.finished_processing = True
    database.commit()

    pass

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

async def query_database(database : Session ,
                         auth : AuthType,
                         toolchain_function_caller: Callable[[Any], Union[Callable, Awaitable[Callable]]],
                         query : Union[str, List[str]], 
                         collection_ids : List[str], 
                         k : int = 10, 
                         use_lexical : bool = False,
                         use_embeddings : bool = True,
                         use_rerank : bool = False,
                         use_web : bool = False,
                         rerank_question : str = "",
                         ratio: float = 0.5,
                         minimum_relevance : float = 0.05) -> List[DocumentEmbeddingDictionary]:
    """
    Create an embedding for a query and lookup a certain amount of relevant chunks.
    """
    
    (_, _) = get_user(database, auth)
    
    print("QUERY ARGUMENTS:", json.dumps({
        "query": query,
        "collection_ids": collection_ids,
        "k": k,
        "use_lexical": use_lexical,
        "use_embeddings": use_embeddings,
        "use_rerank": use_rerank,
        "use_web": use_web,
        "rerank_question": rerank_question,
        "ratio": ratio,
        "minimum_relevance": minimum_relevance,
    }, indent=4))
    
    assert isinstance(query, str) or (isinstance(query, list) and isinstance(query[0], str)), "Query must be a string or a list of strings."
    assert k > 0, "k must be greater than 0."
    assert k <= 1000, "k cannot be more than 1000."
    assert any([use_lexical, use_embeddings]), "At least one of use_lexical or use_embeddings must be True."
    assert any([use_web, (len(collection_ids) > 0)]), "At least one collection must be specified or use_web must be enabled"
    
    # This is a hack to allow the web search to work and avoid locking up the code below.
    if use_web and len(collection_ids) == 0:
        collection_ids = ["fake_collection_id"] 
        
    
    first_pass_k = min(1000, 3*k if use_rerank else k)
    first_pass_k_lexical = int(first_pass_k*ratio) if (use_lexical and use_embeddings) else first_pass_k
    first_pass_k_embedding = first_pass_k - first_pass_k_lexical if use_lexical else first_pass_k
    

    chunk_size = 250
    chunk_overlap = 30
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    # splits = text_splitter.split_documents([Document(page_content=query, metadata={"source": "local"})])
    # splits_text = [doc.page_content for doc in splits]
    # query_embeddings = ndarray.tolist(model.encode(splits_text, normalize_embeddings=True))

    
    
    embedding_call : Awaitable[Callable] = toolchain_function_caller("embedding")
    
    if isinstance(query, str):
        query_embeddings = await embedding_call(auth, [query])
        ordering = DocumentChunk.embedding.cosine_distance(query_embeddings[0])
    else:
        query_embeddings = await embedding_call(auth, query)
        cosine_distances = [DocumentChunk.embedding.cosine_distance(query_embedding) for query_embedding in query_embeddings]
        ordering = func.greatest(*cosine_distances)
    
    
    
    
    if len(collection_ids) > 1:
        lookup_sql_condition = or_(
            *(DocumentChunk.collection_id == collection_hash_id for collection_hash_id in collection_ids)
        )
    else:
        lookup_sql_condition = (DocumentChunk.collection_id == collection_ids[0])
        
    if use_web:
        lookup_sql_condition = or_(
            lookup_sql_condition,
            not_(DocumentChunk.website_url == None)
        )    
    
    
    first_lookup_results : List[DocumentEmbeddingDictionary] = []
    
    if use_lexical:
        results_lexical = search_embeddings_lexical(database, collection_ids, query, first_pass_k_lexical, web_search=use_web)
        # The embedding search will now exclude our results from lexical search.
        lookup_sql_condition = and_(
            lookup_sql_condition,
            not_(or_(*[(DocumentChunk.id == e.id) for (e, _, _) in results_lexical]))
        ) if len(results_lexical) > 0 else lookup_sql_condition
        first_lookup_results.extend([e for (e, _, _) in results_lexical])
    
    # if web_query:
    #     # TODO: Check that this works later when implementing the website search.
    #     lookup_sql_condition = or_(
    #         lookup_sql_condition,
    #         (DocumentEmbedding.website_url != None)
    #     )

    if not use_embeddings:
        first_pass_results = first_lookup_results
    else:
        selection = select(DocumentChunk).where(lookup_sql_condition) \
                                    .order_by(
                                        ordering
                                    ) \
                                    .limit(first_pass_k_embedding)
        
        first_pass_results : List[DocumentChunk] = database.exec(selection)
        
        first_pass_results : List[DocumentEmbeddingDictionary] = list(map(
            lambda x: DocumentEmbeddingDictionary(**{
                key : v \
                for key, v in x.__dict__.items() \
                if key in DocumentEmbeddingDictionary.model_fields.keys()
            }),
            first_pass_results
        ))
    
        # Evenly combine the lexical and embedding results, with lexical coming first.
        if use_lexical:
            first_pass_results_split = split_list(first_pass_results, len(first_lookup_results))
            first_pass_results = [[lookup_result, *first_pass_results_split[i]] for i, lookup_result in enumerate(first_lookup_results)]
            first_pass_results : List[DocumentEmbeddingDictionary] = list(chain.from_iterable(first_pass_results))
    
    new_docs_dict = {} # Remove duplicates
    for i, doc in enumerate(first_pass_results):
        # print(doc)
        content_hash = hash_function(hash_function(doc.text)+hash_function(doc.document_integrity))
        new_docs_dict[content_hash] = {
            key : v for key, v in doc.__dict__.items() if key not in ["_sa_instance_state", "embedding"] and v is not None
        }
        
        
        # cosine_similarity = np.dot(doc.embedding, query_embeddings) / (np.linalg.norm(doc.embedding) * np.linalg.norm(query_embeddings))
        # new_docs_dict[content_hash]["embedding_similarity_dot"] = doc.embedding @ query_embeddings
        # new_docs_dict[content_hash]["embedding_similarity_cosine_similarity"] = cosine_similarity
        
    
    new_documents = list(new_docs_dict.values())
    

    if not use_rerank:
        return new_documents[:min(len(new_documents), k)]

    assert rerank_question is not None, "If using rerank, a rerank question must be provided."
    
    rerank_call : Awaitable[Callable] = toolchain_function_caller("rerank")
    
    rerank_scores = await rerank_call(auth, [
        (
            rerank_question, 
            doc["text"]
        ) for doc in new_documents if len(doc["text"]) > 0
    ])
    
    for i in range(len(new_documents)):
        new_documents[i]["rerank_score"] = rerank_scores[i]
    
    reranked_pairs = sorted(new_documents, key=lambda x: x["rerank_score"], reverse=True)
    
    reranked_pairs = [doc for doc in reranked_pairs if doc["rerank_score"] > minimum_relevance]
    
    return reranked_pairs[:min(len(reranked_pairs), k)]

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

