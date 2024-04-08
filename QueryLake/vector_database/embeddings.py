# from sentence_transformers import SentenceTransformer
# from sentence_transformers import CrossEncoder
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
# from chromadb.api import ClientAPI
from ..database.sql_db_tables import document_raw, DocumentEmbedding, search_embeddings_lexical, DocumentEmbeddingDictionary
from ..api.hashing import random_hash, hash_function
from ..api.single_user_auth import get_user
from typing import List, Callable, Any, Union, Awaitable, Tuple
from numpy import ndarray
import numpy as np
from re import sub, split
from .document_parsing import parse_PDFs
from sqlmodel import Session, select, SQLModel, create_engine, and_, or_, func, not_
from ..typing.config import AuthType
import pgvector
from time import time
from io import BytesIO
import re
from itertools import chain
import json
# from sqlmodel import Session, SQLModel, create_engine, select


# model = SentenceTransformer('BAAI/bge-large-en-v1.5')
# reranker_ce = CrossEncoder('BAAI/bge-reranker-base')
# model_collection_name = "bge-large-en-v1-5"

def split_list(input_list : list, n : int) -> List[list]:
    """
    Evenly split a list into `n` sublists of approximately equal length.
    """
    k, m = divmod(len(input_list), n)
    return [input_list[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

async def create_embeddings_in_database(
                                        # database : Session,
                                        toolchain_function_caller: Callable[[Any], Union[Callable, Awaitable[Callable]]],
                                        auth : AuthType,
                                        document_bytes : bytes, 
                                        document_db_entry_id : str,
                                        document_name : str):
    """
    Add document to chroma db using embeddings.
    """
    
    engine = create_engine("postgresql://admin:admin@localhost:5432/server_database")
        
    SQLModel.metadata.create_all(engine)
    database = Session(engine)
    
    document_db_entry : document_raw = database.exec(select(document_raw).where(document_raw.id == document_db_entry_id)).first()
    assert not document_db_entry is None, "Document not found in database."
    assert isinstance(document_db_entry, document_raw), "Document returned is not type `document_raw`."
    
    # print(json.dumps(document_db_entry)
    
    file_extension = document_name.split(".")[-1]
    
    print("Creating document embeddings")
    
    if file_extension in ["pdf"]:
        text_chunks = parse_PDFs(document_bytes)
    elif file_extension in ["txt", "md"]:
        # text_chunks = [(chunk, {"page": i}) for i, chunk in enumerate(document_bytes.decode("utf-8").split("\n"))]
        text_chunks = [(document_bytes.decode("utf-8"), {"page": 0})]
    else:
        raise ValueError(f"File extension `{file_extension}` not supported, only [pdf, txt, md] are supported at this time.")
    
    await create_text_embeddings(database, auth, toolchain_function_caller, text_chunks, document_db_entry, document_name)
    print("Done creating document embeddings")
    pass

async def create_text_embeddings(database : Session,
                                 auth : AuthType,
                                 toolchain_function_caller: Callable[[Any], Union[Callable, Awaitable[Callable]]],
                                 text_segments : List[Tuple[str, dict]],
                                 document_sql_entry : document_raw, 
                                 document_name: str):
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

    if not document_sql_entry.global_document_collection_hash_id is None:
        collection_type = "global"
        parent_collection_id = document_sql_entry.global_document_collection_hash_id
    elif not document_sql_entry.user_document_collection_hash_id is None:
        collection_type = "user"
        parent_collection_id = document_sql_entry.user_document_collection_hash_id
    elif not document_sql_entry.organization_document_collection_hash_id is None:
        collection_type = "organization"
        parent_collection_id = document_sql_entry.organization_document_collection_hash_id
    else:
        return

    text_combined, text_combined_strip = "", ""
    text_combined_chunk_assignments = np.array([], dtype=np.int32)
    
    # We are assuming an input of tuples with text and metadata.
    # We do this to keep track of location/ordering metadata such as page number.
    # This code concatenates the text and allows us to recover the minimum location index
    # After splitting for embedding.
    
    print("Concatenating text")
    for chunk_i, chunk in enumerate(text_segments):
        chunk_combined_strip = sub(r"[\s]+", "", chunk[0]).lower()
        text_combined_strip += chunk_combined_strip
        text_combined += chunk[0] + " "
        text_combined_chunk_assignments = np.concatenate((text_combined_chunk_assignments, np.full((len(chunk_combined_strip)), chunk_i, dtype=np.int32)), axis=None)


    print("Splitting text")
    splits = text_splitter.split_documents([Document(page_content=text_combined, metadata={"type": "document"})])

    print("Re-assigning metadata")
    splits_text = [doc.page_content for doc in splits]
    splits_metadata = []
    text_size_iterator = 0
    for doc in splits:
        text_stripped = sub(r"[\s]+", "", doc.page_content).lower()
        index = text_combined_strip.find(text_stripped)
        try:
            if index != -1:
                metadata = text_segments[text_combined_chunk_assignments[index]][1]
                text_size_iterator = index
            else:
                metadata = text_segments[text_combined_chunk_assignments[text_size_iterator+len(text_stripped)]][1]
            # text_size_iterator += len(text_stripped)
            splits_metadata.append(metadata)
        except:
            splits_metadata.append(metadata[-1])

    del text_combined_chunk_assignments
    
    print("Running embeddings")
    embedding_call : Awaitable[Callable] = toolchain_function_caller("embedding")
    
    embeddings = await embedding_call(auth, splits_text)

    print("Adding to database")
    for i in range(len(embeddings)):
        
        document_db_entry = DocumentEmbedding(
            collection_type=collection_type,
            document_id=document_sql_entry.hash_id,
            parent_collection_hash_id=parent_collection_id,
            document_name=document_name,
            document_integrity=document_sql_entry.integrity_sha256,
            embedding=embeddings[i],
            text=splits_text[i]
        )
        database.add(document_db_entry)
        
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
        
        document_db_entry = DocumentEmbedding(
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
                         rerank_question : str = "",
                         web_query : bool = False,
                         ratio: float = 0.5,
                         minimum_relevance : float = 0.05):
    """
    Create an embedding for a query and lookup a certain amount of relevant chunks.
    """
    
    (_, _) = get_user(database, auth)
    
    assert isinstance(query, str) or (isinstance(query, list) and isinstance(query[0], str)), "Query must be a string or a list of strings."
    assert k > 0, "k must be greater than 0."
    assert k <= 1000, "k cannot be more than 1000."
    assert any([use_lexical, use_embeddings]), "At least one of use_lexical or use_embeddings must be True."
    
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
        
        ordering = DocumentEmbedding.embedding.cosine_distance(query_embeddings[0])
    else:
        query_embeddings = await embedding_call(auth, query)
        cosine_distances = [DocumentEmbedding.embedding.cosine_distance(query_embedding) for query_embedding in query_embeddings]
        ordering = func.greatest(*cosine_distances)
        
        
    
    
    if len(collection_ids) > 1:
        lookup_sql_condition = or_(
            *(DocumentEmbedding.parent_collection_hash_id == collection_hash_id for collection_hash_id in collection_ids)
        )
    else:
        lookup_sql_condition = (DocumentEmbedding.parent_collection_hash_id == collection_ids[0])
    
    
    first_lookup_results : List[DocumentEmbeddingDictionary] = []
    
    if use_lexical:
        results_lexical = search_embeddings_lexical(database, collection_ids, query, first_pass_k_lexical)
        # The embedding search will now exclude our results from lexical search.
        lookup_sql_condition = and_(
            lookup_sql_condition,
            not_(or_(*[(DocumentEmbedding.id == e.id) for (e, _, _) in results_lexical]))
        )
        first_lookup_results.extend([e for (e, _, _) in results_lexical])
        print("FIRST PASS LEXICAL RESULTS:", len(first_lookup_results))
        
        

    # if web_query:
    #     # TODO: Check that this works later when implementing the website search.
    #     lookup_sql_condition = or_(
    #         lookup_sql_condition,
    #         (DocumentEmbedding.website_url != None)
    #     )

    if not use_embeddings:
        first_pass_results = first_lookup_results
    else:
        selection = select(DocumentEmbedding).where(lookup_sql_condition) \
                                    .order_by(
                                        ordering
                                    ) \
                                    .limit(first_pass_k_embedding)
        
        first_pass_results : List[DocumentEmbedding] = database.exec(selection)
        
        first_pass_results : List[DocumentEmbeddingDictionary] = list(map(
            lambda x: DocumentEmbeddingDictionary(**{
                key : v \
                for key, v in x.__dict__.items() \
                if key in DocumentEmbeddingDictionary.__fields__.keys()
            }),
            first_pass_results
        ))
        
        print("FIRST PASS EMBEDDING RESULTS:", len(first_pass_results))
    
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
            key : v for key, v in doc.__dict__.items() if key not in ["_sa_instance_state", "embedding"]
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
        (rerank_question, doc["text"]) 
    
    for doc in new_documents if len(doc["text"]) > 0])
    
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
            *(DocumentEmbedding.parent_collection_hash_id == collection_hash_id for collection_hash_id in collection_hash_ids)
        )
    else:
        
        lookup_sql_condition = (DocumentEmbedding.parent_collection_hash_id == collection_hash_ids[0])

    selection = select(DocumentEmbedding).where(lookup_sql_condition).limit(k)
    
    first_pass_results : List[DocumentEmbedding] = database.exec(selection)
    
    new_docs_dict = {} # Remove duplicates
    for i, doc in enumerate(first_pass_results):
        # print(doc)
        content_hash = hash_function(hash_function(doc.text)+hash_function(doc.document_integrity))
        new_docs_dict[content_hash] = {
            key : v for key, v in doc.__dict__.items() if key not in ["_sa_instance_state", "embedding"]
        }
    
    new_documents = list(new_docs_dict.values())
    
    return new_documents[:min(len(new_documents), k)]

