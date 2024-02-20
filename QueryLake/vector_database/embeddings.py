# from sentence_transformers import SentenceTransformer
# from sentence_transformers import CrossEncoder
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
# from chromadb.api import ClientAPI
from ..database.sql_db_tables import document_raw, DocumentEmbedding
from ..api.hashing import random_hash, hash_function
from typing import List, Callable, Any, Union, Awaitable, Tuple
from numpy import ndarray
import numpy as np
from re import sub
from .document_parsing import parse_PDFs
from sqlmodel import Session, select, and_, or_
from ..typing.config import AuthType
import pgvector
from time import time
from io import BytesIO
import re


# model = SentenceTransformer('BAAI/bge-large-en-v1.5')
# reranker_ce = CrossEncoder('BAAI/bge-reranker-base')
# model_collection_name = "bge-large-en-v1-5"

async def create_embeddings_in_database(database : Session,
                                        toolchain_function_caller: Callable[[Any], Union[Callable, Awaitable[Callable]]],
                                        auth : AuthType,
                                        document_bytes : bytes, 
                                        document_db_entry : document_raw,
                                        document_name : str):
    """
    Add document to chroma db using embeddings.
    """
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
            text=splits_text[i],
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
                         query : str, 
                         collection_hash_ids : List[str], 
                         k : int = 10, 
                         use_rerank : bool = True,
                         web_query : bool = False,
                         minimum_relevance : float = 0.05):
    """
    Create an embedding for a query and lookup a certain amount of relevant chunks.
    """
    # vector_collection = vector_database.get_collection(name=model_collection_name)
    first_pass_k = 100

    chunk_size = 250
    chunk_overlap = 30
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    # splits = text_splitter.split_documents([Document(page_content=query, metadata={"source": "local"})])
    # splits_text = [doc.page_content for doc in splits]
    # query_embeddings = ndarray.tolist(model.encode(splits_text, normalize_embeddings=True))

    
    
    embedding_call : Awaitable[Callable] = toolchain_function_caller("embedding")
    
    query_embeddings = await embedding_call(auth, [query])
    query_embeddings = query_embeddings[0]
    
    # print("Query Embeddings")
    # print(query_embeddings)

    if len(collection_hash_ids) > 1:
        
        lookup_sql_condition = or_(
            *(DocumentEmbedding.parent_collection_hash_id == collection_hash_id for collection_hash_id in collection_hash_ids)
        )
        
        # lookup_sql_condition = {"$or": [{
        #     "parent_collection_hash_id": {
        #         "$eq": collection_hash_id
        #     }
        # } for collection_hash_id in collection_hash_ids]}
    else:
        
        lookup_sql_condition = (DocumentEmbedding.parent_collection_hash_id == collection_hash_ids[0])
        # lookup_sql_condition = {
        #     "parent_collection_hash_id": {
        #         "$eq": collection_hash_ids[0]
        #     }
        # }

    if web_query:
        # lookup_sql_condition = {"$or": [
        #     lookup_sql_condition,
        #     {
        #         "type" : {
        #             "$eq": "website"
        #         }
        #     }
        # ]}
        
        # TODO: Check that this works later when implementing the website search.
        lookup_sql_condition = or_(
            lookup_sql_condition,
            (DocumentEmbedding.website_url != None)
        )
    

    # first_pass_results = vector_collection.query(
    #     query_embeddings=query_embeddings,
    #     n_results=first_pass_k,
    #     where=lookup_sql_condition
    # )
    first_pass_results : List[DocumentEmbedding] = database.exec(
        select(DocumentEmbedding).where(lookup_sql_condition)
                                 .order_by(
                                    DocumentEmbedding.embedding
                                                     .cosine_distance(query_embeddings)
                                  )
                                 .limit(first_pass_k if use_rerank else k)
    )
    

    # print(first_pass_results)
    # new_documents = [{
    #     "document": first_pass_results["documents"][0][i],
    #     "metadata": first_pass_results["metadatas"][0][i]
    # } for i in range(len(first_pass_results["documents"][0]))]
    new_docs_dict = {} # Remove duplicates
    for i, doc in enumerate(first_pass_results):
        # print(doc)
        content_hash = hash_function(hash_function(doc.text)+hash_function(doc.document_integrity))
        new_docs_dict[content_hash] = {
            k : v for k, v in doc.__dict__.items() if k not in ["_sa_instance_state", "embedding"]
        }
        # print("LOOKUP DOC", new_docs_dict[content_hash])
    
    new_documents = list(new_docs_dict.values())
    

    if not use_rerank:
        return new_documents[:min(len(new_documents), k)]

    # print("Rerank Query")
    # print([(query, doc["document"]) for doc in new_documents if len(doc["document"]) > 0])
    
    rerank_call : Awaitable[Callable] = toolchain_function_caller("rerank")
    
    rerank_scores = await rerank_call(auth,
                                      [(query, doc["text"]) for doc in new_documents if len(doc["text"]) > 0])
    
    # print("Rerank Scores", rerank_scores)
    
    
    for i in range(len(new_documents)):
        new_documents[i]["rerank_score"] = rerank_scores[i]
    
    reranked_pairs = sorted(new_documents, key=lambda x: x["rerank_score"], reverse=True)

    
    reranked_pairs = [doc for doc in reranked_pairs if doc["rerank_score"] > minimum_relevance]
    
    
    return reranked_pairs[:min(len(reranked_pairs), k)]




