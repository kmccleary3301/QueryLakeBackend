from sqlmodel import Session, select, and_
from ..database import sql_db_tables, encryption
from .hashing import random_hash
from .user_auth import get_user, get_organization_private_key, get_user_private_key
from serpapi import GoogleSearch
import json, time, requests
from typing import List, Callable, Awaitable, Dict, Tuple, Any, Union
from ..vector_database.document_parsing import parse_url
from ..vector_database.embeddings import create_website_embeddings, create_text_embeddings
from ..typing.config import AuthType, getUserType
from io import BytesIO
from hashlib import sha256
from asyncio import gather
from .user_auth import get_user_external_providers_dict

def parse_urls(database : Session, 
               auth : AuthType,
               urls : List[str]):
    """
    Embed URLs into the database.
    """
    (user, user_auth) = get_user(database, auth)
    results = list(map(parse_url, urls))
    for result in results:
        print()
        print(result[:20])
    # return {"success": True, "result": results}
    return {"result": results}

async def embed_urls(database : Session,
                     toolchain_function_caller: Callable[[Any], Union[Callable, Awaitable[Callable]]],
                     auth : AuthType,
                     urls : List[str],
                     titles : List[str] = None):
    """
    Download Urls, convert them to markdown, and
    chunk them into the database.
    """
    (_, user_auth) = get_user(database, auth)
    # results = list(map(parse_url, urls))
    
    web_scrape_call = toolchain_function_caller("web_scrape")
    
    results = await web_scrape_call(auth, urls)
    
    document_entries : List[Tuple[sql_db_tables.document_raw, str]] = []
    embedding_coroutines = []
    
    for i, result_dictionary in enumerate(results):
        url = urls[i]
        if result_dictionary is None:
            print(f"Failed to scrape {url}")
            continue
        
        content = result_dictionary["text"]
        if len(content) > (600 * 400):
            continue
        
        website_content_bytes : bytes = content.encode("utf-8")
        new_document = sql_db_tables.document_raw(
            hash_id=random_hash(),
            file_name=result_dictionary["metadata"]["title"],
            author_user_name=user_auth.username,
            organization_hash_id=None,
            creation_timestamp=time.time(),
            integrity_sha256=sha256(website_content_bytes).hexdigest(),
            size_bytes=len(website_content_bytes),
            website_url=url,
            file_data=website_content_bytes,
            finished_processing=False
        )
        database.add(new_document)
        document_entries.append((new_document, content))
        embedding_coroutines.append(
            create_text_embeddings(database, 
                                   auth, 
                                   toolchain_function_caller,
                                   [(content, {"page": 0})],
                                   new_document,
                                   new_document.file_name)
        )
    database.commit()
    
    await gather(*embedding_coroutines)
    
    
    print("Finished web url embeddings")
    # return {"success": True}
    return {"content": results}

async def web_search(database : Session,
                     toolchain_function_caller: Callable[[Any], Union[Callable, Awaitable[Callable]]],
                     auth : AuthType,
                     query : str,
                     results : int = 10):
    """
    Perform a search query and embed URLs into database.
    """
    (_, _) = get_user(database, auth)
    
    assert results <= 100, "Too many results requested"
    
    user_providers_dict = get_user_external_providers_dict(database, auth)
    assert "Serper.dev" in user_providers_dict, "User does not have SERP API key set"
    serp_key = user_providers_dict["Serper.dev"]
    url = "https://google.serper.dev/search"

    payload = json.dumps({
        "q": query,
        "num": results
    })
    headers = {
        'X-API-KEY': serp_key,
        'Content-Type': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    result = response.json().get("organic", [])
    await embed_urls(database, toolchain_function_caller, auth, [e["link"] for e in result], [e["title"] for e in result])
    
    return True
    






        




