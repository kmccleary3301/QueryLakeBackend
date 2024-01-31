from sqlmodel import Session, select, and_
from ..database import sql_db_tables, encryption
from .user_auth import get_user, get_organization_private_key, get_user_private_key
from serpapi import GoogleSearch
import json, time
from typing import List, Callable, Awaitable, Dict
from ..vector_database.document_parsing import parse_url
from ..vector_database.embeddings import create_website_embeddings, query_database
from ..typing.config import AuthType, getUserType


def set_user_serp_key(database : Session, 
                      auth : AuthType,
                      serp_key : str):
    """
    Sets user SERP key in SQL db.
    SERP API Key is necessary to perform web search via Google's API.
    """
    (user, user_auth) = get_user(database, auth)
    encrypted_serp_key = encryption.ecc_encrypt_string(user.public_key, serp_key)
    user.serp_api_key_encrypted = encrypted_serp_key
    database.commit()
    # return {"success": True}
    return True

def set_organization_serp_key(database : Session, 
                              auth : AuthType,
                              serp_key : str,
                              organization_hash_id : str):
    """
    Sets user SERP key.
    SERP API Key is necessary to perform web search via Google's API.
    """
    (user, user_auth) = get_user(database, auth)
    organization = database.exec(select(sql_db_tables.organization).where(sql_db_tables.organization.hash_id == organization_hash_id)).first()

    memberships = database.exec(select(sql_db_tables.organization_membership).where(and_(sql_db_tables.organization_membership.organization_id == organization.id,
                                                                                sql_db_tables.organization_membership.user_name == user_auth.username))).all()
    assert len(memberships) > 0, "User not authorized with organization"
    assert memberships[0].role in ["owner", "admin", "member"], "User not authorized to set SERP key"

    encrypted_serp_key = encryption.ecc_encrypt_string(organization.public_key, serp_key)

    organization.serp_api_key_encrypted = encrypted_serp_key
    database.commit()
    # return {"success": True}
    return True

def get_serp_key(database : Session, 
                 auth : AuthType,
                 organization_hash_id : str = None):
    """
    Retrieve user or organization SERP API key.
    SERP API Key is necessary to perform web search via Google's API.
    """
    (user, user_auth) = get_user(database, auth)
    
    if not organization_hash_id is None:
        organization_private_key = get_organization_private_key(database, auth, organization_hash_id)["private_key"]
        organization = database.exec(select(sql_db_tables.organization).where(sql_db_tables.organization.hash_id == organization_hash_id)).first()
        organization_serp_key_encrypted = organization.serp_api_key_encrypted
        assert not organization_serp_key_encrypted is None, "Organization SERP key not set"
        serp_key = encryption.ecc_decrypt_string(organization_private_key, organization_serp_key_encrypted)
        return {"success": True, "result": serp_key}

    user_private_key = get_user_private_key(database, auth)["private_key"]
    serp_key_encrypted = user.serp_api_key_encrypted
    assert not serp_key_encrypted is None, "User SERP key not set"
    serp_key = encryption.ecc_decrypt_string(user_private_key, serp_key_encrypted)
    # return {"success": True, "result": serp_key}
    return {"serp_key": serp_key}

def search_google(database : Session, 
                  auth : AuthType,
                  query : str,
                  results : int = 10,
                  serp_key : str = None,
                  organization_hash_id : str = None):
    """
    Perform a google search request via SERPAPI.
    """
    (user, user_auth) = get_user(database, auth)
    # assert results < 20, "Too many results requested"
    query = query.strip()

    if serp_key is None:
        # search_pre_existing = database.exec(select(sql_db_tables.web_search).where(sql_db_tables.web_search.query == query)).all()
        # if len(search_pre_existing) > 0:
        #     return {"success": True, "result": json.loads(search_pre_existing[0].result), "new_search": False}
        try:
            print("Getting serp_key")
            serp_key = get_serp_key(database, auth, organization_hash_id=organization_hash_id)["serp_key"]
            print("Got key:", serp_key)
        except:
            pass

    print("Search args", [query, serp_key])
    if not (serp_key is None):
        print("D 1 called")
        search = GoogleSearch({
            "q": query, 
            "location": "Baton Rouge, Louisiana",
            "api_key": serp_key
        }).get_dict()
    else:
        print("D 2 called")
        search = None

    search_links = search["organic_results"]
    search_links = sorted(search_links, key=lambda x: x["position"])
    search_links = [link["link"] for link in search_links]

    new_db_entry = sql_db_tables.web_search(
        author_user_name=user_auth.username,
        organization_hash_id=organization_hash_id,
        query=query,
        timestamp=time.time(),
        result=json.dumps(search_links)
    )

    database.add(new_db_entry)
    database.commit()

    # return {"success": True, "result": search_links, "new_search": True}
    return {"result": search_links, "new_search": True}

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
                     text_models_callback : Callable[[Dict, str], Awaitable[List[List[float]]]],
                     auth : AuthType,
                     urls : List[str]):
    """
    Download Urls, convert them to markdown, and
    chunk them into the database.
    """
    (user, user_auth) = get_user(database, auth)
    results = list(map(parse_url, urls))
    pairs = [(urls[i], results[i]) for i in range(len(urls)) if not results[i] is None]
    await create_website_embeddings(database, text_models_callback, pairs)
    # return {"success": True}
    return True

def perform_search_query(database : Session,
                         text_models_callback : Callable[[Dict, str], Awaitable[List[List[float]]]],
                         auth : AuthType,
                         query : str,
                         results : int = 10,
                         organization_hash_id : str = None):
    """
    Perform a search query and embed URLs into database.
    """
    get_links = search_google(database, auth, query, results=results, organization_hash_id=organization_hash_id)
    # get_links = get_links["result"]
    # embed_urls(database, vector_database, username, password_prehash, get_links)
    # return {"success": True, "results": query_database(vector_database, query, [], k=10, use_rerank=True, web_query=True)}
    # return {"success": True, "result": get_links}
    return {"links": get_links}
    






        




