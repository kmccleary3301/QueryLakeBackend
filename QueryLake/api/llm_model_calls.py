# from ..models.model_manager import LLMEnsemble
from .user_auth import get_user
from sqlmodel import Session, select, and_
from ..database import sql_db_tables
from sse_starlette.sse import EventSourceResponse
from ..models.langchain_sse import ThreadedGenerator
from threading import Thread
from ..instruction_templates import google_query_builder
from .document import query_vector_db
from chromadb.api import ClientAPI


async def llm_call_chat_session(database : Session,
                                llm_ensemble,
                                vector_database : ClientAPI,
                                username : str, 
                                password_prehash : str,
                                history: list,
                                session_hash : str = None,
                                parameters : dict = None,
                                model_choice : str = None,
                                context : list = None,
                                organization_hash_id : str = None,
                                provided_generator : ThreadedGenerator = None):
    return llm_call_chat_session_direct(database=database,
                                        llm_ensemble=llm_ensemble,
                                        vector_database=vector_database,
                                        username=username,
                                        password_prehash=password_prehash,
                                        history=history,
                                        session_hash=session_hash,
                                        parameters=parameters,
                                        model_choice=model_choice,
                                        context=context,
                                        organization_hash_id=organization_hash_id,
                                        provided_generator=provided_generator)
    

def llm_call_chat_session_direct(database : Session,
                                llm_ensemble,
                                vector_database : ClientAPI,
                                username : str, 
                                password_prehash : str,
                                history: list,
                                session_hash : str = None,
                                parameters : dict = None,
                                model_choice : str = None,
                                collection_hash_ids : list = None,
                                context : list = None,
                                web_search : bool = False,
                                organization_hash_id : str = None,
                                provided_generator : ThreadedGenerator = None):
    """
    Much like the OpenAI API, takes a sequence of inputs of system instruction, user messages, and model messages.
    Example inputs:
    [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
    ],
    They will be trimmed to fit inside model context before being passed to the model.

    Inputs must meet the following criteria:
        Either the first element is system instructions, or no system instructions at all (default will be set).
        Only one system element.
        Everything that follows must be user, assistant alternating.
        First element (after system element) must be from user.
        Last element must be from user.
    
    Optionally provide context, model choice, and/or organization id.
    Output is a server-sent event.
    """
    # function_args=["username", "database", "session_hash", "query", "parameters", "model_choice", "context", "password_prehash", "organization_hash_id"]
    # context = None
    if not (collection_hash_ids is None or collection_hash_ids == []):
        query = craft_google_query_from_history(database=database,
                                                llm_ensemble=llm_ensemble,
                                                username=username,
                                                password_prehash=password_prehash,
                                                history=history,
                                                parameters=parameters,
                                                model_choice=model_choice,
                                                organization_hash_id=organization_hash_id)
        query = query["query"]
    
        context = query_vector_db(database, 
                                  vector_database=vector_database, 
                                  username=username, 
                                  password_prehash=password_prehash,
                                  use_rerank=True,
                                  collection_hash_ids=collection_hash_ids,
                                  query=query)["result"]

    try:
        lookup_role = {0 : "user", 1 : "assistant"}
        assert type(history) is list
        sum_system = 0
        for entry in history:
            if entry["role"] == "system":
                sum_system += 1
            assert "content" in entry
        assert sum_system <= 1
        for i in range(len(history)-sum_system):
            assert history[i+sum_system]["role"] == lookup_role[i % 2]
    except:
        raise ValueError("History not properly formatted.")
    
    filter_args = {
        "database" : database,
        "username" : username, 
        "password_prehash" : password_prehash,
        "history": history,
        "provided_generator": provided_generator
    }
    optional_args = {
        "parameters" : parameters,
        "model_choice" : model_choice,
        "context" : context,
        "organization_hash_id" : organization_hash_id,
        "session_hash" : session_hash,
    }
    for key, value in optional_args.items():
        if not value is None:
            filter_args.update({ key: value })
    if provided_generator is None:
        print("Returning generator")
        return llm_ensemble.chain(**filter_args)
    else:
        print("Returning provided generator")
        Thread(target=llm_ensemble.chain, kwargs=filter_args).start()
        result = {
            "model_response": provided_generator,
        }
        if not context is None:
            result.update({"sources": context})
        return result

def craft_google_query_from_history(database : Session,
                                    llm_ensemble,
                                    username : str, 
                                    password_prehash : str,
                                    history: list,
                                    parameters : dict = None,
                                    model_choice : str = None,
                                    organization_hash_id : str = None):
    new_question = "\n".join(["Given the previous chat history, craft a lexical query to answer the following question, and do not write anything else except for the query.",
	"Do not search for visual data, such as images or videos, as these will be completely useless.",
	"USER: ", history[-1]["content"]])
    new_history = [{"role": "system", "content": google_query_builder}] + history[:-1]
    new_history.append({"role": "user", "content": new_question})
    query = llm_call_model_synchronous(database=database,
                                        llm_ensemble=llm_ensemble,
                                        username=username,
                                        password_prehash=password_prehash,
                                        history=new_history,
                                        parameters=parameters,
                                        model_choice=model_choice,
                                        organization_hash_id=organization_hash_id)
    query = query["model_response"].replace(r"^[\s]*(Sure)[\!]?[^\n]*\n", "").replace(r"(?i)(sure)", "").replace(r"^[\s]*[\"|\'|\`]*", "").replace(r"[\"|\'|\`]*[\s]*$", "")
    print("Created google query:", query)
    return {"query": query}


def llm_call_model_synchronous(database : Session,
                               llm_ensemble,
                               username : str, 
                               password_prehash : str,
                               history: list,
                               parameters : dict = None,
                               model_choice : str = None,
                               context : list = None,
                               organization_hash_id : str = None):
    """
    Much like the OpenAI API, takes a sequence of inputs of system instruction, user messages, and model messages.
    Example inputs:
    [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
    ],
    They will be trimmed to fit inside model context before being passed to the model.

    Inputs must meet the following criteria:
        Either the first element is system instructions, or no system instructions at all (default will be set).
        Only one system element.
        Everything that follows must be user, assistant alternating.
        First element (after system element) must be from user.
        Last element must be from user.
    
    Optionally provide context, model choice, and/or organization id.
    Output is a server-sent event.
    """
    # function_args=["username", "database", "session_hash", "query", "parameters", "model_choice", "context", "password_prehash", "organization_hash_id"]
    try:
        lookup_role = {0 : "user", 1 : "assistant"}
        assert type(history) is list, "History isn't list"
        sum_system = 0
        for entry in history:
            if entry["role"] == "system":
                sum_system += 1
            assert "content" in entry
        assert sum_system <= 1
        for i in range(len(history)-sum_system):
            assert history[i+sum_system]["role"] == lookup_role[i % 2]
    except:
        raise ValueError("History not properly formatted.")
    
    filter_args = {
        "database" : database,
        "username" : username, 
        "password_prehash" : password_prehash,
        "history": history,
    }
    optional_args = {
        "parameters" : parameters,
        "model_choice" : model_choice,
        "context" : context,
        "organization_hash_id" : organization_hash_id
    }
    for key, value in optional_args.items():
        if not value is None:
            filter_args.update({ key: value })
    # return {"success": True, "result": llm_ensemble.single_sync_chat(**filter_args)}
    return {
        "model_response": llm_ensemble.single_sync_chat(**filter_args)
    }
