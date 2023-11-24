from ..models.model_manager import LLMEnsemble
from .user_auth import get_user
from sqlmodel import Session, select, and_
from ..database import sql_db_tables
from sse_starlette.sse import EventSourceResponse

def llm_call_chat_session(database : Session,
                          llm_ensemble : LLMEnsemble,
                          username : str, 
                          password_prehash : str,
                          history: list,
                          session_hash : str = None,
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
    return EventSourceResponse(llm_ensemble.chain(**filter_args))

def llm_call_model_synchronous(database : Session,
                               llm_ensemble : LLMEnsemble,
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
    return {"success": True, "result": llm_ensemble.single_sync_chat(**filter_args)}
