
from typing import Callable, Any, Union, Awaitable, List
from sqlmodel import Session
from ...typing.config import AuthType, ChatHistoryEntry, Config
from ...misc_functions.prompt_construction import async_construct_chat_history_old
from copy import deepcopy
from ..single_user_auth import get_user
from sqlmodel import Session
from ...vector_database.embeddings import expand_source
from ...api.search import search_hybrid
from .standalone_question import llm_isolate_question
from ...database.sql_db_tables import DocumentEmbeddingDictionary
import re
from ..web_search import web_search
from ..user_auth import get_user_external_providers_dict

search_func_def = {
    "name": "search_database",
    "description": "Perform a search of the database for information.",
    "parameters": [
        {
            "name": "query",
            "type": "str",
            "description": "A topical search. Will be used to retrieve information from the database via term similarity. It's more effective to look up terms similar to a google search than an actual direct question.",
        },
        {
            "name": "question",
            "type": "str",
            "description": "After retrieval using the query, results will be ranked by relevance to the true question using a reranker AI model. This is the question that will be used for reranking.",
        }
    ]
}

expand_source_func_def = {
    "name": "expand_sources",
    "description": "Expand a given source to include the text before and after the current content.",
    "parameters": [
        {
            "name": "sources",
            "type": "List[int]",
            "description": "If you have sources listed as 'SOURCE_1', 'SOURCE_2', etc., you can expand them by providing a list of the source numbers. For example, 'SOURCE_1' would map to 1.",
        }
    ]
}

delete_sources_func_def = {
    "name": "delete_sources",
    "description": "Delete sources from the current working set of information to make room for new sources.",
    "parameters": [
        {
            "name": "sources",
            "type": "List[int]",
            "description": "If you have sources listed as 'SOURCE_1', 'SOURCE_2', etc., you can delete them by providing a list of the source numbers. For example, 'SOURCE_1' would map to 1.",
        }
    ]
}

add_note_func_def = {
    "name": "add_note",
    "description": "If you've observed some type of pattern or information that could be useful for future searches relating to the current query, you can add a note.",
    "parameters": [
        {
            "name": "note",
            "type": "str",
            "description": "Write your note down here. It will be stored for future reference.",
        }
    ]
}

exit_func_def = {
    "name": "exit",
    "description": "Exit the current search and return the current working set of information.",
    "parameters": []
}



MULTI_STEP_SEARCH_PROMPT = """
Your task is to compile information relevant to the question stated below.
You are not to answer the question.

The question is as follows:
<PRIMARY_QUESTION>
{full_question}
</PRIMARY_QUESTION>

Below is your current working set of information and sources.
<CURRENT_INFORMATION>
{current_information}   
</CURRENT_INFORMATION>

Here are your previous commands that have been performed on this question.
<PREVIOUS_COMMANDS>
{previous_searches}
</PREVIOUS_COMMANDS>

And here are the notes that have been made following previous commands.
<NOTES_SECTION>
{notes}
</NOTES_SECTION>

Your goal is to modify this information and currate new information to better answer the question.
You cannot have more than 12 sources.

Use the provided function calls to perform your actions.
Do not do anything other than call the provided functions
"""

all_functions_available = [search_func_def, expand_source_func_def, delete_sources_func_def, add_note_func_def, exit_func_def]

async def llm_multistep_search(database : Session,
                               global_config : Config,
                               auth : AuthType,
                               toolchain_function_caller: Callable[[Any], Union[Callable, Awaitable[Callable]]],
                               chat_history: List[dict],
                               collection_ids: List[str] = [],
                               model_choice : str = None,
                               max_searches : int = 5,
                               search_web : bool = False,
                               web_timeout : float = 10) -> str:
    """
    Given a chat history with a most recent question, 
    perform iterative search using the LLM as an agent.
    """
    (_, _) = get_user(database, auth)
    
    if (not search_web) and len(collection_ids) == 0:
        return []
    
    assert max_searches > 0, "You must have at least one search."
    assert max_searches < 20, "You cannot perform more than 20 search steps."
    
    if search_web:
        search_web = "Serper.dev" in get_user_external_providers_dict(database, auth)
    
    if len(collection_ids) == 0 and search_web is False:
        return {"sources": [], "commands": [], "notes": "No collections provided and web search is disabled."}
    
    if model_choice is None:
        model_choice = global_config.default_models.llm
        
    max_actions = max_searches * 4
    llm_call = toolchain_function_caller("llm")
    
    logs = []
    
    standalone_question = await llm_isolate_question(database, 
                                                     global_config, 
                                                     auth, 
                                                     toolchain_function_caller, 
                                                     chat_history, 
                                                     model_choice)
    
    # print("GOT STANDALONE QUESTION:", standalone_question)
    
    if standalone_question is False:
        return {"sources": [], "commands": [], "notes": "Not a question or request."}
    else:
        logs.append({"action": "make_standalone_question", "content": standalone_question})
    
    primary_question : str = standalone_question["output"]
    
    notes, previous_commands, chunk_sizes = [], [], []
    sources_bank : List[DocumentEmbeddingDictionary] = []
    source_ids = []
    action_count, search_count = 0, 0
    
    
    async def search_database(query: str, question: str):
        nonlocal sources_bank, source_ids, chunk_sizes, logs, database
        if search_web:
            await web_search(database, toolchain_function_caller, auth, query, 10, web_timeout=web_timeout)
        
        search_results = await search_hybrid(database,
                                              auth,
                                              toolchain_function_caller,
                                              query,
                                              collection_ids=collection_ids,
                                              use_lexical=True,
                                              use_embeddings=True,
                                              use_web=search_web,
                                              k=5)
        
        logs.append({"action": "search", "content": question})
        new_sources = [source for source in search_results if source["id"] not in source_ids]
        sources_bank += new_sources
        source_ids += [source["id"] for source in new_sources]
        chunk_sizes += [1 for _ in range(len(search_results))]
    
    async def expand_sources(sources):
        nonlocal sources_bank, source_ids, chunk_sizes, logs, database
        for source_number in sources:
            sources_bank[source_number-1] = expand_source(database,
                                                          auth,
                                                          sources_bank[source_number-1]["id"],
                                                          (-1, chunk_sizes[source_number-1]))
    
    async def delete_sources(sources):
        nonlocal sources_bank, source_ids, chunk_sizes, logs, database
        for source_number in sorted(sources, reverse=True):
            sources_bank.pop(source_number-1)
            source_ids.pop(source_number-1)
            chunk_sizes.pop(source_number-1)
            
    async def add_note(note):
        nonlocal notes, logs
        notes.append(note)
    
    
    function_map = {
        "search_database": search_database,
        "expand_sources": expand_sources,
        "delete_sources": delete_sources,
        "add_note": add_note,
    }
            
        
    
    while (action_count < max_actions and search_count < max_searches):
        action_count += 1
        
        sources_formatted = "\n\n".join(["SOURCE_%d\n\n%s" % (i+1, source["text"]) for i, source in enumerate(sources_bank)]).strip() \
            if len(sources_bank) > 0 else "No sources have been provided."
        
        previous_searches_formatted = "\n".join(previous_commands).strip() \
            if len(previous_commands) > 0 else "No previous searches have been made."
        
        notes_formatted = "\n".join(notes).strip() \
            if len(notes) > 0 else "No notes have been provided." 
        
        question_make = deepcopy(MULTI_STEP_SEARCH_PROMPT).format(
            full_question=primary_question,
            current_information=sources_formatted,
            previous_searches=previous_searches_formatted,
            notes=notes_formatted
        )
        
        current_response = await llm_call(
            auth=auth,
            chat_history= [
                {"role": "system", "content": "You are a helpful search assistant. You only respond by making function calls using the provided functions."},
                {"role": "user", "content": question_make}
            ],
            functions_available= all_functions_available,
            model_parameters={
                "model_choice": model_choice,
                "max_tokens": 200,
            }
        )
        # current_response = current_response["output"]
        print("GOT MULTISEARCH RESPONSE:", current_response)
        
        if "function_calls" in current_response and len(current_response["function_calls"]) > 0:
            call = current_response["function_calls"][0]
            function_name = call["function"]
            if function_name == "exit":
                break
            await function_map[function_name](**call["arguments"])
            if function_name == "search_database":
                search_count += 1
        else:
            assert False, "No function calls were made by the model."
    
    if len(sources_bank) > 0:
        rerank_scores = await toolchain_function_caller("rerank")(
            auth,
            [(primary_question, source["text"]) for source in sources_bank]
        )
        for i in range(len(sources_bank)):
            sources_bank[i]["rerank_score"] = rerank_scores[i]
    
    sources_bank = sorted(sources_bank, key=lambda x: x["rerank_score"], reverse=True)
    
    return {"sources": sources_bank, "commands": previous_commands, "logs": logs, "notes": notes}
    
    