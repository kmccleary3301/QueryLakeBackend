
from typing import Callable, Any, Union, Awaitable, List
from sqlmodel import Session
from ...typing.config import AuthType, ChatHistoryEntry, Config
from ...misc_functions.prompt_construction import async_construct_chat_history_old
from copy import deepcopy
from ..single_user_auth import get_user
from sqlmodel import Session
from ...vector_database.embeddings import query_database, expand_source
from .standalone_question import llm_isolate_question
from ...database.sql_db_tables import DocumentEmbeddingDictionary
import re
from ..web_search import web_search
from datetime import datetime

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
You are to respond with a specific term to indicate your next action, and some arguments if applicable.
Below are your options, written as the term you can use to respond and the effect it will have.
Follow the examples as closely as possible.

1.  SEARCH: Perform a search for information on the question. Arguments are required!
    This requires two strings as inputs, a topical search similar to a Google search,
    and a direct question for the search engine. This is because it works in two steps,
    one with a lexical lookup and one that can evaluate the questions with the actual question in mind.
    Example: (SEARCH: "1980s Soviet era video game design", "What was the influence of the Soviet Union on video game design in the 1980s?")
   
2.  EXPAND_SOURCE: If one of the provided sources seems relevant, expand the surrounding context.
    Example: (EXPAND_SOURCE: "SOURCE_1")
    
3.  DELETE_SOURCES: If one of the provided sources seems irrelevant, delete it.
    Example: (DELETE_SOURCES: "SOURCE_2", "SOURCE_3")
    
4.  NOTE: Add a note to the notes section to better guide future searches.
    Example: (NOTE: "Soviet Union's influence on video game design in the 1980s centered on Tetris")
    
5.  EXIT: If you are satisfied with the information you have, respond with EXIT.
    Example: (EXIT)
    
Your response must be formatted exactly like the examples, meaning it must be written as
(ACTION: "ARGUMENT1", "ARGUMENT2", ...) if you are providing arguments.
Try not to include anything else.
"""

async def llm_multistep_search(database : Session,
                               global_config : Config,
                               auth : AuthType,
                               toolchain_function_caller: Callable[[Any], Union[Callable, Awaitable[Callable]]],
                               chat_history: List[dict],
                               collection_ids: List[str],
                               model_choice : str = None,
                               max_searches : int = 5,
                               search_web : bool = False) -> str:
    """
    Given a chat history with a most recent question, 
    perform iterative search using the LLM as an agent.
    """
    (_, _) = get_user(database, auth)
    
    assert max_searches > 0, "You must have at least one search."
    assert max_searches < 20, "You cannot perform more than 20 search steps."
    
    if model_choice is None:
        model_choice = global_config.default_model
        
    max_actions = max_searches * 4
    llm_call = toolchain_function_caller("llm")
    
    standalone_question = await llm_isolate_question(database, 
                                                     global_config, 
                                                     auth, 
                                                     toolchain_function_caller, 
                                                     chat_history, 
                                                     model_choice)
    
    if standalone_question["output"] is False:
        return []
    
    
    primary_question : str = standalone_question["output"]
    
    notes, previous_commands, chunk_sizes = [], [], []
    sources : List[DocumentEmbeddingDictionary] = []
    source_ids = []
    action_count, search_count = 0, 0
    
    while (action_count < max_actions and search_count < max_searches):
        action_count += 1
        
        sources_formatted = "\n\n".join(["SOURCE_%d\n\n%s" % (i+1, source["text"]) for i, source in enumerate(sources)]).strip() \
            if len(sources) > 0 else "No sources have been provided."
        
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
            question=question_make,
            model_parameters={
                "model_choice": model_choice,
            }
        )
        current_response = current_response["output"]
        
        # print("RESPONSE:", current_response)
        
        if "SEARCH" in current_response:
            search_count += 1
            response_call : re.Match = re.search(r'SEARCH\:\s*(\"(.*?)\"\,?\s*)+', current_response)
            if response_call is None:
                previous_commands.append(f"INVALID COMMAND: {current_response}")
                continue
            response_string = response_call.group(0)
            previous_commands.append(response_string)
            queries = [e.group(0).strip("\"") for e in list(re.finditer(r'\"(.*?)\"', response_string))]
            if len(queries) == 1:
                if search_web:
                    await web_search(database, toolchain_function_caller, auth, queries[0], 10)
                
                search_results = await query_database(database,
                                                      auth,
                                                      toolchain_function_caller,
                                                      queries[0],
                                                      collection_ids=collection_ids,
                                                      use_lexical=True,
                                                      use_embeddings=True,
                                                      use_web=True,
                                                      k=5)
            elif len(queries) == 2:
                if search_web:
                    await web_search(database, toolchain_function_caller, auth, queries[1], 10)
                search_results = await query_database(database,
                                                      auth,
                                                      toolchain_function_caller,
                                                      queries[0],
                                                      use_lexical=True,
                                                      use_embeddings=True,
                                                      use_web=True,
                                                      collection_ids=collection_ids,
                                                      k=5,
                                                      use_rerank=True,
                                                      minimum_relevance=1,
                                                      rerank_question=queries[1])
            else:
                previous_commands.append(f"INVALID COMMAND: {current_response}")
                continue
            
            new_sources = [source for source in search_results if source["id"] not in source_ids]
            sources += new_sources
            source_ids += [source["id"] for source in new_sources]
            chunk_sizes += [1 for _ in range(len(search_results))]
        elif "EXPAND_SOURCE" in current_response:
            response_call : re.Match = re.search(r'EXPAND_SOURCE\:\s*(\"(.*?)\"\,?\s*)+', current_response)
            if response_call is None:
                previous_commands.append(f"INVALID COMMAND: {current_response}")
                continue
            response_string = response_call.group(1)
            previous_commands.append(response_call.group(0))
            search_strings = [e.group(0).strip("\"") for e in list(re.finditer(r'\"(.*?)\"', response_string))]
            
            for search_string in search_strings:
                if re.search(r'SOURCE_\d+', search_string):
                    source_number = int(search_string.replace("SOURCE_", "").strip(r"\"|\'|\s"))
                    # print("SOURCE NUMBER:", source_number)
                    sources[source_number-1] = expand_source(database,
                                                            auth,
                                                            sources[source_number-1]["id"],
                                                            (-1, chunk_sizes[source_number-1]))
                    chunk_sizes[source_number-1] += 2
            
        elif "DELETE_SOURCES" in current_response:
            deletions = []
            response_call : re.Match = re.search(r'DELETE_SOURCES\:\s*(\"(.*?)\"\,?\s*)+', current_response)
            if response_call is None:
                previous_commands.append(f"INVALID COMMAND: {current_response}")
                continue
            response_string = response_call.group(0)
            previous_commands.append(response_string)
            search_strings = [e.group(0).strip("\"") for e in list(re.finditer(r'\"(.*?)\"', response_string))]
            for search_string in search_strings:
                if re.search(r'SOURCE_\d+', search_string):
                    source_number = int(search_string.replace("SOURCE_", "").strip(r"\"|\'|\s"))
                    # print("SOURCE NUMBER:", source_number)
                    deletions.push(source_number)
                    
            deletions = sorted(deletions, reverse=True)
            for deletion in deletions:
                sources.pop(deletion-1)
                chunk_sizes.pop(deletion-1)
        
        elif "NOTE" in current_response:
            response_call : re.Match = re.search(r'NOTE\:\s*(\"(.*?)\"\,?\s*)+', current_response)
            if response_call is None:
                previous_commands.append(f"INVALID COMMAND: {current_response}")
                continue
            response_string = response_call.group(0)
            previous_commands.append(response_string)
            search_strings = [e.group(0).strip("\"") for e in list(re.finditer(r'\"(.*?)\"', response_string))]
            if len(search_strings) > 0:
                notes.append(search_strings[0])
                # print(search_strings[0])
            
        elif "EXIT" in current_response:
            return sources
        else:
            previous_commands.append(f"INVALID COMMAND: {current_response}")
        
    return {"sources": sources, "commands": previous_commands, "notes": notes}
    
    