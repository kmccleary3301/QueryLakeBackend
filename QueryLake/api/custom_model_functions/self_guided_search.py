from typing import Callable, Any, Union, Awaitable, List, Dict
from sqlmodel import Session
from ...typing.config import AuthType
from ..single_user_auth import get_user
from sqlmodel import Session
import time
from ..search import DocumentChunkDictionary
import json
import inspect


TRAINING_SYSTEM_PROMPT = """
You are tasked with searching for information to answer a question.
You will be given a question, and you must perform searches to find the information necessary to answer it.
DO NOT answer with information not provided by searches.
DO NOT make up information.
ONLY answer with information you have found in the database.
"""

TRAINING_PROMPT_1 = """
You are showing how to do a sample exam to prepare students in a research class.
You will be given a question on a topic.
You must attempt to answer it by performing consecutive searches and retrieving sources until you are ready to answer.
You must perform these searches and make notes of the information as you parse through it until you feel confident that you can answer the question.
When you perform a search or otherwise call a function, you will be met with the result as a response. You may then continue.
Only respond with your next step.
Complete the process in an ideal way by calling provided functions.
Your functions will be search_database() for searching, ready_to_answer() for when you decide to answer, and cannot_answer() for if you feel you couldn't answer effectively.
Take as many searches as you need, and feel free to try different angles to gain insight.. 
Only respond with your reasoning and actions.

Here is your question: {question}

Only respond with your next step as if you were taking the exam.
While searching, provide some brief reasoning or observations before performing the search if helpful.
"""


TRAINING_PROMPT_2 = """
You will be given a question on a topic requiring 1-2 paragraphs to answer
You must attempt to answer it by performing consecutive searches and retrieving sources until you are ready to answer.
You must perform these searches and make notes of the information as you parse through it until you feel confident that you can answer the question.
When you perform a search or otherwise call a function, you will be met with the result as a response. You may then continue.
Only respond with your next step.
Complete the process in an ideal way by calling provided functions.
Your functions will be search_database() for searching, ready_to_answer() to indicate that you are ready to answer, and cannot_answer() for if you feel you couldn't answer effectively.
You are allowed {max_search_string} before answering, no more.
Only respond with your reasoning and actions.

Here is your question: {question}

Only respond with your next step.
"""

answer_enabled_prompt = """
Answering enabled. Go ahead and write your final answer, and nothing else. 
Please cite your sources with frequent inline citations whenever they are even slightly applicable.
Use the citations provided inside the <CITATION> XML of respective sources (i.e. {cite:1}, do not include the XML tags).
Again, citations should be as frequent as possible. While claiming something, 
you should ideally have an appropriate citation at the end of every sentence.
Do not make any claims not supported by the sources provided.
Continue.
"""

could_not_answer_prompt = """
You have indicated that you cannot answer.
Please provide a brief explanation as to why you cannot answer the question.
Also, provide what information you have uncovered. 
Please cite your sources with frequent inline citations whenever they are applicable.
Use the citations provided inside the <CITATION> XML of respective sources (i.e. {cite:1}, do not include the XML tags).
Continue.
"""


search_func_description = """
Perform a search of the database for information.
This is deterministic, so don't repeat the same search.
"""

search_func_def = {
    "name": "search_database",
    "description": search_func_description,
    "parameters": [
        {
            "name": "question",
            "type": "str",
            "description": "Effectively a google search. Will be used to retrieve information from the database via term similarity.",
        }
    ]
}

answer_func_def = {
    "name": "ready_to_answer",
    "description": """
If you feel you have enough information to answer the question, call this to let the system know you are ready to answer.
Only then can you write your answer.
""",
    "parameters": []
}

cannot_answer_func_def = {
    "name": "cannot_answer",
    "description": "If you feel you are unable to answer the question even with the tools available, call this.",
    "parameters": []
}

async def self_guided_search(
    database : Session,
    auth : AuthType,
    toolchain_function_caller: Callable[[Any], Union[Callable, Awaitable[Callable]]],
    question: str,
    collection_ids: List[str] = [],
    model : str = None,
    max_searches : int = 5,
    stream_callables: Dict[str, Awaitable[Callable[[str], None]]] = None
) -> str:
    """
    Self guided search.
    """
    (_, _) = get_user(database, auth)

    start_time = time.time()

    llm_call = toolchain_function_caller("llm")

    search_bm25_function = toolchain_function_caller("search_bm25")
    
    start_time = time.time()

    if stream_callables is None:
        stream_callables = {}

    answer_flag, answer, sources_matched, responses, searches = False, None, [], 0, 0
    ready_to_answer_flag = False
    demo_sequence = []
    max_search_string = f"{max_searches} searches" if max_searches > 1 else "1 search"
    chat_history_1 = [
        {"role": "system", "content": TRAINING_SYSTEM_PROMPT},
        {"role": "user", "content": TRAINING_PROMPT_2.format(question=question, max_search_string=max_search_string)}
    ]

    max_responses, prompt_tokens, output_tokens = 15, 0, 0

    previous_searches, previous_results, all_sources = set(), [], []
    answer_found = False
    
    searches_return = []
    
    
    async def on_new_search(search: str):
        nonlocal searches_return
        searches_return.append({"search": search})
        if "searches" in stream_callables:
            if inspect.iscoroutinefunction(stream_callables["searches"]):
                await stream_callables["searches"]({"search": search})
            else:
                stream_callables["searches"]({"search": search})
                
    async def on_new_source(source_in: dict):
        nonlocal all_sources
        all_sources.append(source_in)
        if "sources" in stream_callables:
            if inspect.iscoroutinefunction(stream_callables["sources"]):
                await stream_callables["sources"](source_in)
            else:
                stream_callables["sources"](source_in)
    
    while True:
        if ready_to_answer_flag:
            answer_flag = True
        
        model_parameters = {
            "model": model,
            "max_tokens": 200 if not answer_flag else 4096,
            "temperature": 0,
            "top_p": 1.0,
            "repetition_penalty": 1.15
        }
        # print("Looping with chat history:", [small_map[chat["role"]] for chat in chat_history_1])
        all_functions_available = [answer_func_def, cannot_answer_func_def]
        last_statement = ""
        if searches < max_searches and not answer_flag:
            all_functions_available.append(search_func_def)
            last_statement = f"\n\nYou have {max_searches - searches} searches remaining."
        elif searches >= max_searches and not answer_flag:
            # answer_flag = True
            last_statement = "\n\n-------ATTENTION------- You have no searches remaining."
        
        chat_history_1[-1]["content"] += last_statement
        

        
        model_response = await llm_call(
            auth=auth,
            chat_history=chat_history_1, 
            model_parameters=model_parameters,
            functions_available=all_functions_available if not answer_flag else [],
            stream_callables=stream_callables if answer_flag and stream_callables else None
        )
        
        prompt_tokens += model_response.get("input_token_count", 0)
        output_tokens += model_response.get("output_token_count", 0)
        

        responses += 1
        chat_history_1.append({"role": "assistant", "content": model_response["output"], "function_calls": model_response.get("function_calls", [])})
        demo_sequence.append({"role": "assistant", "content": model_response["output"]})

        citation_map = {}
        
        if "function_calls" in model_response and len(model_response["function_calls"]) > 0:
            
            if model_response["function_calls"][-1]["function"] == "search_database" and "question" in model_response["function_calls"][-1]["arguments"]:
                new_search = model_response["function_calls"][-1]["arguments"]["question"]
                if new_search in previous_searches:
                    response = {"role": "user", "content": "You have already requested this search. Refer to the results from that attempt. Continue."}
                    chat_history_1.append(response)
                    demo_sequence.append(response)
                    continue
                
                searches += 1
                excluded_chunks = " ".join([f"-id:\"{result}\"" for result in previous_results])

                search_make = model_response["function_calls"][-1]["arguments"]["question"] + f" {excluded_chunks}"
                await on_new_search(f"Searching: \"{new_search}\"")
                
                # print("PREVIOUS RESULTS:", previous_results)
                # print("SEARCH MADE:", search_make)
                searched_sources : List[DocumentChunkDictionary] = search_bm25_function(
                    database=database,
                    auth=auth,
                    query=search_make,
                    collection_ids=collection_ids,
                    limit=5,
                )
                searched_sources = [source.model_dump(exclude_defaults=True) for source in searched_sources]
                
                for source in searched_sources:
                    await on_new_source(source)
                
                for i in range(len(searched_sources)):
                    citation_map[
                        searched_sources[i]["id"] 
                        if isinstance(searched_sources[i]["id"], str) 
                        else searched_sources[i]["id"][0]
                    ] = len(sources_matched) + i + 1
                
                for source in searched_sources:
                    if isinstance(source["id"], list):
                        sources_matched.extend(source["id"])
                    else:
                        sources_matched.append(source["id"])
                        
                    
                
                sources_represented = [
                    "<CITATION>\n\t{cite:" + str(citation_map[
                        source["id"]
                        if isinstance(source["id"], str) 
                        else source["id"][0]
                    ]) + "}\n</CITATION>\n" + \
                    f"<CONTENT>\n{source['text']}\n</CONTENT>\n<METADATA>\n" +
                    json.dumps({
                        k: v for k, v in source.items() 
                        if k not in ["embedding", "collection_id", "collection_type", "creation_timestamp", "text"]    
                    }, indent=4)  + "\n</METADATA>"
                    for source in searched_sources
                ]
                searched_sources_string = "\n\n".join([source_repr for source_repr in sources_represented])
                # print("SEARCHED SOURCES:", searched_sources)
                response = {"role": "user", "content": f"<SEARCH_RESULTS>\n{searched_sources_string}\n</SEARCH_RESULTS>"}
                chat_history_1.append(response)
                demo_sequence.append({**response, "sources": searched_sources})
                previous_searches.add(new_search)
                previous_results.extend([source["id"] for source in searched_sources])
            
            elif model_response["function_calls"][-1]["function"] == "ready_to_answer":

                # print("Got function calls:", model_response["function_calls"])
                # if len(model_response["output"].split("&& > ready_to_answer() &&")[-1]) > 45:
                #     answer = model_response["output"].split("&& > ready_to_answer() &&")[-1]
                #     break
                response = {"role": "user", "content": answer_enabled_prompt}
                chat_history_1.append(response)
                demo_sequence.append(response)
                # print("Getting ready to answer.")
                ready_to_answer_flag = True
                answer_flag = False
                answer_found = True

            elif model_response["function_calls"][-1]["function"] == "cannot_answer":
                # if len(model_response["output"].split("&& > ready_to_answer() &&")[-1]) > 45:
                #     answer = model_response["output"].split("&& > ready_to_answer() &&")[-1]
                #     break
                response = {"role": "user", "content": could_not_answer_prompt}
                chat_history_1.append(response)
                demo_sequence.append(response)
                # print("Getting ready to answer.")
                ready_to_answer_flag = True
                answer_flag = False
        
        if answer_flag:
            answer = model_response["output"]
            break
        
        if chat_history_1[-1]["role"] != "user":
            response = {"role": "user", "content": "No function calls were parsed. Please remember all function calls must be ended with ' &&'. Continue"}
            chat_history_1.append(response)
            demo_sequence.append(response)

        if responses >= max_responses:
            max_cite_index = max([int(cite.split(":")[-1][:-1]) for cite in citation_map.keys()])
            assert max_cite_index == len(sources_matched), f"Max cite index: {max_cite_index}, sources matched: {len(sources_matched)}, citation map: {citation_map}"
            
            return {
                "chat_history": chat_history_1, 
                "output": "Model ran out of responses.", 
                "responses": responses, 
                "time_taken": time.time() - start_time, 
                "sources": [], 
                "answer_found": answer_found,
                "searches": searches_return,
                "input_token_count": prompt_tokens,
                "output_token_count": output_tokens
            }


    # print("RETURNING DEMO SEQUENCE WITH LENGTH", len(demo_sequence))
    return {
        "chat_history": chat_history_1, 
        "output": answer,
        "responses": responses,
        "time_taken": time.time() - start_time,
        "sources": all_sources,
        "answer_found": answer_found,
        "searches": searches_return,
        "input_token_count": prompt_tokens,
        "output_token_count": output_tokens
    }