from typing import Callable, Any, Union, Awaitable, List, Dict
from sqlmodel import Session
from ...typing.config import AuthType
from ..single_user_auth import get_user
from sqlmodel import Session
import time
from ..search import DocumentChunkDictionary
import json
import inspect
from ...runtime.search_budgeting import (
    SearchBudgetCounters,
    effective_max_searches,
    evaluate_stop_conditions,
    record_step_cost,
    resolve_budget_policy,
)


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
    use_hybrid: bool = False,
    use_rerank: int = None,
    stream_callables: Dict[str, Awaitable[Callable[[str], None]]] = None,
    budget_policy: Dict[str, Any] = None,
    tenant_scope: str = None,
    toolchain_id: str = None,
) -> str:
    """
    Self guided search.
    """
    (_, _) = get_user(database, auth)

    start_time = time.time()
    resolved_tenant_scope = tenant_scope
    if not isinstance(resolved_tenant_scope, str) or len(resolved_tenant_scope.strip()) == 0:
        resolved_tenant_scope = collection_ids[0] if len(collection_ids) > 0 else None

    policy = resolve_budget_policy(
        budget_policy,
        defaults={
            "max_searches": int(max_searches),
            "max_reranks": int(max_searches if use_rerank is None else max(1, use_rerank)),
        },
        tenant_scope=resolved_tenant_scope,
        toolchain_id=toolchain_id,
    )
    budget_counters = SearchBudgetCounters()
    cost_accounting: Dict[str, Any] = {}
    confidence_history: List[float] = []
    stop_reason: str | None = None
    invalid_function_calls = 0
    duplicate_search_retries = 0

    llm_call = toolchain_function_caller("llm")

    search_bm25_function = toolchain_function_caller("search_bm25")
    search_hybrid_function = toolchain_function_caller("search_hybrid")
    
    start_time = time.time()

    if stream_callables is None:
        stream_callables = {}

    answer_flag, answer, sources_matched, responses, searches = False, None, [], 0, 0
    ready_to_answer_flag = False
    demo_sequence = []
    max_search_string = (
        f"{int(policy.max_searches)} searches"
        if int(policy.max_searches) > 1
        else "1 search"
    )
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
        budget_counters.depth = responses
        effective_search_cap = effective_max_searches(
            policy=policy,
            counters=budget_counters,
            confidence_history=confidence_history,
        )
        stop_eval = evaluate_stop_conditions(
            policy=policy,
            counters=budget_counters,
            start_time=start_time,
            confidence_history=confidence_history,
        )
        if stop_eval.should_stop and not ready_to_answer_flag and not answer_flag:
            stop_reason = stop_eval.reason
            response = {
                "role": "user",
                "content": f"Stop condition reached ({stop_eval.reason}). Provide your best final answer now using available sources.",
            }
            chat_history_1.append(response)
            demo_sequence.append(response)
            ready_to_answer_flag = True

        if ready_to_answer_flag:
            answer_flag = True
        
        model_parameters = {
            "model": model,
            "max_tokens": 200 if not answer_flag else 4096,
            "temperature": 0,
            "top_p": 1.0,
            "repetition_penalty": 1.15
        }
        if policy.strict_deterministic_mode:
            model_parameters["temperature"] = 0
            model_parameters["top_p"] = 1.0
            model_parameters["repetition_penalty"] = 1.0
            model_parameters["seed"] = 0
        # print("Looping with chat history:", [small_map[chat["role"]] for chat in chat_history_1])
        all_functions_available = [answer_func_def, cannot_answer_func_def]
        last_statement = ""
        if searches < effective_search_cap and not answer_flag:
            all_functions_available.append(search_func_def)
            last_statement = (
                f"\n\nYou have {effective_search_cap - searches} searches remaining"
                f" in the current depth tier."
            )
        elif searches >= effective_search_cap and not answer_flag:
            # answer_flag = True
            if policy.adaptive_depth_enabled:
                last_statement = (
                    "\n\n-------ATTENTION------- You have no searches remaining in the current "
                    f"adaptive tier ({effective_search_cap}/{policy.max_searches})."
                )
            else:
                last_statement = "\n\n-------ATTENTION------- You have no searches remaining."
        
        chat_history_1[-1]["content"] += last_statement
        

        
        model_response = await llm_call(
            auth=auth,
            chat_history=chat_history_1, 
            model_parameters=model_parameters,
            functions_available=all_functions_available if not answer_flag else [],
            stream_callables=stream_callables if answer_flag and stream_callables else None
        )
        
        step_in_tokens = int(model_response.get("input_token_count", 0))
        step_out_tokens = int(model_response.get("output_token_count", 0))
        prompt_tokens += step_in_tokens
        output_tokens += step_out_tokens
        budget_counters.prompt_tokens += step_in_tokens
        budget_counters.completion_tokens += step_out_tokens
        cost_accounting = record_step_cost(
            cost_accounting,
            step="llm_round",
            prompt_tokens=step_in_tokens,
            completion_tokens=step_out_tokens,
            model_calls=1,
        )
        

        responses += 1
        chat_history_1.append({"role": "assistant", "content": model_response["output"], "function_calls": model_response.get("function_calls", [])})
        demo_sequence.append({"role": "assistant", "content": model_response["output"]})

        citation_map = {}
        handled_function_call = False
        invalid_function_call = False
        function_calls = model_response.get("function_calls") or []
        if len(function_calls) > 0:
            call = function_calls[-1]
            call_name = call.get("function") if isinstance(call, dict) else None
            call_args = call.get("arguments", {}) if isinstance(call, dict) else {}
            call_args = call_args if isinstance(call_args, dict) else {}

            if call_name == "search_database":
                new_search = call_args.get("question")
                if not isinstance(new_search, str) or len(new_search.strip()) == 0:
                    invalid_function_call = True
                    response = {
                        "role": "user",
                        "content": "Malformed `search_database` call (missing `question`). Retry with valid arguments.",
                    }
                    chat_history_1.append(response)
                    demo_sequence.append(response)
                elif new_search in previous_searches:
                    duplicate_search_retries += 1
                    if duplicate_search_retries > int(policy.max_duplicate_search_retries):
                        if stop_reason is None:
                            stop_reason = "duplicate_search_loop"
                        response = {
                            "role": "user",
                            "content": "Repeated duplicate searches detected. Provide your best final answer now using existing sources.",
                        }
                        chat_history_1.append(response)
                        demo_sequence.append(response)
                        ready_to_answer_flag = True
                    else:
                        response = {
                            "role": "user",
                            "content": "You have already requested this search. Refer to the prior results and continue.",
                        }
                        chat_history_1.append(response)
                        demo_sequence.append(response)
                    handled_function_call = True
                else:
                    duplicate_search_retries = 0
                    handled_function_call = True
                    searches += 1
                    budget_counters.searches = searches
                    excluded_chunks = " ".join([f"-id:\"{result}\"" for result in previous_results])

                    search_make = new_search + f" {excluded_chunks}"
                    await on_new_search(f"Searching: \"{new_search}\"")

                    if not use_hybrid:
                        searched_sources: List[DocumentChunkDictionary] = search_bm25_function(
                            database=database,
                            auth=auth,
                            query=search_make,
                            collection_ids=collection_ids,
                            limit=5,
                        )
                        searched_sources = [source.model_dump(exclude_defaults=True) for source in searched_sources]
                    else:
                        split_size = 5 if use_rerank is None else use_rerank // 2

                        searched_sources = await search_hybrid_function(
                            database=database,
                            toolchain_function_caller=toolchain_function_caller,
                            auth=auth,
                            query=search_make,
                            collection_ids=collection_ids,
                            limit_bm25=split_size,
                            limit_similarity=split_size,
                            rerank=(True if not use_rerank is None else False),
                        )
                        searched_sources: List[DocumentChunkDictionary] = searched_sources["rows"]
                        searched_sources = searched_sources[:5]
                        if use_rerank is not None:
                            budget_counters.reranks += 1
                    cost_accounting = record_step_cost(
                        cost_accounting,
                        step="search_call",
                        model_calls=0,
                        web_calls=0,
                    )

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
                        ]) + "}\n</CITATION>\n"
                        + f"<CONTENT>\n{source['text']}\n</CONTENT>\n<METADATA>\n"
                        + json.dumps(
                            {
                                k: v for k, v in source.items()
                                if k not in ["embedding", "collection_id", "collection_type", "creation_timestamp", "text"]
                            },
                            indent=4,
                        )
                        + "\n</METADATA>"
                        for source in searched_sources
                    ]
                    searched_sources_string = "\n\n".join([source_repr for source_repr in sources_represented])
                    response = {"role": "user", "content": f"<SEARCH_RESULTS>\n{searched_sources_string}\n</SEARCH_RESULTS>"}
                    chat_history_1.append(response)
                    demo_sequence.append({**response, "sources": searched_sources})
                    previous_searches.add(new_search)
                    previous_results.extend([source["id"] for source in searched_sources])
                    confidence_history.append(
                        min(
                            1.0,
                            float(len(previous_results)) / float(max(1, policy.max_searches * 5)),
                        )
                    )

            elif call_name == "ready_to_answer":
                handled_function_call = True
                response = {"role": "user", "content": answer_enabled_prompt}
                chat_history_1.append(response)
                demo_sequence.append(response)
                ready_to_answer_flag = True
                answer_flag = False
                answer_found = True
                confidence_history.append(1.0)

            elif call_name == "cannot_answer":
                handled_function_call = True
                response = {"role": "user", "content": could_not_answer_prompt}
                chat_history_1.append(response)
                demo_sequence.append(response)
                ready_to_answer_flag = True
                answer_flag = False
                if stop_reason is None:
                    stop_reason = "model_cannot_answer"
            else:
                invalid_function_call = True
                response = {
                    "role": "user",
                    "content": f"Unknown function `{call_name}`. Use one of: search_database, ready_to_answer, cannot_answer.",
                }
                chat_history_1.append(response)
                demo_sequence.append(response)
        elif not answer_flag:
            invalid_function_call = True
            response = {
                "role": "user",
                "content": "No function calls were parsed. Ensure calls use the expected function schema and retry.",
            }
            chat_history_1.append(response)
            demo_sequence.append(response)

        if handled_function_call:
            invalid_function_calls = 0
        elif invalid_function_call and not answer_flag:
            invalid_function_calls += 1
            if invalid_function_calls >= int(policy.max_invalid_function_calls):
                if stop_reason is None:
                    stop_reason = "malformed_function_calls"
                response = {
                    "role": "user",
                    "content": "Too many malformed function calls. Provide your best final answer now using available sources.",
                }
                chat_history_1.append(response)
                demo_sequence.append(response)
                ready_to_answer_flag = True
        
        if answer_flag:
            answer = model_response["output"]
            break
        
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
                "output_token_count": output_tokens,
                "budget_policy": policy.model_dump(),
                "budget_counters": budget_counters.model_dump(),
                "cost_accounting": cost_accounting,
                "stop_reason": "max_responses" if stop_reason is None else stop_reason,
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
        "output_token_count": output_tokens,
        "budget_policy": policy.model_dump(),
        "budget_counters": budget_counters.model_dump(),
        "cost_accounting": cost_accounting,
        "stop_reason": stop_reason,
    }
