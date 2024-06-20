
from typing import Callable, Any, Union, Awaitable, List
from sqlmodel import Session
from ...typing.config import AuthType, ChatHistoryEntry, Config
from ...misc_functions.prompt_construction import async_construct_chat_history_old
from copy import deepcopy
from ..single_user_auth import get_user
from sqlmodel import Session

ISOLATE_QUESTION_PROMPT = """
Below is a chat conversation between a user and an assistant. The user asks a question, and the model responds.

<CHAT_HISTORY>
{chat_history}
</CHAT_HISTORY>

Let's focus on that last question from the user.
Rewrite the last question from the user by prefacing it with the previous context.
The idea is that the rewrite on its own has the exact same meaning as the original question had with the context.
The rewritten question should not refer to context that it does not include (i.e. `as stated in the conversation`).
Do not leave any ambiguity. Include all necessary details to make the question clear without context.
Be as elaborate as necessary, and include more than one sentence if needed.
Remember, you are rewriting the question, not answering it.
Respond **only** with the rephrased question, no prefacing or introduction (i.e. do not start with `Here is your rewritten question`).
"""


async def llm_isolate_question(database : Session,
                               global_config : Config,
                               auth : AuthType,
                               toolchain_function_caller: Callable[[Any], Union[Callable, Awaitable[Callable]]],
                               chat_history: List[dict],
                               model_choice : str = None) -> str:
    """
    Given a chat history with a most recent question, rephrase the question so
    that it is completely clear without context.
    """
    (_, _) = get_user(database, auth)
    
    if model_choice is None:
        model_choice = global_config.default_models.llm
    
    llm_call = toolchain_function_caller("llm")
    
    question_check_prompt = f"""
Respond with only `YES` or `NO` Does the following resemble a question or request? Assume that the statement is from a conversation.

Statement
---------
{chat_history[-1]["content"]}
""".strip()
    
    question_check = await llm_call(
        auth=auth,
        question=question_check_prompt,
        model_parameters={
            "model_choice": model_choice,
        }
    )
    question_check = question_check["output"]
    # print("QUESTION CHECK RESULT:", question_check)
    
    if not 'YES' in question_check:
        return False
    
    
    if len(chat_history) == 1:
        return {"output": chat_history[0]["content"]}
    
    if chat_history[0]["role"] != "system":
        chat_history = [{"role": "system", "content": ""}] + chat_history
    
    chat_history = [ChatHistoryEntry(**entry) for entry in chat_history]
    
    token_count_general = toolchain_function_caller("llm_count_tokens")
    
    async def token_counter(text : str) -> int:
        return await token_count_general(model_choice, text)
    
    
    chat_history_stated = await async_construct_chat_history_old(
        max_tokens=4096,
        token_counter=token_counter,
        sys_instr_pad="",
        usr_entry_pad="### USER\n\n{question}\n\n",
        bot_response_pad="### ASSISTANT\n\n{response}\n\n",
        chat_history=chat_history,
        minimum_context_room=1,
        pairwise=False,
        preface_format=False
    )
    
    
    question = deepcopy(ISOLATE_QUESTION_PROMPT).format(
        chat_history="\n"+chat_history_stated,
        # last_question=chat_history[-1].content
    )

    # print("QUESTION PROMPT:", question)
    # print("QUESTION PROMPT:", [question])
    
    result = await llm_call(
        auth=auth,
        question=question,
        model_parameters={
            "model_choice": model_choice,
        }
    )
    # print("STANDALONE QUESTION:", result["output"])
    
    return result
    