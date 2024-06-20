
from typing import Callable, Any, Union, Awaitable, List, Dict
from sqlmodel import Session
from ...typing.config import AuthType, ChatHistoryEntry, Config
from ...misc_functions.prompt_construction import async_construct_chat_history_old

CONVERSATION_TITLE_PROMPT = """
Below is a chat conversation between a user and an assistant. The user asks a question, and the model responds.

<CHAT_HISTORY>
{chat_history}
</CHAT_HISTORY>

Write a very short (5 words or less) title/header for this conversation.
Avoid the words 'Chat' and 'Conversation', or anything similarly redundant. It's just the topic.
Don't stylize your response with things like astericks or quotes. Just the title.
Respond **only** with the new title, no prefacing or introduction (i.e. do not start with `Here is your title`).
"""
# If you think it would add to the title, start the title with an appropriate emoji.


async def llm_make_conversation_title(global_config : Config,
                                      auth : AuthType,
                                      toolchain_function_caller: Callable[[Any], Union[Callable, Awaitable[Callable]]],
                                      chat_history: List[dict],
                                      stream_callables: Dict[str, Awaitable[Callable[[str], None]]] = None,
                                      model_choice : str = None) -> str:
    """
    Given a chat history with a most recent question, rephrase the question so
    that it is completely clear without context.
    """
    
    if model_choice is None:
        model_choice = global_config.default_models.llm
    
    llm_call = toolchain_function_caller("llm")
    
    if len(chat_history) == 1:
        return chat_history[0]["content"]
    
    if chat_history[0]["role"] != "system":
        chat_history = [{"role": "system", "content": ""}] + chat_history
    
    chat_history : List[ChatHistoryEntry] = [ChatHistoryEntry(**entry) for entry in chat_history]
    
    token_count_general = toolchain_function_caller("llm_count_tokens")
    
    async def token_counter(text : str) -> int:
        return await token_count_general(model_choice, text)
    
    chat_history_stated = await async_construct_chat_history_old(
        max_tokens=6144,
        token_counter=token_counter,
        sys_instr_pad="",
        usr_entry_pad="### USER\n\n{question}\n\n",
        bot_response_pad="### ASSISTANT\n\n{response}\n\n",
        chat_history=chat_history,
        minimum_context_room=1
    )
    
    question = CONVERSATION_TITLE_PROMPT.format(
        chat_history="\t"+chat_history_stated.replace("\n", "\n\t"),
        last_question=chat_history[-1].content
    )

    result = await llm_call(
        auth=auth,
        question=question,
        model_parameters={
            "model_choice": model_choice,
        },
        stream_callables=stream_callables
    )
    
    return result
    