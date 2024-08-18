from sqlmodel import Session, select
from ..database.sql_db_tables import model
from ..typing.config import Padding, Model, ChatHistoryEntry
import tiktoken
from typing import Callable, List, Awaitable

def num_tokens_from_string(string: str, model : str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(string))
    return num_tokens

default_context_wrapper = """Use and cite the following pieces of context to answer requests with accurate information.
Do not cite anything that is not provided here. Do not make up a source, link a source, or write out the identifier of a source.
<context>
{{context}}
</context>"""

default_sys_instr_pad = ""

def add_context(system_instructions, context_segments, context_wrapper=None):
    if len(context_segments) == 0:
        return system_instructions
    if context_wrapper is None:
        context_wrapper = default_context_wrapper
    context_list = ""
    for i, segment in enumerate(context_segments):
        context_list += "%d. %s\n\n" % (i, segment["document"])
    context_wrapper = context_wrapper.replace("{{context}}", context_list)

    return system_instructions + "\n" + context_wrapper

def construct_params(database: Session, model_id):
    if type(model_id) is str:
        model_entry_db = database.exec(select(model).where(model.name == model_id)).first()
        sys_instr_pad = model_entry_db.system_instruction_wrapper
        usr_entry_pad = model_entry_db.user_question_wrapper
        bot_response_pad = model_entry_db.bot_response_wrapper
        context_pad = model_entry_db.context_wrapper
        return sys_instr_pad, usr_entry_pad, bot_response_pad, context_pad
    else:
        sys_instr_pad = "<<INSTRUCTIONS>>\n{system_instruction}\n<</INSTRUCTIONS>>\n\n"
        usr_entry_pad = "<<USER_QUESTION>>\n{question}\n<</USER_QUESTION>>\n\n<<ASSISTANT>>\n"
        bot_response_pad = "{response}\n<</ASSISTANT>>\n\n"
        context_pad = default_context_wrapper
        return sys_instr_pad, usr_entry_pad, bot_response_pad, context_pad

def construct_chat_history_old(max_tokens : int, 
                               token_counter : Callable[[str], int], 
                               sys_instr_pad : str, 
                               usr_entry_pad : str, 
                               bot_response_pad : str, 
                               chat_history: List[ChatHistoryEntry],
                               minimum_context_room : int = 1000,
                               return_chat_history : bool = False) -> str:
    """
    Construct model input, trimming from beginning until minimum context room is allowed.
    chat history should be ordered from oldest to newest, and entries in the input list
    should be pairs of the form (user_question, model_response).
    """
    system_instruction_prompt = sys_instr_pad.replace("{system_instruction}", chat_history[0].content)
    sys_token_count = token_counter(system_instruction_prompt)    
    
    chat_history_new, chat_history_new_formated, token_counts = [], [], []
    # for i, entry in enumerate(chat_history[1:]):
    # print("CHAT HISTORY", chat_history)
    
    
    print("Wrapping chat entries:", [i for i in range(len(chat_history), 0, -2)])
    
    begin_prefix = bot_response_pad.split("{response}")[0]
    prefix_tokens = token_counter(begin_prefix)
    
    for i in range(1, len(chat_history), 2):
        if i == len(chat_history) - 1: # Last entry is a user question
            new_entry = usr_entry_pad.replace("{question}", chat_history[i].content)
            new_entry_formatted = {"role": "user", "content": chat_history[i].content}
        else: 
            new_entry = usr_entry_pad.replace("{question}", chat_history[i].content)+bot_response_pad.replace("{response}", chat_history[i+1].content)
            new_entry_formatted = [{"role": "user", "content": chat_history[i].content}, {"role": "assistant", "content": chat_history[i+1].content}]
        
        token_counts.append(token_counter(new_entry))
        chat_history_new.append(new_entry)
        chat_history_new_formated.append(new_entry_formatted)
    
    
    token_count_total = sys_token_count
    construct_prompt_array, construct_chat_history_array = [], []
    
    token_counts = token_counts[::-1]
    chat_history_new_formated = chat_history_new_formated[::-1]
    
    for i, entry in enumerate(chat_history_new[::-1]):
        token_count_tmp = token_counts[i]
        if (token_count_total+token_count_tmp+prefix_tokens) > (max_tokens - minimum_context_room):
            break
        token_count_total += token_counts[i]
        construct_prompt_array.append(entry)
        if isinstance(chat_history_new_formated[i], list):
            construct_chat_history_array.extend(chat_history_new_formated[i][::-1])
        else:
            construct_chat_history_array.append(chat_history_new_formated[i])
    
    final_result = system_instruction_prompt + "".join(construct_prompt_array[::-1]) + begin_prefix
    final_result_formatted = [{"role": "system", "content": chat_history[0].content}] + construct_chat_history_array[::-1]
    
    if return_chat_history:
        return final_result, final_result_formatted
    
    return final_result

def construct_chat_history(model : Model, 
                           token_counter : Callable[[str], int],
                           chat_history : List[dict],
                           minimum_free_token_space : int,
                           return_chat_history : bool = False) -> str:
    
    chat_history : List[ChatHistoryEntry] = [ChatHistoryEntry(**entry) for entry in chat_history]
    if chat_history[0].role != "system":
        chat_history = [ChatHistoryEntry(role="system", content=model.default_system_instruction)] + chat_history
    
    assert all([entry.role != "system" for entry in chat_history[1:]])
    assert all([entry.role == "user" for entry in chat_history[1::2]])
    assert len(chat_history) % 2 == 0 or len(chat_history) == 1
    
    if len(chat_history) > 2:
        assert all([entry.role == "assistant" for entry in chat_history[2::2]])
    return construct_chat_history_old(model.max_model_len, 
                                      token_counter, 
                                      model.padding.system_instruction_wrap, 
                                      model.padding.question_wrap, 
                                      model.padding.response_wrap, 
                                      chat_history,
                                      minimum_free_token_space,
                                      return_chat_history)
    




async def async_construct_chat_history_old(max_tokens : int, 
                                           token_counter : Callable[[str], Awaitable[int]], 
                                           sys_instr_pad : str, 
                                           usr_entry_pad : str, 
                                           bot_response_pad : str, 
                                           chat_history: List[ChatHistoryEntry],
                                           minimum_context_room : int = 1000,
                                           pairwise : bool = True,
                                           preface_format : bool = True) -> str:
    """
    Construct model input, trimming from beginning until minimum context room is allowed.
    chat history should be ordered from oldest to newest, and entries in the input list
    should be pairs of the form (user_question, model_response).
    """
    
    
    system_instruction_prompt = sys_instr_pad.replace("{system_instruction}", chat_history[0].content)
    sys_token_count = await token_counter(system_instruction_prompt)    
    
    chat_history_new, token_counts, roles = [], [], []
    
    begin_prefix = bot_response_pad.split("{response}")[0] if preface_format else ""
    prefix_tokens = await token_counter(begin_prefix) if preface_format else 0
    
    for i in range(1, len(chat_history)):
        if chat_history[i].role == "user": 
            new_entry = usr_entry_pad.replace("{question}", chat_history[i].content)
        elif chat_history[i].role == "assistant": 
            new_entry = bot_response_pad.replace("{response}", chat_history[i].content)
        
        token_counts.append(await token_counter(new_entry))
        chat_history_new.append(new_entry)
        roles.append(chat_history[i].role)
    
    last_role_is_assistant = False
    
    token_counts = token_counts[::-1]
    token_count_total = sys_token_count
    construct_prompt_array = []
    token_counts = token_counts[::-1]
    for i, entry in enumerate(chat_history_new[::-1]):
        token_count_tmp = token_counts[i]
        if (token_count_total+token_count_tmp+prefix_tokens) > (max_tokens - minimum_context_room):
            break
        token_count_total += token_counts[i]
        construct_prompt_array.append(entry)
        last_role_is_assistant = (roles[i] == "assistant")
        
    if pairwise and last_role_is_assistant:
        construct_prompt_array.pop()
    
    final_result = system_instruction_prompt + "".join(construct_prompt_array[::-1]) + begin_prefix
    
    return final_result