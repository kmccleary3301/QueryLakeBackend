from sqlmodel import Session, select
from ..database.sql_db_tables import model

import tiktoken

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

def construct_chat_history(max_tokens : int, 
                           token_counter, 
                           sys_instr_pad : str, 
                           usr_entry_pad : str, 
                           bot_response_pad : str, 
                           chat_history: list,
                           system_instruction : str,
                           new_question : str,
                           minimum_context_room : int = 1000):
    """
    Construct model input, trimming from beginning until minimum context room is allowed.
    chat history should be ordered from oldest to newest, and entries in the input list
    should be pairs of the form (user_question, model_response).
    """
    system_instruction_prompt = sys_instr_pad.replace("{system_instruction}", system_instruction)
    sys_token_count = token_counter(system_instruction_prompt)

    new_question_formatted = usr_entry_pad.replace("{question}", new_question)

    chat_history_new, token_counts = [], []
    for entry in chat_history:
        new_entry = usr_entry_pad.replace("{question}", entry[1])+bot_response_pad.replace("{response}", entry[1])
        token_counts.append(token_counter(new_entry))
        chat_history_new.append(new_entry)
        # print("%40s %d" % (new_entry[:40], token_counts[-1]))
    # print(max_tokens)
    token_count_total = sys_token_count + token_counter(new_question_formatted)
    construct_prompt_array = []
    token_counts = token_counts[::-1]
    for i, entry in enumerate(chat_history_new[::-1]):
        token_count_tmp = token_counts[i]
        if (token_count_total+token_count_tmp) > (max_tokens - minimum_context_room):
            break
        token_count_total += token_counts[i]
        construct_prompt_array.append(entry)
    
    final_result = system_instruction_prompt + "".join(construct_prompt_array[::-1]) + new_question_formatted

    # print("FINAL PROMPT")
    # print(final_result)
    return final_result


