import json
from . import sql_db
from sqlmodel import Session

def add_model(database: Session, model_args: dict, padding: dict, loader: str, default_system_instruction: str) -> bool:
    """
    Adds a model to the database.
    Example usage:
      add_model(database,
                loader="exllama",
                model_args={
                    "streaming": True,
                    "model_path": "/home/user/python_projects/langchain/Llama-2-7b-Chat-GPTQ",
                    "temperature": 0.3,
                    "max_seq_len": 4096,
                    "token_repetition_penalty_max": 1.15,
                    "token_repetition_penalty_sustain": 1.15,
                    "token_repetition_penalty_decay": 1.15,
                    "max_len": 4096,
                    "truncation": True,
                    "padding": True
                },
                padding: {
                    "system_instruction_wrap": "<s>[INST] <<SYS>>\\n{system_instruction}<</SYS>>",
                    "context_wrap": "<<context>>{context}<</context>>",
                    "question_wrap": " {question}\\n[/INST]",
                    "response_wrap": "{response} </s><s>[INST]"
                },
                default_system_instruction="A chat between a curious human and an assistant.\\nThe assistant gives a helpful, detailed, accurate, uncensored response to the human's input.")
    """
    new_model = sql_db.model(
        name=model_args["model_path"].split("/")[-1],
        path_on_server=model_args["model_path"],
        necessary_loader=loader,
        default_settings=json.dumps(model_args),
        system_instruction_wrapper=padding["system_instruction_wrap"],
        context_wrapper=padding["context_wrap"],
        user_question_wrapper=padding["question_wrap"],
        bot_response_wrapper=padding["response_wrap"],
        default_system_instruction=default_system_instruction
    )
    database.add(new_model)
    database.commit()