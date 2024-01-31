import json
from ..database import sql_db_tables
from sqlmodel import Session
from ..typing.config import Model

def add_model(database: Session, model_config : Model) -> bool:
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
    # Check if model already exists
    if database.get(sql_db_tables.model, model_config.name):
        return False
    
    
    new_model = sql_db_tables.model(
        id=model_config.name,
        name=model_config.name,
        path_on_server=model_config.system_path,
        quantization=model_config.quantization,
        default_settings=json.dumps(model_config.default_parameters.dict(), indent=4),
        system_instruction_wrapper=model_config.padding.system_instruction_wrap,
        context_wrapper=model_config.padding.context_wrap,
        user_question_wrapper=model_config.padding.question_wrap,
        bot_response_wrapper=model_config.padding.response_wrap,
        default_system_instruction=model_config.default_system_instructions
    )
    database.add(new_model)
    database.commit()