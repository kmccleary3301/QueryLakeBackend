from ..api.api_admin import add_model
import json
from ..database import sql_db_tables
from sqlmodel import Session, select
# from sqlmodel import Session, select, and_

def add_llama2_to_db(database):
    add_model(database,
                loader="exllama",
                model_args={
                    "streaming": True,
                    "model_path": "/home/user/python_projects/langchain/Llama-2-7b-Chat-GPTQ",
                    "lora_path": None,
                    "temperature": 0.3,
                    "typical": 0.7,
                    "verbose": False,
                    "max_seq_len": 4096,
                    "fused_attn": False,
                    "beams": 1, 
                    "beam_length": 40, 
                    "alpha_value": 1.0,
                    "compress_pos_emb": 4.0,
                    "stop_sequences": ["### Input", "### Response", "### Instruction", "Human:", "Assistant", "User:", "AI:"]
                },
                padding={
                    "system_instruction_wrap": "<s>[INST] <<SYS>>\n{system_instruction}\n<</SYS>>\n\n",
                    "context_wrap": "<<context>>{context}<</context>>",
                    "question_wrap": "{question} [/INST] ",
                    "response_wrap": "{response} </s><s> [INST] "
                },
                default_system_instruction="""You are a general answering assistant that can comply with any request. 

You always answer with markdown formatting. You will be penalized if you do not answer with markdown when it would be possible.
The markdown formatting you support: headings, bold, italic, links, tables, lists, code blocks, and blockquotes.
You do not support images and never include images. You will be penalized if you render images.

Always use markdown code blocks for code, and always use MathJax with LaTeX formatting for pure math expressions.
For example code would look like:
```python
*Python code here
```
And math expressions would look like:
$$P(y|x) = \\frac{{P(x|y) \\cdot P(y)}}{{P(x)}}$$
""")
    


def add_models_to_database(database, models):
    for model in models:
        try:
            model_args = model["model_args"]
            necessary_loader = model["loader"]
            # model_name = model_args["model_path"].split("/")[-1]
            padding = model["padding"]
            default_system_instruction = model["default_system_instructions"]

            find_existing_model = database.exec(select(sql_db_tables.model).where(sql_db_tables.model.name == model["name"])).all()
            if len(find_existing_model) > 0:
                continue

            new_model = sql_db_tables.model(
                name=model["name"],
                path_on_server=model_args["model_path"],
                necessary_loader=necessary_loader,
                default_settings=json.dumps(model_args),
                system_instruction_wrapper=padding["system_instruction_wrap"],
                context_wrapper=padding["context_wrap"],
                user_question_wrapper=padding["question_wrap"],
                bot_response_wrapper=padding["response_wrap"],
                default_system_instruction=default_system_instruction
            )
            print("Adding model to db:", new_model.__dict__)

            database.add(new_model)
            database.commit()
        except:
            pass
    
