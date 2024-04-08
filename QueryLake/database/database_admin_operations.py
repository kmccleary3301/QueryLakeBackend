from ..api.api_admin import add_model
import json
from ..database import sql_db_tables
from sqlmodel import Session, select
from ..database import sql_db_tables
from typing import Dict, List
from ..typing.config import Model
from ..typing.toolchains import ToolChain

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

def add_models_to_database(database, models : List[Model]) -> None:
    for model_config in models:
        try:

            find_existing_model = database.exec(select(sql_db_tables.model).where(sql_db_tables.model.id == model_config.id)).all()
            if len(find_existing_model) > 0:
                continue

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
                default_system_instruction=model_config.default_system_instruction
            )
            # print("Adding model to db:", new_model.__dict__)

            database.add(new_model)
            database.commit()
        except:
            pass
    
def add_toolchains_to_database(database : Session,
                               toolchains : Dict[str, ToolChain]) -> None:
    for toolchain_id, toolchain_content in toolchains.items():
        
        find_existing_toolchain = database.exec(select(sql_db_tables.toolchain).where(sql_db_tables.toolchain.toolchain_id == toolchain_id)).all()
        if len(find_existing_toolchain) > 0:
            find_existing_toolchain[0].title = toolchain_content.name
            find_existing_toolchain[0].category = toolchain_content.category
            find_existing_toolchain[0].content = json.dumps(toolchain_content.dict(), indent=4)
            database.commit()
        else:
            new_toolchain = sql_db_tables.toolchain(
                toolchain_id=toolchain_id,
                title=toolchain_content.name,
                category=toolchain_content.category,
                content=json.dumps(toolchain_content.dict(), indent=4)
            )
            database.add(new_toolchain)
        database.commit()
    return