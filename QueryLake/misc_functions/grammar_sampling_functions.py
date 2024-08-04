

# from ray import serve

from lmformatenforcer.integrations.vllm import VLLMLogitsProcessor
import re
from pydantic import BaseModel
from typing import Optional, Literal, Tuple, List, Union
from lmformatenforcer import JsonSchemaParser

# from lmformatenforcer import CharacterLevelParser
# from lmformatenforcer.integrations.vllm import build_vllm_logits_processor, build_vllm_token_enforcer_tokenizer_data
from .vllm_lmformating_modifed_banned_tokens import build_vllm_logits_processor, build_vllm_token_enforcer_tokenizer_data

from lmformatenforcer.regexparser import RegexParser
# from ray.serve.handle import DeploymentHandle

def ts_to_pydantic(ts_type: str) -> BaseModel:
    """
    Create a pydantic class from a typescript string.
    Potential security risk given that `exec` is used.
    TODO: Revisit this at a later date.
    """
    def parse_literal(literal: str):
        
        if literal[-2:] == "[]":
            return f"List[{parse_literal(literal[:-2])[0]}]", "type"
        
        if literal.lower() == 'true':
            return "True", "value"
        elif literal.lower() == 'false':
            return "False", "value"
        elif literal.lower() == 'null':
            return "None", "value"
        elif literal.lower() == 'string':
            return "str", "type"
        elif literal.lower() == 'number':
            return "float", "type"
        elif literal.lower() == 'boolean':
            return "bool", "type"
        # return f"Literal[{literal}]"
        return literal, "value"

    lines = ts_type.strip().split('\n')
    # class_name = lines[0].split(' ')[1]
    fields = [re.split(r'(?<!\\): ', line.strip()[:-1]) for line in lines[1:-1] if line.strip()]
    pydantic_class = f"class UserSpecifiedClass(BaseModel):\n"
    
    required_fields = []
    for name, type_get in fields:
        default_value = None
        
        if "|" in type_get:
            # test_literal_process = [e.strip() for e in re.split(r'(?<!\\)\|', type_get.strip())]
            # print(test_literal_process)
            # literals = [re.split(r'(?<!\\)\|', literal.strip())[0] for literal in type_get.split("|")]
            literals = [e.strip() for e in re.split(r'(?<!\\)\|', type_get.strip())]
            # print(literals)
            literals_rewrap = [parse_literal(e) for e in literals]
            literals_types = [literal for (literal, classification) in literals_rewrap if classification == "type"]
            literals_values = [literal for (literal, classification) in literals_rewrap if classification == "value"]
            if len(literals_values) > 0:
                literals_values_string = f"Literal[{', '.join(literals_values)}]"
            else:
                literals_values_string = ""
            
            default_value = literals_values[0]
            pre_string = [e for e in literals_types + [literals_values_string] if e.strip() != ""]
            py_type = f"Union[{', '.join(pre_string)}]" if len(pre_string) > 1 else pre_string[0]
        else:
            py_type, _ = parse_literal(type_get)
        
        
        if name.strip()[-1] == "?":
            py_type = f"Optional[{py_type}]"
        else: 
            # py_type = f"Field[{py_type}, default=None]"
            required_fields.append(name.strip())
        name = name.strip().strip("?")
        # pydantic_class += f"    {name}: {py_type}\n"
        
        # for name, type_get in fields:
        # ... existing code ...

        if py_type == 'str':
            default_value = "''"
        elif py_type == 'int':
            default_value = '0'
        elif py_type == 'bool':
            default_value = 'False'
        elif py_type == 'float':
            default_value = '0.0'
        elif default_value is None:
            default_value = 'None'
        # Add more types as needed

        pydantic_class += f"    {name}: {py_type} = {default_value}\n"
    
    # print(pydantic_class)
    exec_globals = {'BaseModel': BaseModel, 'Optional': Optional, 'Literal': Literal, 'List': List, 'Union': Union}
    exec(pydantic_class, None, exec_globals)
    # return exec_globals[class_string.split('\n')[0].split(' ')[1]]
    pre_class = exec_globals["UserSpecifiedClass"]
    
    
    return pre_class()

def get_logits_processor_from_grammar_options(
        grammar_options : Tuple[Literal["regex", "typescript"], str], 
        tokenizer_data,
        # space_tokens : List[int],
        # special_ids : List[int]
    ) -> VLLMLogitsProcessor:
    """
    Create a logits processor from a grammar option provided by the user.
    """
    if grammar_options[0] == "regex":
        # regex_pattern = re.compile(grammar_options[1])
        parser_tmp = RegexParser(grammar_options[1])
        logits_processor = build_vllm_logits_processor(tokenizer_data, parser_tmp, analyze=True)
        return logits_processor
    elif grammar_options[0] == "typescript":
        pydantic_class = ts_to_pydantic(grammar_options[1])
        schema_parser = JsonSchemaParser(pydantic_class.schema(), num_consecutive_whitespaces=0)
        logits_processor = build_vllm_logits_processor(
            tokenizer_data, 
            schema_parser, 
            # banned_tokens=space_tokens
            banned_tokens=[]
        )
        return logits_processor
    
    return None

def get_token_id(tokenizer, token_string : str) -> int:
    newline_id = tokenizer.encode(token_string)
    newline_id = [e for e in newline_id if e not in [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id]]
    return newline_id[-1]
