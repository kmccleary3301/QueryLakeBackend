from ray.serve.handle import DeploymentResponseGenerator
from typing import Awaitable, Callable, AsyncGenerator, List, Optional, Union, Literal
import json, inspect
from pydantic import BaseModel
from ..typing.function_calling import FunctionCallDefinition
from ..typing.config import Model
from copy import deepcopy
from .function_run_clean import get_function_call_preview
import re

# function_call_description_template = """
# You may call a function to execute an action.
# Here is an example of a valid function call:

# ```FUNCTION_CALL
# function_name(arg_1=value1, arg_2=value2)
# ```

# It must be formatted with the syntax of a python method call,
# and it must be wrapped as code block with title FUNCTION_CALL.
# The call MUST begin with the sequence "```FUNCTION_CALL" and MUST end with the sequence "```" to be valid.
# Due to parser limitations, you must also always specify passed arguments with a keyword/kwargs. (i.e. don't do `func(\"hi\")`)
# The inner content must be valid python code, and it must be one line.
# You can only call functions, no other code is allowed.

# Here are your available functions:

# {available_functions}
# """

function_call_description_template = """
You may call a python functions to execute an action.
To do so, you must wrap it in the following template:

&& > function_name(arg_1=value1, arg2=value2, ...) &&

It can be any valid python code, as long as it is a function call,
and it is wrapped as && > ... &&.
The call MUST begin with the sequence "&& > " and MUST end with the sequence " &&" to be valid.
Due to parser limitations, you must also always specify passed arguments with a keyword. (i.e. don't do `&& > func(\"hi\") &&`)
The inner content must be valid python code, and it must be one line.

Here are your available functions:

{available_functions}
"""


def find_function_calls(text_in: str):
    # pattern = r"\`\`\`FUNCTION\_CALL\n([\s\S]*?)\`\`\`"
    
    pattern = r"\&\& \> ([^\&]*) \&\&"
    function_calls = []
    for match in re.finditer(pattern, text_in):
        call = match.group(1)
        print(call)
        
        call = [e for e in call.split("\n") if e.strip() != ""][0]
        arguments = {}
        quote_segments = re.finditer(r"[^\\](\".*?[^\\]\"|\'.*?[^\\]\')", call)
        for i, segment in enumerate(list(quote_segments)):
            # This is a hack since JSON sometimes can't handle special characters raw.
            original_string = segment.group(1)
            
            arguments[f"arg_{i}"] = original_string
        
        
        for key, value in arguments.items():
            call = call.replace(key, f"%{key}")
            call = call.replace(value, key)
        
        call = call.replace("=None", "=null")
        
        for match in re.finditer(r"([a-zA-Z_]+)=", call):
            call = call.replace(match.group(0), f"\"{match.group(1)}\":")
            
        for key, value in arguments.items():
            call = call.replace(key, f"\"{value[1:-1]}\"")
            call = call.replace(f"%{key}", key)
        
        call = call.strip()
        
        call_split = re.search(r"^([a-zA-Z_]+)\((.*?)\)?$", call)
        
        assert not call_split is None, f"Failed to parse function call: {{{call}}}, Original call: {text_in}"
        
        try:
            arguments_parsed = json.loads("{" + call_split.group(2) + "}")
            call_result = {
                "function": call_split.group(1),
                "arguments": arguments_parsed
            }
        except json.decoder.JSONDecodeError:
            print("Failed to JSON load:", "{" + call_split.group(2) + "}")
            call_result = {
                "error": "Failed to parse arguments",
                "attempt": f"{{{call_split.group(2)}}}"
            }
        
        function_calls.append(call_result)
        
    return function_calls
     

def construct_functions_available_prompt(functions_available: List[Union[FunctionCallDefinition, dict]]) -> str:
    
    for i in range(len(functions_available)):
        if isinstance(functions_available[i], dict):
            functions_available[i] = FunctionCallDefinition(**functions_available[i])
    
    functions_available_strings = []
    for func in functions_available:
        function_args = []
        for f_arg in func.parameters:
            f_arg.description = f_arg.description.replace('\n', ' ')
            
            function_args.append(f"{f_arg.name}" + \
                (f" : {f_arg.type}" if f_arg.type is not None else "") + \
                (f" : {f_arg.default}" if f_arg.default is not None else "") + \
                (f"\t # {f_arg.description}" if f_arg.description is not None else ""))
            
        function_args = "\n\t".join(function_args)
        functions_available_strings.append(f"def {func.name}(\t\t{function_args}\n)\n\"\"\"\n{func.description}\n\"\"\"")
    
    prompt_make = deepcopy(function_call_description_template).format(
        available_functions="\n\n".join(functions_available_strings)
    )
    
    return prompt_make

async def stream_results_tokens(results_generator: DeploymentResponseGenerator,
                                model_config: Model,
                                encode_output : bool = False,
                                on_new_token: Awaitable[Callable[[str], None]] = None,
                                stop_sequences: List[str] = None,
                                ) -> AsyncGenerator[bytes, None]:
    
    num_returned, tokens_returned, stop_queue, hold_queue = 0, [], [], False
    
    async def new_token_call(text_input):
        nonlocal tokens_returned
        if not on_new_token is None:
            if inspect.iscoroutinefunction(on_new_token):
                await on_new_token(text_input)
            else:
                on_new_token(text_input)
        
        tokens_returned.append(text_input)
    
    def check_stop_sequence(text_in):
        if text_in == "":
            return False
        if stop_sequences is not None:
            for stop_sequence in stop_sequences:
                if stop_sequence.startswith(text_in):
                    return True
        return False
    
    def yield_function(text_in):
        return (json.dumps({"text": text_in}) + "\n").encode("utf-8") if encode_output else text_in
    
    async for request_output in results_generator:
        
        if model_config.engine == "vllm":
            text_outputs = [output.text for output in request_output.outputs]
            assert len(text_outputs) == 1
            text_output = text_outputs[0][num_returned:]
            num_returned = num_returned + len(text_output)
        elif model_config.engine == "exllamav2":
            print("Attempting to grab output from exllamav2")
            text_output = request_output.get("text", "")
        # The following code is responsible for withholding the output if 
        # a stop sequence is being matched. This avoids a partial stop sequence
        # being returned just before termination.
        if stop_sequences is not None:
            if not hold_queue:
                for i in range(len(text_output)):
                    match_stop = check_stop_sequence(text_output[i:])
                    if match_stop:
                        last_valid = text_output[:i]
                        if len(last_valid) > 0:
                            await new_token_call(last_valid)
                            yield yield_function(last_valid)
                        stop_queue.append(text_output[i:])
                        hold_queue = True
                        break
            else:
                stop_queue.append(text_output)
                stop_queue_full = "".join(stop_queue)
                if (any([stop_sequence in stop_queue_full for stop_sequence in stop_sequences])):
                    print("Stopping sequence found:", stop_queue_full)
                    return
                elif (check_stop_sequence(stop_queue_full)):
                    continue
                else:
                    text_output = stop_queue_full
                    hold_queue = False
                    stop_queue = []
        
        
        if not hold_queue:
            await new_token_call(text_output)
            yield yield_function(text_output)
            
            
async def consume_deployment_response(results_generator: DeploymentResponseGenerator) -> List[str]:
    token_list, i = [], 0
    async for result in results_generator:
        token_list.append(result.output)
    if len(token_list) == 1:
        token_list = token_list[0].outputs
    return token_list