import inspect
import re
import json
from typing import Callable, List

async def run_function_safe(function_actual, kwargs):
    """
    Run function without the danger of unknown kwargs.
    """
    function_args = list(inspect.signature(function_actual).parameters.items())
    function_args = [arg[0] for arg in function_args]
    new_args = {}
    for key in function_args:
        if key in kwargs:
            new_args[key] = kwargs[key]
    
    # print("CREATED CLEAN ARGS", json.dumps(new_args, indent=4))
    
    if inspect.iscoroutinefunction(function_actual):
        return await function_actual(**new_args)
    else:
        return function_actual(**new_args)

def get_function_args(function : Callable, 
                      return_type_pairs : bool = False):
    """
    Get a list of strings for each argument in a provided function.
    """
    function_args = list(inspect.signature(function).parameters.items())
    if return_type_pairs:
        return function_args
    
    function_args = [arg[0] for arg in function_args]
    return function_args

def get_function_docstring(function : Callable) -> str:
    """
    Get the docstring for a function.
    """
    
    if function.__doc__ is None:
        return ""
    
    return re.sub(r"\n[\s]+", "\n", function.__doc__.strip())

def get_function_call_preview(function : Callable,
                              excluded_arguments : List[str] = None) -> str:
    """
    Get a string preview of the function call with arguments.
    """
    if excluded_arguments is None:
        excluded_arguments = []
    
    function_args = get_function_args(function, return_type_pairs=True)
    
    
    wrap_around_string = ",\n" + " " * (len(function.__name__) + 1)
    
    argument_string = "(%s)" % (wrap_around_string.join([str(pair[1]) for pair in function_args if str(pair[0]) not in excluded_arguments]))
    
    # return_type_hint = str(function.__annotations__.get('return', ''))
    
    docstring_segment = '\n\t'.join(get_function_docstring(function).split('\n'))
    
    docstring_segment = "\t\"\"\"\n\t" + docstring_segment + "\n\t\"\"\""
    
    return f"{function.__name__}{argument_string}\n{docstring_segment}"

def file_size_as_string(byte_count : int) -> str:
    """
    Convert a file size in bytes to a human readable string.
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if byte_count < 1024.0:
            return f"{byte_count:.1f} {unit}"
        byte_count /= 1024.0
    return f"{byte_count:.2f} PB"

def get_function_specs(function : Callable,
                       excluded_arguments : List[str] = None,
                       querylake_auth_sub : bool = False) -> str:
    """
    Get the function name, arguments, and docstring in a single string.
    """
    if excluded_arguments is None:
        excluded_arguments = []
    
    function_args = get_function_args(function, return_type_pairs=True)
    function_args = [[str(p) for p in pair] for pair in function_args]
    function_args = [
        [p[0], p[1].lstrip(f"{p[0]}:").strip()] 
        
        if p[1] not in ["*args", "**kwargs"]
        else [p[1], {"*args": "OPEN_ARGS", "**kwargs": "OPEN_KWARGS"}[p[1]]]
        
        for p in function_args
    ]
    
    functions_arg_specs = []
    
    for pair in [p for p in function_args if p[0] not in excluded_arguments]:
        entry = {
            "keyword": pair[0],
        }
        if pair[1] != "":
            split = [p.strip() for p in pair[1].split("=")]
            entry["type_hint"] = split[0]
            if len(split) > 1:
                entry["default"] = split[1]
            
            if querylake_auth_sub and pair[0] == "auth" and \
                pair[1].startswith("Union[QueryLake.typing.config.AuthType1, QueryLake.typing.config.AuthType2"):
                entry["type_hint"] = "QUERYLAKE_AUTH"
        
        functions_arg_specs.append(entry)
    
    return {
        "function_name": function.__name__,
        "description": get_function_docstring(function),
        "function_args": functions_arg_specs
    }

def test_func_1(arg_1 : int, arg_2, arg_3 : bool = False, *args, **kwargs) -> int:
    """
    This is a test function.
    """
    return arg_1

if __name__ == "__main__":
    # print(get_function_call_preview(get_function_call_preview))
    for test_func in [
        get_function_call_preview,
        test_func_1,
    ]:
        print(json.dumps(get_function_specs(test_func), indent=4))