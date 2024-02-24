import inspect
import re
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


if __name__ == "__main__":
    print(get_function_call_preview(get_function_call_preview))