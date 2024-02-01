import inspect
from copy import deepcopy

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
            
    if inspect.iscoroutinefunction(function_actual):
        return await function_actual(**new_args)
    else:
        return function_actual(**new_args)

def get_function_args(function):
    """
    Get a list of strings for each argument in a provided function.
    """
    function_args = list(inspect.signature(function).parameters.items())
    function_args = [arg[0] for arg in function_args]
    return function_args