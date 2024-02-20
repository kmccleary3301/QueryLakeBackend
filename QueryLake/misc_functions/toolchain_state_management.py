import os, json
from copy import deepcopy, copy
from typing import Callable, Any, List, Dict, Union, Awaitable
from ..typing.toolchains import *
from io import BytesIO

getFilesCallableType = Callable[[ToolChainSessionFile], Union[bytes, BytesIO, str]]

def append_in_route(object_for_static_route : Union[list, dict], route : List[Union[str, int]], value : Any) -> Union[list, dict]:
    """
    Insert a value into an object using a list of strings and ints.
    """
    
    if len(route) > 0:
        object_for_static_route[route[0]] = append_in_route(object_for_static_route[route[0]], route[1:], value)
    else:
        if isinstance(object_for_static_route, list):
            object_for_static_route.append(value)
        else:
            object_for_static_route += value
    
    return object_for_static_route

def retrieve_value_from_obj(input_dict : Union[list, dict], 
                            directory : Union[str, List[str]]):
    try:
        if isinstance(directory, str):
            return input_dict[directory]
        else:
            current_dict = input_dict
            for entry in directory:
                current_dict = current_dict[entry]
            return current_dict
    except KeyError:
        raise KeyError
    except:
        return None

def dict_path_is_valid(input_dict : dict, directory : Union[str, List[str]]):
    try:
        if isinstance(directory, str):
            return input_dict[directory]
        else:
            current_dict = input_dict
            for entry in directory:
                current_dict = current_dict[entry]
            return True
    except:
        return False

def safe_serialize(obj, **kwargs) -> str:
    """
    Serialize an object, but if an element is not serializable, return a string representation of said element.
    """
    def default_callable(o):
        try:
            return o.dict()
        except:
            return f"<<non-serializable: {type(o).__qualname__}>>"
    
    # default = lambda o: f"<<non-serializable: {type(o).__qualname__}>>"
    default = lambda o: default_callable(o)
    return json.dumps(obj, default=default, **kwargs)

def evaluate_static_route(route : staticRoute,
                          toolchain_state : Union[list, dict],
                          node_inputs_state : Union[list, dict],
                          node_outputs_state : Union[list, dict],
                          toolchain_files : Union[list, dict],
                          get_files_callable : getFilesCallableType = None) -> List[Union[str, int]]:
    """
    Convert a staticRoute type into a list of strings and ints.
    """
    
    state_kwargs = {
        "toolchain_state" : toolchain_state,
        "node_inputs_state" : node_inputs_state,
        "node_outputs_state" : node_outputs_state,
        "toolchain_files" : toolchain_files,
        "get_files_callable" : get_files_callable
    }
    
    def check_element(element : staticRouteElementType) -> Union[str, int]:
        if isinstance(element, int) or isinstance(element, str):
            return element
        elif isinstance(element, indexRouteRetrievedNew):
            if isinstance(element, staticValue):
                assert isinstance(element.value, (int, str)), "staticValue used in staticRoute, but value was not int or str"
                return element.value
            elif isinstance(element, indexRouteRetrieved):
                return get_value_obj_global(element.getFrom, **state_kwargs)
            elif isinstance(element, indexRouteRetrievedStateValue):
                return traverse_static_route_global(toolchain_state, element.getFromState.route, **state_kwargs)[0]
            elif isinstance(element, indexRouteRetrievedInputArgValue):
                return traverse_static_route_global(node_inputs_state, element.getFromInputs.route, **state_kwargs)[0]
            elif isinstance(element, indexRouteRetrievedOutputArgValue):
                return traverse_static_route_global(node_outputs_state, element.getFromOutputs.route, **state_kwargs)[0]
            
    # Convert the route to a list of strings and ints using `check_element` via map.
    return list(map(check_element, route))

def traverse_static_route_global(object_for_static_route : Union[list, dict], 
                                 route : staticRoute,
                                 toolchain_state : Union[list, dict],
                                 node_inputs_state : Union[list, dict],
                                 node_outputs_state : Union[list, dict],
                                 toolchain_files : Union[list, dict],
                                 get_files_callable : getFilesCallableType = None
                                 ) -> Tuple[Union[list, dict], List[Union[str, int]]]:
    """
    Traverse an object using a static route.
    The condition tree here reflects the branches of the element types.
    """
    state_kwargs = {
        "toolchain_state" : toolchain_state,
        "node_inputs_state" : node_inputs_state,
        "node_outputs_state" : node_outputs_state,
        "get_files_callable" : get_files_callable
    }
    
    object_static_in_focus = object_for_static_route
    indices = []
    for action in route:
        if isinstance(action, int) or isinstance(action, str): # Finished
            object_static_in_focus = object_static_in_focus[action]
            indices.append(action)
        elif isinstance(action, indexRouteRetrievedNew): # Finished
            # Possible types: staticValue, indexRouteRetrieved, indexRouteRetrievedStateValue, indexRouteRetrievedInputArgValue, indexRouteRetrievedOutputArgValue
            if isinstance(action, staticValue):
                object_in_focus = object_in_focus[action.value]
                indices.append(action.value)
                
            elif isinstance(action, indexRouteRetrieved): # Finished All Possibilities
                # Possible types: staticValue, stateValue, getNodeInput, getNodeOutput
                
                get_value = get_value_obj_global(action.getFrom, **state_kwargs)
                object_static_in_focus = object_static_in_focus[get_value]
                indices.append(get_value)
            
            elif isinstance(action, indexRouteRetrievedStateValue): # Finished
                # Only one subfield (`getFromState`) type: stateValue
                get_value, _ = traverse_static_route_global(toolchain_state, action.getFromState.route, **state_kwargs)
                object_static_in_focus = object_static_in_focus[get_value]
                indices.append(get_value)
            
            elif isinstance(action, indexRouteRetrievedInputArgValue): # Finished
                # Only one subfield (`getFromInputs`) type: getNodeInput
                get_value, _ = traverse_static_route_global(node_inputs_state, action.getFromInputs.route, **state_kwargs)
                object_static_in_focus = object_static_in_focus[get_value]
                indices.append(get_value)
            
            elif isinstance(action, indexRouteRetrievedOutputArgValue): # Finished
                # Only one subfield (`getFromOutput`) type: getNodeOutput
                get_value, _ = traverse_static_route_global(node_outputs_state, action.getFromOutputs.route, **state_kwargs)
                object_static_in_focus = object_static_in_focus[get_value]
                indices.append(get_value)
        
    return object_static_in_focus, indices

def get_value_obj_global(value_obj : Union[valueObj, indexRouteRetrievedNew],
                         toolchain_state : Union[list, dict],
                         node_inputs_state : Union[list, dict],
                         node_outputs_state : Union[list, dict],
                         toolchain_files : Union[list, dict],
                         get_files_callable : getFilesCallableType = None) -> Any:
    """
    For traversing value Objects and static Route elements.
    """
    
    state_kwargs = {
        "toolchain_state" : toolchain_state,
        "node_inputs_state" : node_inputs_state,
        "node_outputs_state" : node_outputs_state,
        "toolchain_files" : toolchain_files,
        "get_files_callable" : get_files_callable
    }
    
    
    if isinstance(value_obj, staticValue):
        return value_obj.value
    elif isinstance(value_obj, getLengthValue):
        get_value = get_value_obj_global(value_obj.getLength, **state_kwargs)
        assert isinstance(get_value, list, str), "getLengthValue used, but the value retrieved was not a list or string"
        return len(get_value)     
    elif isinstance(value_obj, stateValue):
        target_object, target_route = toolchain_state, value_obj.route
    elif isinstance(value_obj, indexRouteRetrievedStateValue):
        target_object, target_route = toolchain_state, value_obj.getFromState.route
    elif isinstance(value_obj, getNodeInput):
        target_object, target_route = node_inputs_state, value_obj.route
    elif isinstance(value_obj, indexRouteRetrievedInputArgValue):
        target_object, target_route = node_inputs_state, value_obj.getFromInputs.route
    elif isinstance(value_obj, getNodeOutput):
        target_object, target_route = node_outputs_state, value_obj.route
    elif isinstance(value_obj, indexRouteRetrievedOutputArgValue):
        target_object, target_route = node_outputs_state, value_obj.getFromOutputs.route
    elif isinstance(value_obj, getFiles):
        pass
    
    get_value, _ = traverse_static_route_global(target_object, target_route, **state_kwargs)
    
    return get_value

def insert_in_static_route_global(object_for_static_route : Union[list, dict], 
                                  route : staticRoute, 
                                  value : Any,
                                  toolchain_state : Union[list, dict],
                                  node_inputs_state : Union[list, dict],
                                  node_outputs_state : Union[list, dict],
                                  toolchain_files : Union[list, dict],
                                  get_files_callable : getFilesCallableType = None,
                                  return_indices : bool = False,
                                  append : bool = False,
                                  route_need_conversion : bool = True) -> Union[list, dict]:
    """
    Insert a value into an object using a static route.
    """
    object_for_static_route = copy(object_for_static_route)
    
    if len(route) == 0:
        assert isinstance(object_for_static_route, dict), "insert_in_static_route called with an empty route and a non-dict object"
        object_for_static_route.update(value)
        return object_for_static_route
    
    state_kwargs = {
        "toolchain_state" : toolchain_state,
        "node_inputs_state" : node_inputs_state,
        "node_outputs_state" : node_outputs_state,
        "toolchain_files" : toolchain_files,
        "get_files_callable" : get_files_callable
    }
    
    if route_need_conversion:
        route_evaluated = evaluate_static_route(route, **state_kwargs)
    else:
        route_evaluated = route
    
    
    if len(route) > 1:
        # Recursively call insert_in_static_route_global.
        
        next_directory, _ = traverse_static_route_global(object_for_static_route, [route_evaluated[0]], **state_kwargs)
        
        
        object_for_static_route[route_evaluated[0]]= insert_in_static_route_global(next_directory, 
                                                                                   route_evaluated[1:], 
                                                                                   value, 
                                                                                   **state_kwargs, 
                                                                                   append=append, 
                                                                                   route_need_conversion=False)
    else:
        # Single route case.
        
        if isinstance(object_for_static_route, list):
            print("LIST", object_for_static_route, route[0])
            
            if route[0] >= len(object_for_static_route):
                assert route[0] == len(object_for_static_route), "When running insert_in_static_route, the route index was greater than the length of the list, overshooting a normal append."
                
                # print("Appending to list", object_for_static_route, value)
                object_for_static_route.append(value)
                
        else:
            route_last = route[0]
            if not isinstance(route_last, (list, str)):
                route_last = get_value_obj_global(route_last, **state_kwargs)
            
            if append:
                # print("Appending to list", [object_for_static_route[route_last], value])
                
                if isinstance(object_for_static_route[route_last], list):
                    object_for_static_route[route_last].append(value)
                else:
                    
                    
                    object_for_static_route[route_last] += value
                    
                # print("Appending to list done", [object_for_static_route[route_last], value])
            else:
                object_for_static_route[route_last] = value
    
    if return_indices:
        return object_for_static_route, route_evaluated
    
    return object_for_static_route

def run_sequence_action_on_object(subject_state : Union[list, dict],
                                  toolchain_state : Union[list, dict],
                                  node_inputs_state : Union[list, dict],
                                  node_outputs_state : Union[list, dict],
                                  toolchain_files : Union[list, dict],
                                  sequence : List[sequenceAction],
                                  provided_object : Any = None,
                                  get_files_callable : getFilesCallableType = None,
                                  deepcopy_object : bool = False,
                                  return_provided_object_routes : bool = False) -> Union[Union[list, dict], Tuple[Union[list, dict], List[Union[int, str]]]]:
    """
    TODO: Think about how we want to efficiently communicate changes here to
    the client. Maybe the client needs to have a copy of it's own state and
    run the sequence actions themselves. This would save data,
    but client performance would be worse.
    
    Run a set of sequence actions on an object.
    Keeping the local functions here, as the states are already kept in scope.
    """
    
    state_kwargs = {
        "toolchain_state" : toolchain_state,
        "node_inputs_state" : node_inputs_state,
        "node_outputs_state" : node_outputs_state,
        "toolchain_files" : toolchain_files,
        "get_files_callable" : get_files_callable
    }
    
    routes_for_provided_object = []
    
    if deepcopy_object:
        object = deepcopy(subject_state)
    else:
        object = subject_state.copy()
    
    object_in_focus = object
    current_indices = []
    
    for action in sequence:
        # Possible types : staticRouteElementType, createAction, deleteAction, updateAction, appendAction, operatorAction, backOut
        # TODO: Thoroughly review the logic here.

        if not isinstance(action, staticRouteElementType) and not action.condition is None:
            condition_evaluated = evaluate_condition(condition=action.condition, provided_object=provided_object, **state_kwargs)
            if not condition_evaluated:
                continue
        
        if isinstance(action, staticRouteElementType):
            object_in_focus, new_indices = traverse_static_route_global(object_in_focus, [action], **state_kwargs)
            current_indices += new_indices
        
        elif isinstance(action, createAction):
            
            # TODO: Thoughly review here to make sure the routes retrieved are always correct.
            
            if not action.initialValue is None:
                initial_created_obj, use_provided = get_value_obj_global(action.initialValue, **state_kwargs), False
            else:
                initial_created_obj, use_provided = provided_object, True
            
            insertion_routes_of_action = evaluate_static_route(action.route, **state_kwargs)
            
            
            for s_list_i, insertion_route in enumerate(action.insertions):
                passed_value = get_value_obj_global(action.insertion_values[s_list_i], **state_kwargs) if not action.insertion_values[s_list_i] is None else provided_object
                initial_created_obj, insertion_routes_in_created_obj = insert_in_static_route_global(initial_created_obj, 
                                                                                                     insertion_route, 
                                                                                                     passed_value, 
                                                                                                     **state_kwargs, 
                                                                                                     return_indices=True)
                if action.insertion_values[s_list_i] is None:
                    routes_for_provided_object.append(current_indices + insertion_routes_of_action + insertion_routes_in_created_obj)
            
            
            # Insert our created object into the current directory of the object.
            object_in_focus = insert_in_static_route_global(object_in_focus, insertion_routes_of_action, initial_created_obj, **state_kwargs, route_need_conversion=False)
            
            # If we didn't construct an object first, simply add the action route as an insertion route.
            if use_provided:
                routes_for_provided_object.append(current_indices + insertion_routes_of_action)
            
            # Update the global object.
            object = insert_in_static_route_global(object, current_indices, object_in_focus, **state_kwargs)
        
        elif isinstance(action, appendAction): # Same as createAction, but appends to a list at the route.
            
            # TODO: Thoughly review here to make sure the routes retrieved are always correct.
            
            if not action.initialValue is None:
                initial_created_obj, use_provided = get_value_obj_global(action.initialValue, **state_kwargs), False
            else:
                initial_created_obj, use_provided = provided_object, True
            
            # print("CALLING APPEND", action.dict())
            
            insertion_routes_of_action = evaluate_static_route(action.route, **state_kwargs)
            
            
            # Pre-navigate to the insertion route.
            object_to_append_to, _ = traverse_static_route_global(object_in_focus, insertion_routes_of_action, **state_kwargs)
            
            
            if len(action.insertions) > 0:
                assert (isinstance(object_to_append_to, list)), "appendAction used with insertions, but the object in focus wasn't a list"
            
            for s_list_i, insertion_route in enumerate(action.insertions):
                # print("APPEND INSERTION ROUTE", insertion_route)
                
                passed_value = get_value_obj_global(action.insertion_values[s_list_i], **state_kwargs) if not action.insertion_values[s_list_i] is None else provided_object
                initial_created_obj, insertion_routes_in_created_obj= insert_in_static_route_global(initial_created_obj, insertion_route, passed_value, **state_kwargs, return_indices=True)
                
                # print("GOT TMP ROUTES IN APPEND FROM INSERTION", insertion_routes_in_created_obj)
                
                if action.insertion_values[s_list_i] is None:
                    
                    routes_for_provided_object.append(current_indices + insertion_routes_of_action + [len(object_to_append_to)] + insertion_routes_in_created_obj)
                
            
            assert isinstance(object_to_append_to, list) or (isinstance(object_to_append_to, str) and isinstance(initial_created_obj, str)), "appendAction used, but the object in focus was not a list or string"
            object_to_append_to.append(initial_created_obj)
            
            
            
            
            object_in_focus, insertion_routes_in_current_dir = insert_in_static_route_global(object_in_focus, insertion_routes_of_action, object_to_append_to, **state_kwargs, return_indices=True)
            
            if use_provided:
                
                last_sequence = [len(object_to_append_to)-1] if isinstance(object_to_append_to, list) else [] # Empty if appending to a string.
                routes_for_provided_object.append(current_indices + insertion_routes_in_current_dir + last_sequence)
            
            object = insert_in_static_route_global(object, current_indices, object_in_focus, **state_kwargs)
        
        elif isinstance(action, deleteAction):
            if not action.route is None:
                route_get = action.route
                
                object_to_append_to, tmp_indices = traverse_static_route_global(object_in_focus, route_get[:-1], **state_kwargs)
                del object_to_append_to[route_get[-1]]
                object_in_focus = insert_in_static_route_global(object_in_focus, tmp_indices, object_to_append_to, **state_kwargs)
            
            if not action.routes is None:
                for route_get in action.routes:
                    object_to_append_to, tmp_indices = traverse_static_route_global(object_in_focus, route_get[:-1], **state_kwargs)
                    del object_to_append_to[route_get[-1]]
                    object_in_focus = insert_in_static_route_global(object_in_focus, tmp_indices, object_to_append_to, **state_kwargs)

            object = insert_in_static_route_global(object, current_indices, object_in_focus, **state_kwargs)
                    
        elif isinstance(action, updateAction):
            if not action.value is None:
                initial_created_obj, use_provided = get_value_obj_global(action.value, **state_kwargs), False
            else:
                initial_created_obj, use_provided = provided_object, True
            
            object_in_focus, insertion_routes_tmp = insert_in_static_route_global(object_in_focus, action.route, initial_created_obj, **state_kwargs, return_indices=True)
            object = insert_in_static_route_global(object, current_indices, object_in_focus, **state_kwargs)
            if use_provided:
                routes_for_provided_object.append(current_indices + insertion_routes_tmp)
            
        elif isinstance(action, operatorAction):
            if not action.value is None:
                new_value = get_value_obj_global(action.value, **state_kwargs)
            else:
                new_value = provided_object
            # assert False
            
            object_to_append_to, operator_indices = traverse_static_route_global(object_in_focus, action.route, **state_kwargs)
            if action.action == "+=":
                object_to_append_to += new_value
            else:
                object_to_append_to -= new_value
            object_in_focus = insert_in_static_route_global(object_in_focus, operator_indices, object_to_append_to, **state_kwargs)
            object = insert_in_static_route_global(object, current_indices, object_in_focus, **state_kwargs)
        
        # May add insertions here.
        
        elif isinstance(action, backOut):
            current_indices = current_indices[:-action.count]
            object_in_focus = retrieve_value_from_obj(object, current_indices)
    
    object = insert_in_static_route_global(object, current_indices, object_in_focus, **state_kwargs)
    
    if return_provided_object_routes:
        return object, routes_for_provided_object
    
    return object

def evaluate_condition_basic(toolchain_state : Union[list, dict],
                             node_inputs_state : Union[list, dict],
                             node_outputs_state : Union[list, dict],
                             toolchain_files : Union[list, dict],
                             condition : conditionBasic,
                             provided_object : Any,
                             get_files_callable : getFilesCallableType = None) -> bool:
    """
    Evaluate a singular feed mapping condition.
    """
    
    state_kwargs = {
        "toolchain_state" : toolchain_state,
        "node_inputs_state" : node_inputs_state,
        "node_outputs_state" : node_outputs_state,
        "toolchain_files" : toolchain_files,
        "get_files_callable" : get_files_callable
    }
        
    if condition.variableOne is None:
        variable_one = provided_object
    else:
        variable_one = get_value_obj_global(condition.variableOne, **state_kwargs)
        
    variable_two = get_value_obj_global(condition.variableTwo, **state_kwargs)
    
    # condition.operator is of type Literal["==", "!=", ">", "<", ">=", "<=", "in", "not in", "is", "is not", "is instance"]
    
    # Evaluate the condition.
    if condition.operator == "==":
        return variable_one == variable_two
    elif condition.operator == "!=":
        return variable_one != variable_two
    elif condition.operator == ">":
        return variable_one > variable_two
    elif condition.operator == "<":
        return variable_one < variable_two
    elif condition.operator == ">=":
        return variable_one >= variable_two
    elif condition.operator == "<=":
        return variable_one <= variable_two
    elif condition.operator == "in":
        return variable_one in variable_two
    elif condition.operator == "not in":
        return variable_one not in variable_two
    elif condition.operator == "is":
        return variable_one is variable_two
    elif condition.operator == "is not":
        return not variable_one is variable_two
    # elif condition.operator == "is instance":
    #     return isinstance(variable_one, variable_two)
    else:
        raise ValueError(f"Unknown operator: {condition.operator}")
 
def evaluate_condition(toolchain_state : Union[list, dict],
                       node_inputs_state : Union[list, dict],
                       node_outputs_state : Union[list, dict],
                       toolchain_files : Union[list, dict],
                       condition : Union[Condition, conditionBasic],
                       provided_object : Any,
                       get_files_callable : getFilesCallableType = None) -> bool:
    """
    Evaluate a singular feed mapping condition.
    """
    
    state_kwargs = {
        "toolchain_state" : toolchain_state,
        "node_inputs_state" : node_inputs_state,
        "node_outputs_state" : node_outputs_state,
        "toolchain_files" : toolchain_files,
        "get_files_callable" : get_files_callable
    }
    
    # Check if condition has attribute `type` to determine if it is a condition or conditionBasic.
    if not hasattr(condition, "type"):
        return evaluate_condition_basic(condition=condition, provided_object=provided_object, **state_kwargs)
    
    if condition.type == "singular":
        return evaluate_condition_basic(condition=condition.statement, provided_object=provided_object, **state_kwargs)
    elif condition.type == "not":
        return not evaluate_condition_basic(condition=condition.statement, provided_object=provided_object, **state_kwargs)
    elif condition.type == "and":
        return all([evaluate_condition(condition=condition.statement[i], provided_object=provided_object, **state_kwargs) for i in range(len(condition.statement))])
    elif condition.type == "or":
        return any([evaluate_condition(condition=condition.statement[i], provided_object=provided_object, **state_kwargs) for i in range(len(condition.statement))])

def dict_diff_deleted(d1 : dict, d2 : dict) -> dict:
    """
    This function returns a list containing strings and/or dicts. 
    The strings are top-level keys that have been fully deleted, 
    while the dicts specify new routes. Any value in the dict should 
    follow this same structure of strings and new dicts.
    """
    assert isinstance(d1, dict) and isinstance(d2, dict), "dict_diff_deleted called with non-dict type"
    
    diff = []
    for k in d2.keys():
        if k not in d1:
            diff.append(k)
        elif isinstance(d2[k], dict):
            nested_diff = dict_diff_deleted(d1.get(k, {}), d2[k])
            if nested_diff:
                diff.append({k: nested_diff})
    return diff

def dict_diff_append_and_update(d1 : dict, d2 : dict) -> Tuple[List[List[Union[str, int]]], dict, dict]:
    """
    This works at finding elements differences in lists and strings that can just be appended.
    It also finds differences in dictionaries that need to be updated.
    It returns three values as a tuple of (diff_route, diff_append, diff_update).
    diff_route contains the routes at which to insert the diff_append values to avoid confusion.
    """
    
    assert isinstance(d1, dict) and isinstance(d2, dict), "dict_diff_append_and_update called with non-dict type"
    
    diff_append_routes, diff_append, diff_update = [], {}, {}
    for k, v in d1.items():
        if k not in d2:
            if isinstance(v, dict):
                nested_diff_append_routes, nested_diff_append, nested_diff_update = dict_diff_append_and_update(v, {})
                if nested_diff_append:
                    diff_append[k] = nested_diff_append
                    for route in nested_diff_append_routes:
                        diff_append_routes.append([k] + route)
                if nested_diff_update:
                    diff_update[k] = nested_diff_update
            else:
                diff_update[k] = v
        elif isinstance(v, dict) and isinstance(d2[k], dict):
            nested_diff_append_routes, nested_diff_append, nested_diff_update = dict_diff_append_and_update(v, d2[k])
            
            if nested_diff_append:
                diff_append[k] = nested_diff_append
                for route in nested_diff_append_routes:
                    diff_append_routes.append([k] + route)
            if nested_diff_update:
                diff_update[k] = nested_diff_update
        elif isinstance(v, (str, list)) and len(v) > len(d2[k]) and v[:len(d2[k])] == d2[k]:
            diff_append[k] = v[len(d2[k]):]
            diff_append_routes.append([k])
        elif v != d2[k]:
            diff_update[k] = v
    return diff_append_routes, diff_append, diff_update

def recursive_shallow_copy(input_dict : dict):
    """
    Recursive shallow copy, effectively the same as shallow
    copy but it works for nested dictionaries.
    """
    def value_copy(value_in : Any):
        if isinstance(value_in, dict):
            return recursive_shallow_copy(value_in)
        elif isinstance(value_in, list):
            return [value_copy(e) for e in value_in]
        else:
            return value_in
    
    new_dict = {}
    
    for key, value in input_dict.items():
        new_dict[key] = value_copy(value)
        
    return new_dict