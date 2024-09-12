from websockets.sync.client import connect, Connection
from copy import deepcopy
import time
from typing import List, Union, Any
import json

def append_in_route(object_for_static_route : Union[list, dict], 
                    route : List[Union[str, int]], 
                    value : Any, 
                    only_add : bool = False) -> Union[list, dict]:
    """
    Insert a value into an object using a list of strings and ints.
    """
    
    if len(route) > 0:
        # print(route)
        object_for_static_route[route[0]] = append_in_route(object_for_static_route[route[0]], route[1:], value, only_add=only_add)
    else:
        if only_add:
            # print("ADDING", value, "TO", object_for_static_route)
            object_for_static_route += value
            # print("RESULT", object_for_static_route)
        elif isinstance(object_for_static_route, list):
            object_for_static_route.append(value)
        else:
            print(value, end="")
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

def run_delete_state(state_input : dict, delete_states: Union[List[Union[str, int, dict]], dict]):
    if isinstance(delete_states, dict):
        for key, value in delete_states.items():
            state_input[key] = run_delete_state(state_input[key], value)
        return state_input
    
    assert isinstance(delete_states, list), "Something went wrong, delete_states must be a list if not a dict."
    
    if len(delete_states) > 0 and isinstance(delete_states[0], int):
        delete_states = sorted(delete_states, reverse=True)
        
    for e in delete_states:
        del state_input[e]
    
    return state_input
    

def run_state_diff(state_input : dict, state_diff_specs):

    append_routes : List[List[Union[str, int]]] = state_diff_specs["append_routes"] if "append_routes" in state_diff_specs else []
    append_state: dict = state_diff_specs["append_state"] if "append_state" in state_diff_specs else {}
    update_state: dict = state_diff_specs["update_state"] if "update_state" in state_diff_specs else {}
    delete_states: List[Union[str, int, dict]] = state_diff_specs["delete_states"] if "delete_states" in state_diff_specs else []
    
    for route in append_routes:
        val_get = retrieve_value_from_obj(append_state, route)
        # print("APPENDING", val_get, "TO", route)
        
        state_input = append_in_route(
            state_input, 
            route, 
            val_get,
            only_add=True
        )
    
    state_input.update(update_state)
    
    for delete_state in delete_states:
        state_input = run_delete_state(state_input, delete_state)
    
    return state_input

def check_keys(keys_1 : List[str], keys_2 : List[str]):
    return (sorted(keys_1) == sorted(keys_2))

async def wait_for_command_finish(websocket : Connection, toolchain_state : dict):
    final_output, stream_mappings = {}, {}
    
    while True:
        response = websocket.recv()
        response : dict = json.loads(response)
        
        # print(json.dumps(response, indent=4))
        if "ACTION" in response and response["ACTION"] == "END_WS_CALL":
            break
        else:
            final_output = response
        
        if "trace" in response:
            print("ERROR RECIEVED")
            print(response["trace"])
            return
            
        elif "state" in response:
            toolchain_state = response["state"]
            
        elif "type" in response:
            
            if response["type"] == "streaming_output_mapping":
                stream_mappings[response["stream_id"]] = response["routes"]
            
            elif response["type"] == "state_diff":
                toolchain_state = run_state_diff(deepcopy(toolchain_state), response)
                
        elif check_keys(["s_id", "v"], list(response.keys())):
            
            routes_get : List[List[Union[str, int]]] = stream_mappings[response["s_id"]]
            
            for route_get in routes_get:
                toolchain_state = append_in_route(toolchain_state, route_get, response["v"])
            
        elif check_keys(["event_result"], list(response.keys())):
            final_output = response["event_result"]
    
    return final_output, toolchain_state
