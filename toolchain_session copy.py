import os, json
from copy import deepcopy
from typing import Callable, Any, List, Dict, Union, Awaitable
from QueryLake.typing.toolchains import *
import time

from QueryLake.misc_functions.toolchain_state_management import *


def dict_diff(d1, d2):
    diff = {}
    for k, v in d1.items():
        if k not in d2:
            diff[k] = v
        elif isinstance(v, dict) and isinstance(d2[k], dict):
            nested_diff = dict_diff(v, d2[k])
            if nested_diff:
                diff[k] = nested_diff
    return diff

def dict_diff_v2(d1, d2):
    diff = {}
    for k, v in d1.items():
        if k not in d2:
            diff[k] = v
        elif isinstance(v, dict) and isinstance(d2[k], dict):
            nested_diff = dict_diff_v2(v, d2[k])
            if nested_diff:
                diff[k] = nested_diff
        elif v != d2[k]:
            diff[k] = v
    return diff

def dict_diff_v3_string(d1, d2):
    diff = {}
    for k, v in d1.items():
        if k not in d2:
            diff[k] = v
        elif isinstance(v, dict) and isinstance(d2[k], dict):
            nested_diff = dict_diff_v3_string(v, d2[k])
            if nested_diff:
                diff[k] = nested_diff
        elif (isinstance(v, (list, str)) and v != d2[k]):
            diff[k] = v
    return diff

def dict_diff_v4_strings_lists(d1, d2):
    diff = {}
    for k, v in d1.items():
        if k not in d2:
            diff[k] = v
        elif isinstance(v, dict) and isinstance(d2[k], dict):
            nested_diff = dict_diff_v4_strings_lists(v, d2[k])
            if nested_diff:
                diff[k] = nested_diff
        elif isinstance(v, (list, str)) and v != d2.get(k):
            diff[k] = v
    return diff

def dict_diff_v5_append(d1, d2):
    """
    This works at finding elements differences in lists and strings that can just be appended.
    
    Ideally, we would have this, and another difference dictionary for new values not including the ones that can be appended returned here.
    """
    
    diff = {}
    for k, v in d1.items():
        if k not in d2:
            if isinstance(v, (list, str)):
                diff[k] = v
            elif isinstance(v, dict):
                nested_diff = dict_diff_v5_append(v, {})
                if nested_diff:
                    diff[k] = nested_diff
        elif isinstance(v, dict) and isinstance(d2[k], dict):
            nested_diff = dict_diff_v5_append(v, d2[k])
            if nested_diff:
                diff[k] = nested_diff
        elif isinstance(v, str) and v != d2[k]:
            diff[k] = v.replace(d2[k], '')
        elif isinstance(v, list) and v != d2[k]:
            diff[k] = [item for item in v if item not in d2[k]]
    return diff

def dict_diff_v5_append(d1, d2):
    """
    This works at finding elements differences in lists and strings that can just be appended.
    
    Ideally, we would have this, and another difference dictionary for new values not including the ones that can be appended returned here.
    """
    
    diff = {}
    for k, v in d1.items():
        if k not in d2:
            if isinstance(v, (list, str)):
                diff[k] = v
            elif isinstance(v, dict):
                nested_diff = dict_diff_v5_append(v, {})
                if nested_diff:
                    diff[k] = nested_diff
        elif isinstance(v, dict) and isinstance(d2[k], dict):
            nested_diff = dict_diff_v5_append(v, d2[k])
            if nested_diff:
                diff[k] = nested_diff
        elif isinstance(v, str) and v != d2[k]:
            diff[k] = v.replace(d2[k], '')
        elif isinstance(v, list) and v != d2[k]:
            diff[k] = [item for item in v if item not in d2[k]]
    return diff


def dict_diff_v6_delete(d1, d2, path=''):
    diff = {}
    for k, v in d1.items():
        new_path = f'{path}.{k}' if path else k
        if k not in d2:
            if isinstance(v, (list, str)):
                diff[new_path] = v
            elif isinstance(v, dict):
                nested_diff = dict_diff_v6_delete(v, {}, new_path)
                diff.update(nested_diff)
        elif isinstance(v, dict) and isinstance(d2[k], dict):
            nested_diff = dict_diff_v6_delete(v, d2[k], new_path)
            diff.update(nested_diff)
        elif isinstance(v, str) and v != d2[k]:
            diff[new_path] = v.replace(d2[k], '')
        elif isinstance(v, list) and v != d2[k]:
            diff[new_path] = [item for item in v if item not in d2[k]]
    return diff

def dict_diff_deleted(d1, d2):
    """
    This function returns a list containing strings and/or dicts. 
    The strings are top-level keys that have been fully deleted, 
    while the dicts specify new routes. Any value in the dict should 
    follow this same structure of strings and new dicts.
    """
    diff = []
    for k in d1.keys():
        if k not in d2:
            diff.append(k)
        elif isinstance(d1[k], dict):
            nested_diff = dict_diff_deleted(d1[k], d2.get(k, {}))
            if nested_diff:
                diff.append({k: nested_diff})
    return diff

def dict_diff_v7_append_and_create(d1 : dict, d2 : dict):
    """
    This works at finding elements differences in lists and strings that can just be appended.
    
    Ideally, we would have this, and another difference dictionary for new values not including the ones that can be appended returned here.
    """
    
    diff_append, diff_update = {}, {}
    for k, v in d1.items():
        if k not in d2:
            # if isinstance(v, (list, str)):
            #     diff_update[k] = v
            if isinstance(v, dict):
                nested_diff_append, nested_diff_update = dict_diff_v7_append_and_create(v, {})
                if nested_diff_append:
                    diff_append[k] = nested_diff_append
                if nested_diff_update:
                    diff_update[k] = nested_diff_update
            else:
                diff_update[k] = v
        elif isinstance(v, dict) and isinstance(d2[k], dict):
            nested_diff_append, nested_diff_update = dict_diff_v7_append_and_create(v, d2[k])
            
            if nested_diff_append:
                diff_append[k] = nested_diff_append
            if nested_diff_update:
                diff_update[k] = nested_diff_update
        elif isinstance(v, (str, list)) and len(v) > len(d2[k]) and v[:len(d2[k])] == d2[k]:
            diff_append[k] = v[len(d2[k]):]
        elif v != d2[k]:
            diff_update[k] = v
    return diff_append, diff_update



if __name__ == "__main__":
    test_toolchain_state = {
        "dir_1": {
            "dir_2": {
                "state_value": "this is from toolchain state"
            }
        }
    }
    
    test_node_inputs_state = {
        "dir_1": {
            "dir_2": {
                "node_input": "this is from node input"
            }
        }
    }
    
    test_node_outputs_state = {
        "dir_1": {
            "dir_2": {
                "node_output": "this is from node output"
            }
        }
    }
    
    test_node_branching_state = {
        "test_object": {
            "test_corner": {
                "val_1": "this is from branching state"
            }
        }
    }
    
    staticValue(value=2)
    
    createAction(
        # insertion_values=[staticValue(value=2)],
        initialValue=staticValue(value=2),
        route = [ "val_2" ]
    )
    
    
    test_sequence_actions : List[sequenceAction] = [
        createAction(
            # initialValue=valueFromBranchingState(route=[ "test_object", "test_corner", "val_1" ]),
            # initialValue=staticValue(value=2),
            route = [ "passed_initial_value_top_level_dir" ]
        ),
        "dir_1",
        createAction(
            # initialValue=valueFromBranchingState(route=[ "test_object", "test_corner", "val_1" ]),
            route = [ "passed_initial_value_pre" ]
        ),
        "dir_2",
        createAction(
            initialValue=staticValue(value=2),
            route = [ "val_2" ]
        ),
        backOut(count=1),
        createAction(
            initialValue=stateValue(route=["dir_1", "dir_2", "state_value"]),
            route = [ "state_value" ]
        ),
        createAction(
            # initialValue=valueFromBranchingState(route=[ "test_object", "test_corner", "val_1" ]),
            route = [ "passed_initial_value" ]
        ),
        createAction(
            initialValue=valueFromBranchingState(route=[ "test_object", "test_corner", "val_1" ]),
            route = [ "branching_state_retrieved" ]
        ),
        createAction(
            initialValue=getNodeInput(route=[ "dir_1", "dir_2", "node_input" ]),
            route = [ "node_input_retrieved" ]
        ),
        createAction(
            initialValue=getNodeOutput(route=[ "dir_1", "dir_2", "node_output" ]),
            route = [ "node_output_retrieved" ]
        ),
        createAction(
            initialValue=staticValue(value={
                "role": "user"
            }),
            insertion_values=[None],
            insertions=[ ["content"] ],
            route = [ "attempt_chat_reconstruction" ]
        ),
        operatorAction(
            action="+=",
            value=staticValue(value=" this is an appended message"),
            route = [ "attempt_chat_reconstruction", "content" ]
        ),
        createAction(
            initialValue=staticValue(value=[
                1, 2, 3, 4, 4, 5, 6
            ]),
            route = [ "counting_sequence" ]
        ),
        deleteAction(
            route=[ "counting_sequence", 4 ]
            # routes=[
            #     [ "counting_sequence", 4 ],
            #     [ "counting_sequence", 4 ],
            # ]
        ),
        deleteAction(
            route=[ "counting_sequence", 4 ]
            # routes=[
            #     [ "counting_sequence", 4 ],
            #     [ "counting_sequence", 4 ],
            # ]
        ),
        appendAction(
            route=["counting_sequence"],
        ),
        createAction(
            initialValue=staticValue(value="this message should not exist"),
            route = [ "create_action_condition_false" ],
            condition=conditionBasic(
                operator="is not",
                variableOne=staticValue(value=2),
                variableTwo=staticValue(value=2)
            )
        ),
        createAction(
            initialValue=staticValue(value="this message *should* exist"),
            route = [ "create_action_condition_true" ],
            condition=conditionBasic(
                operator="!=",
                variableOne=staticValue(value=2),
                variableTwo=staticValue(value=3)
            )
        )
    ]
    
    tmp_initial_value = "<<INITIAL_VALUE>>"
    
    print(deleteAction(route=[ "counting_sequence", 4 ]))
    start_time = time.time()
    modification_object = run_sequence_action_on_object(
        test_toolchain_state,
        test_toolchain_state,
        test_node_inputs_state,
        test_node_outputs_state,
        test_sequence_actions,
        tmp_initial_value,
        branching_state=test_node_branching_state
    )
    end_time = time.time()
    
    print(json.dumps(modification_object, indent=4))
    print(f"Time taken: {end_time - start_time}")

    # dict1 = {'a': 1, 'b': {'x': 2, 'y': 3}, 'c': 3, 'e': {'x': 2, 'y': 3}, 'z': 6}
    # dict2 = {'b': {'x': 2}, 'c': 3, 'd': 4, 'e': {'x': 2, 'y': 3}, 'z': 5}

    # difference = dict_diff(dict1, dict2)

    # print(difference)  # Output: {'a': 1, 'b': {'y': 3}}
    
    # dict1 = {'a': 1, 'b': {'x': 2, 'y': 3}, 'c': 3}
    # dict2 = {'b': {'x': 2}, 'c': 4, 'd': 4}

    # difference = dict_diff_v2(dict1, dict2)

    # print(difference)  # Output: {'a': 1, 'b': {'y': 3}, 'c': 3}
    
    # dict1 = {'a': 1, 'b': {'x': 'hello!!', 'y': [1, 2, 3]}, 'c': 'world_2'}
    # dict2 = {'b': {'x': 'hello'}, 'c': 'world', 'd': 4}

    # difference = dict_diff_v3_string(dict1, dict2)

    # print(difference)  # Output: {'a': 1, 'b': {'y': [1, 2, 3]}}
    
    # dict1 = {'a': 1, 'b': {'x': 'hello!!', 'y': [1, 2, 3]}, 'c': 'world_2'}
    # dict2 = {'b': {'x': 'hello'}, 'c': 'world', 'd': 4}

    # difference = dict_diff_v5_append(dict1, dict2)

    # print(difference)  # Output: {'a': 1, 'b': {'x': '!!', 'y': [1, 2, 3]}, 'c': '_2'}
    
    
    # dict1 = {'a': 1, 'b': {'x': 'hello!!', 'y': [1, 2, 3]}, 'c': 'world_2'}
    # dict2 = {'b': {'x': 'hello'}, 'c': 'world', 'd': 4}

    # difference = dict_diff_v6_delete(dict1, dict2)

    # print(difference)  # Output: {'a': 1, 'b.x': '!!', 'b.y': [1, 2, 3], 'c': '_2'}
    
    
    # print("\n\nAppend and update")
    # dict1 = {'a': 1, 'b': {'x': 'hello!!', 'y': [1, 2, 3]}, 'c': 'world_2', 'e': {'x': 2, 'y': 3}, 'z': 6}
    # dict2 = {'b': {'x': 'hello'}, 'c': 'world', 'd': 4}

    # difference, update = dict_diff_v7_append_and_create(dict1, dict2)

    # print(difference)  # Output: ['d']
    # print(update)
    
    # test_dict_1 = {"a": 1, "b": 2, "c": 3}
    # test_dict_2 = {"d": 4, "e": 5, "f": 6}
    
    # test_dict_3 = test_dict_1.copy()
    # test_dict_3.update(test_dict_2)
    # print(test_dict_1)
    # print(test_dict_3)
    
    
    
    inputs_2 = {
        "model_parameters": {
            "model_choice": "mistral-7b-instruct-v0.1",
            "chat_history": [
                {
                    "role": "user",
                    "content": "What is the Riemann-Roch theorem?"
                }
            ],
            "max_tokens": 1000,
            "temperature": 0.5,
            "top_p": 0.9,
            "repetition_penalty": 1.15,
            "stop": [
                "</s>"
            ]
        },
        "auth": {
            "username": "0abb9949-ec97-4f4c-85a8-6557a31c",
            "password_prehash": "7fa5a964699e1ffb695a6785f97d3f4c3fca673e8596040868fa6dc667583dd0"
        }
    }
    
    sequence_2 = [
        {
            "type": "createAction",
            "route": [
                "model_parameters"
            ]
        }
    ]
    
    sequence_2 = [createAction(**elements) for elements in sequence_2]
    
    init_val_2 = inputs_2["model_parameters"]
    
    outputs_2 = {}
    
    target_state = {}
    
    start_time = time.time()
    modification_object_2 = run_sequence_action_on_object(
        target_state,
        {},
        outputs_2,
        outputs_2,
        sequence_2,
        init_val_2,
    )
    end_time = time.time()
    
    print(json.dumps(modification_object_2, indent=4))
    print(f"Time taken: {end_time - start_time}")
    
    
    