import os, json
from copy import deepcopy
from typing import Callable, Any, List, Dict, Union, Awaitable
from QueryLake.typing.toolchains import *
import time
import re
import inspect

from QueryLake.api import api
from QueryLake.misc_functions.toolchain_state_management import *
# from QueryLake.misc_functions.prompt_construction import construct_chat_history
from QueryLake.typing.config import Config, Model
# from tests.toolchain_efficient_receiver import run_state_diff


if __name__ == "__main__":
    
    API_FUNCTIONS = [str(pair[0]) for pair in inspect.getmembers(api, inspect.isfunction)]
    print(json.dumps(API_FUNCTIONS, indent=4))
    print("\nEXCLUDED\n")
    print(json.dumps(api.excluded_member_function_descriptions, indent=4))
    
    API_FUNCTIONS_ALLOWED = list(set(API_FUNCTIONS) - set(api.excluded_member_function_descriptions))
    API_FUNCTIONS_ALLOWED = [func for func in API_FUNCTIONS_ALLOWED if (not re.match(r"__.*?__", func))]
    
    print("\nFINAL\n")
    print(json.dumps(API_FUNCTIONS_ALLOWED, indent=4))
    
    # test_toolchain_state = {
    #     "dir_1": {
    #         "dir_2": {
    #             "state_value": "this is from toolchain state"
    #         },
    #         "test_value": "dir_2"
    #     }
    # }
    
    # test_node_inputs_state = {
    #     "dir_1": {
    #         "dir_2": {
    #             "node_input": "this is from node input"
    #         }
    #     }
    # }
    
    # test_node_outputs_state = {
    #     "dir_1": {
    #         "dir_2": {
    #             "node_output": "this is from node output"
    #         }
    #     }
    # }
    
    
    
    
    
    
    # dynamic_route_test : staticRoute = [
    #     "dir_1",
    #     indexRouteRetrievedStateValue(**{
    #         "getFromState": { "route": [ "dir_1", "test_value" ] } 
    #     })
    # ]
    
    # staticValue(value=2)
    
    # createAction(
    #     # insertion_values=[staticValue(value=2)],
    #     initialValue=staticValue(value=2),
    #     route = [ "val_2" ]
    # )
    
    
    # test_sequence_actions : List[sequenceAction] = [
    #     createAction(
    #         # initialValue=valueFromBranchingState(route=[ "test_object", "test_corner", "val_1" ]),
    #         # initialValue=staticValue(value=2),
    #         route = [ "passed_initial_value_top_level_dir" ]
    #     ),
    # ]
    
    # tmp_initial_value = "<<INITIAL_VALUE>>"
    
    # run_sequence_args = {
    #     "subject_state" : test_toolchain_state,
    #     "toolchain_state" : test_toolchain_state,
    #     "node_inputs_state" : test_node_inputs_state,
    #     "node_outputs_state" : test_node_outputs_state,
    #     "provided_object" : tmp_initial_value
    # }
    
    
    # def run_sequence_test(sequence : List[sequenceAction]) -> dict:
        
    
    #     run_sequence_args["toolchain_state"] = run_sequence_args["subject_state"]
        
    #     # print(deleteAction(route=[ "counting_sequence", 4 ]))
    #     start_time = time.time()
    #     modification_object = run_sequence_action_on_object(
    #         sequence=sequence,
    #         **run_sequence_args
    #     )
    #     end_time = time.time()
        
    #     print(json.dumps(modification_object, indent=4))
    #     print(f"Time taken: {end_time - start_time}")
        
    #     return modification_object
    
    # def check_obj(obj : dict, route : List[Union[str, int]], correct_value : Any, error_message : str):
    #     test_passed = True
        
    #     try:
    #         get_value = retrieve_value_from_obj(obj, route)
    #         assert get_value == correct_value, error_message
    #     except:
    #         test_passed = False
            
    #     assert test_passed == True, error_message
    
    
    
    # # 1 - Test createAction at top level.
    # run_sequence_args["subject_state"] = run_sequence_test(test_sequence_actions)
    # check_obj(run_sequence_args["subject_state"], ["passed_initial_value_top_level_dir"], "<<INITIAL_VALUE>>", "Test 1 failed")
    
    
    
    # # 2 - Test ability to pull from toolchain state
    # test_sequence_actions : List[sequenceAction] = [
    #     createAction(
    #         # initialValue=valueFromBranchingState(route=[ "test_object", "test_corner", "val_1" ]),
    #         # initialValue=staticValue(value=2),
    #         route = dynamic_route_test + ["test_value"]
    #     ),
    # ]
    # run_sequence_args["subject_state"] = run_sequence_test(test_sequence_actions)
    # check_obj(run_sequence_args["subject_state"], ["dir_1", "dir_2", "test_value"], "<<INITIAL_VALUE>>", "Test 2 failed")
    
    # temp_dir = ["dir_1", "dir_2"]
    
    # # 3 - Test createAction with given value in a specified directory.
    # test_sequence_actions : List[sequenceAction] = [
    #     "dir_1",
    #     createAction(
    #         # initialValue=valueFromBranchingState(route=[ "test_object", "test_corner", "val_1" ]),
    #         route = [ "passed_initial_value_pre" ]
    #     )
    # ]
    # run_sequence_args["subject_state"] = run_sequence_test(test_sequence_actions)
    # check_obj(run_sequence_args["subject_state"], ["dir_1", "passed_initial_value_pre"], "<<INITIAL_VALUE>>", "Test 3 failed")
    
    
    
    # # 4 - Test createAction with staticValue.
    # # 5 - Test createAction with stateValue.
    # # 6 - Test createAction with nodeInputValue.
    # # 7 - Test createAction with nodeOutputValue.
    # # 8 - Test createAction with construction of an object and insertions.
    # test_sequence_actions : List[sequenceAction] = temp_dir + [
    #     createAction(
    #         initialValue=staticValue(value=2),
    #         route = [ "val_2" ]
    #     ),
    #     backOut(count=1),
    #     createAction(
    #         initialValue=stateValue(route=["dir_1", "dir_2", "state_value"]),
    #         route = [ "state_value_created" ]
    #     ),
    #     createAction(
    #         # initialValue=valueFromBranchingState(route=[ "test_object", "test_corner", "val_1" ]),
    #         route = [ "passed_initial_value" ]
    #     ),
    #     createAction(
    #         initialValue=getNodeInput(route=[ "dir_1", "dir_2", "node_input" ]),
    #         route = [ "node_input_retrieved" ]
    #     ),
    #     createAction(
    #         initialValue=getNodeOutput(route=[ "dir_1", "dir_2", "node_output" ]),
    #         route = [ "node_output_retrieved" ]
    #     ),
    #     "dir_2",
    #     backOut(count=2),
    #     createAction(
    #         initialValue=staticValue(value={
    #             "role": "user"
    #         }),
    #         insertion_values=[None],
    #         insertions=[ ["content"] ],
    #         route = [ "attempt_chat_reconstruction" ]
    #     )
    # ]
    # run_sequence_args["subject_state"] = run_sequence_test(test_sequence_actions)
    
    # check_obj(run_sequence_args["subject_state"], ["dir_1", "dir_2", "val_2"], 2, "Test 4 failed")
    # check_obj(run_sequence_args["subject_state"], ["dir_1", "state_value_created"], "this is from toolchain state", "Test 5 failed")
    # check_obj(run_sequence_args["subject_state"], ["dir_1", "node_input_retrieved"], "this is from node input", "Test 6 failed")
    # check_obj(run_sequence_args["subject_state"], ["dir_1", "node_output_retrieved"], "this is from node output", "Test 7 failed")
    # check_obj(run_sequence_args["subject_state"], ["attempt_chat_reconstruction"], {
    #     "role": "user",
    #     "content": "<<INITIAL_VALUE>>"
    # }, "Test 8 failed")
    
    
    
    # # 8  - Test operatorAction with appending to string.
    # # 9  - Test create action with array.
    # # 10 - Test deleteAction on said array.
    # # 11 - Test append inital value to said array.
    # test_sequence_actions : List[sequenceAction] = [
    #     operatorAction(
    #         action="+=",
    #         value=staticValue(value=" this is an appended message"),
    #         route = [ "attempt_chat_reconstruction", "content" ]
    #     ),
    #     "dir_1",
    #     createAction(
    #         initialValue=staticValue(value=[
    #             1, 2, 3, 4, 4, 5, 6
    #         ]),
    #         route = [ "counting_sequence" ]
    #     ),
    #     deleteAction(
    #         route=[ "counting_sequence", 4 ]
    #         # routes=[
    #         #     [ "counting_sequence", 4 ],
    #         #     [ "counting_sequence", 4 ],
    #         # ]
    #     ),
    #     deleteAction(
    #         route=[ "counting_sequence", 4 ]
    #         # routes=[
    #         #     [ "counting_sequence", 4 ],
    #         #     [ "counting_sequence", 4 ],
    #         # ]
    #     ),
    #     appendAction(
    #         route=["counting_sequence"],
    #     ),
    #     createAction(
    #         initialValue=staticValue(value="this message should not exist"),
    #         route = [ "create_action_condition_false" ],
    #         condition=conditionBasic(
    #             operator="is not",
    #             variableOne=staticValue(value=2),
    #             variableTwo=staticValue(value=2)
    #         )
    #     ),
    #     createAction(
    #         initialValue=staticValue(value="this message *should* exist"),
    #         route = [ "create_action_condition_true" ],
    #         condition=conditionBasic(
    #             operator="!=",
    #             variableOne=staticValue(value=2),
    #             variableTwo=staticValue(value=3)
    #         )
    #     )
    # ]
    
    # run_sequence_args["subject_state"] = run_sequence_test(test_sequence_actions)
    # check_obj(run_sequence_args["subject_state"], ["dir_1", "dir_2", "val_2"], 2, "Test 4 failed")
    # check_obj(run_sequence_args["subject_state"], ["dir_1", "state_value_created"], "this is from toolchain state", "Test 5 failed")
    # check_obj(run_sequence_args["subject_state"], ["dir_1", "node_input_retrieved"], "this is from node input", "Test 6 failed")
    # check_obj(run_sequence_args["subject_state"], ["dir_1", "node_output_retrieved"], "this is from node output", "Test 7 failed")
    # check_obj(run_sequence_args["subject_state"], ["attempt_chat_reconstruction"], {
    #     "role": "user",
    #     "content": "<<INITIAL_VALUE>>"
    # }, "Test 8 failed")
    
    
    
    
    dict1 = {'a': 1, 'b': {'x': 'hello!!', 'y': [1, 2, 3]}, 'c': 'world_2', 'e': {'x': 2, 'y': 3}, 'f': {'x': 2}, 'z': 6}
    
    dict2 = {'b': {'x': 'hello'}, 'c': 'world', 'd': 4, 'f': {'x': 2, 'y': 3}, 'z': 6}

    # difference, update = dict_diff_v7_append_and_create(dict1, dict2)
    diff_append_routes, diff_append, diff_update = dict_diff_append_and_update(dict1, dict2)
    diff_delete = dict_diff_deleted(dict1, dict2)
    

    print("const diff_append_routes =", diff_append_routes, ";")  # Output: ['d']
    print("const diff_append =", diff_append, ";")  # Output: {'d': 4}
    print("const diff_update =", diff_update, ";")  # Output: {'c': 'world', 'b': {'x': 'hello'}}
    print("const diff_delete =", diff_delete, ";")  # Output: {'e': {'x': 2, 'y': 3}, 'z': 6}
    
    
    # dict_made = run_state_diff(dict1, diff_append_routes, diff_append, diff_update, diff_delete)
    
    # print("const dict_made_truth =", dict_made, ";")
    
    
    # print(update)
    
    
    