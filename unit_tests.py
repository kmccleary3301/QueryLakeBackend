import os, json
from copy import deepcopy
from typing import Callable, Any, List, Dict, Union, Awaitable
from QueryLake.typing.toolchains import *
import time

from QueryLake.misc_functions.toolchain_state_management import *
from QueryLake.misc_functions.prompt_construction import construct_chat_history
from QueryLake.typing.config import Config, Model

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
    
    inputs_2 = {
        "model_parameters": {
            "model_choice": "mistral-7b-instruct-v0.1",
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
        },
        "question": "What is the Riemann-Roch theorem?"
    }
    
    sequence_2 = [
        {
            "type": "appendAction", 
            "initialValue": {
                "type": "staticValue",
                "value": {
                    "role": "user"           
                }
            },
            "insertion_values": [ None ],
            "insertions": [ [ "content" ] ],
            "route" : [ "chat_history" ]
        }
    ]
    
    sequence_2 = [appendAction(**elements) for elements in sequence_2]
    
    init_val_2 = inputs_2["question"]
    
    outputs_2 = {}
    
    target_state = {"chat_history": []}
    
    
    start_time = time.time()
    modification_object_2, routes_get = run_sequence_action_on_object(
        target_state,
        target_state,
        inputs_2,
        outputs_2,
        sequence_2,
        init_val_2,
        return_provided_object_routes=True
    )
    
    end_time = time.time()
    
    print(json.dumps(modification_object_2, indent=4))
    print("ROUTES")
    print(json.dumps(routes_get, indent=4))
    print(f"Time taken: {end_time - start_time}")
    
    # Routes should contain ["chat_history", 0, "content"]
    
    chat_history = [
        {
            "role": "user",
            "content": "What is the Riemann-Roch theorem?"
        },
        {
            "role": "assistant",
            "content": "The Riemann-Roch Theorem is a fundamental result in number theory which relates the distribution of prime numbers to certain arithmetic properties of elliptic curves over finite fields. It states that for every positive integer $n$, there exists an even integer $k$ such that the number of primes less than or equal to $\\sqrt{2n}$ with residue class $k\\pmod n$ is equal to the number of elliptic curves defined over $\\mathbb{F}_n$ with discriminant divisible by $(n-1)$ and having a point of order $k$. The theorem has many important applications in cryptography, algebraic geometry, and other areas of mathematics."
        },
        {
            "role": "user",
            "content": "Who are the two people the Riemann-Roch Theorem is named after?"
        },
        {
            "role": "assistant",
            "content": "The Riemann-Roch Theorem is a fundamental result in number theory which relates the distribution of prime numbers to certain arithmetic properties of elliptic curves over finite fields. It states that for every positive integer $n$, there exists an even integer $k$ such that the number of primes less than or equal to $\\sqrt{2n}$ with residue class $k\\pmod n$ is equal to the number of elliptic curves defined over $\\mathbb{F}_n$ with discriminant divisible by $(n-1)$ and having a point of order $k$. The theorem has many important applications in cryptography, algebraic geometry, and other areas of mathematics."
        },
        {
            "role": "user",
            "content": "Who are the two people the Riemann-Roch Theorem is named after?"
        },
        # {
        #     "role": "assistant",
        #     "content": "The Riemann-Roch theorem was named after Bernhard Riemann and Niels Henrik Abel."
        # }
    ]
    
    # config_get = Config.parse_file('/home/kyle_m/QueryLake_Development/QueryLakeBackend/config.json')
    # # print(config_get.models)

    # model_get : Model = config_get.models[0]

    # def token_counter(string_in):
    #     return 1

    # new_prompt = construct_chat_history(model_get, token_counter, chat_history, 1000)

    # print(new_prompt)
    
    
    