import os, json

os.chdir(os.path.dirname(os.path.realpath(__file__)))

from QueryLake.typing.toolchains import *
from ..api.single_user_auth import get_user
from sqlmodel import Session, select, and_, not_
from sqlalchemy.sql.operators import is_

from copy import deepcopy, copy
from time import sleep, time
from ..api.hashing import random_hash
from ..api.document import get_file_bytes, get_document_secure
from ..api.user_auth import get_user_private_key
from ..database.encryption import aes_encrypt_zip_file, aes_decrypt_zip_file
# from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi import WebSocket
import zipfile
from ..function_run_clean import run_function_safe
from ..typing.config import AuthType, getUserType
from typing import Callable, Any, List, Dict, Union, Awaitable

from ..misc_functions.toolchain_state_management import *
from ..typing.toolchains import *

class ToolchainSession():
    SPECIAL_NODE_FUNCTIONS = ["<<ENTRY>>", "<<EVENT>>"]
    SPECIAL_ARGUMENT_ORIGINS = ["<<SERVER_ARGS>>", "<<USER>>", "<<STATE>>"]

    def __init__(self,
                 toolchain_id,
                 toolchain_pulled : Union[dict, ToolChain], 
                 function_getter : Callable[[str], Union[Awaitable[Callable], Callable]],
                 session_hash,
                 author : str) -> None:
        """
        This should initialize the toolchain session with the toolchain id and the toolchain object itself.
        """
        
        self.author = author
        self.session_hash = session_hash
        self.toolchain_id = toolchain_id
        
        if isinstance( toolchain_pulled, dict ):
            self.toolchain : ToolChain = ToolChain( **toolchain_pulled )
        else:
            self.toolchain : ToolChain = toolchain_pulled
        
        self.nodes_dict : Dict[str, toolchainNode] = { node.id : node for node in self.toolchain.nodes }
        
        self.function_getter : Callable[[str], Union[Awaitable[Callable], Callable]] = function_getter
        self.reset_everything()
        self.log : List[str] = []

    def reset_everything(self):
        """
        Reset everything, as if the toolchain session had just been created.
        Only the toolchain object itself is unaffected.
        """
        self.entry_called = False
        self.node_carousel = {}
        # self.state_arguments = {"session_hash": self.session_hash}
        self.state : dict = deepcopy(self.toolchain.initial_state)
        
        if not "title" in self.state:
            self.state["title"] = self.toolchain.name
        
        self.firing_queue : Dict[str, Union[Literal[False], dict]] = {}
        self.reset_firing_queue()
    
    def log_event(self, header: str, event : dict):
        tab_make = "\t"*1
        
        string_make = header + "\n" +  tab_make + safe_serialize(event, indent=4).replace("\n", "\n"+tab_make)
        assert isinstance(string_make, str), f"Log event is not a string: {string_make}"
        self.log.append(string_make)
    
    async def send_websocket_msg(self, message : json, ws : WebSocket = None):
        # self.log_event("WEBSOCKET MESSAGE", message)
        
        if not ws is None:
            await ws.send_text(safe_serialize(message))

    def reset_firing_queue(self):
        self.firing_queue : Dict[str, Union[Literal[False], dict]] = {
            key : False for key, node in self.nodes_dict.items() if not (node.is_event)
        }
    
    def assert_split_inputs(self):
        for node_id in self.node_carousel.keys():
            self.node_carousel[node_id]["requires_split_inputs"] = False
            for arg_entry in self.node_carousel[node_id]["arguments"]:
                if "merge_parallel_outputs" in arg_entry and arg_entry["merge_parallel_outputs"]:
                    self.node_carousel[node_id]["requires_split_inputs"] = True
                    break

    def assert_event_path(self, node):
        self.node_carousel[node["id"]]["requires_event"] = True
        for entry in node["feed_to"]:
            if entry["destination"] in ToolchainSession.SPECIAL_ARGUMENT_ORIGINS:
                continue
            else:
                target_node = self.node_carousel[entry["destination"]]
                if self.node_carousel[target_node["id"]]["requires_event"]:
                    continue
                self.assert_event_path(target_node)
    
    async def create_streaming_callables(self, 
                                         node : toolchainNode,
                                         state_args : dict,
                                         ws : WebSocket = None) -> Tuple[dict, Dict[str, Awaitable[Callable[[Any], None]]]]:
        """
        Create a dictionary of callables for streaming outputs to be passed to an API function.

        Outputs new state and a dictionary of callables.
        """
        
        streaming_callables : Dict[str, Awaitable[Callable[[Any], None]]] = {}
        
        for feed_map in node.feed_mappings:
            
            if not feed_map.stream:
                continue
            
            assert feed_map.destination == "<<STATE>>", f"Streaming feed map \'{feed_map.id}\' does not have destination <<STATE>>"
            
            random_streaming_id = random_hash()[:16]
            
            # self.state = await self.run_feed_map_on_object(target_object=self.state, feed_map=feed_map, **state_args)    
            self.state, new_routes = run_sequence_action_on_object(self.state, 
                                                                   sequence=feed_map.sequence, 
                                                                   provided_object=feed_map.stream_initial_value, 
                                                                   **state_args, 
                                                                   return_provided_object_routes=True)
            
            await self.send_websocket_msg({
                "type": "streaming_output_mapping",
                "node_id": node.id,
                "stream_id": random_streaming_id,
                "routes": new_routes
            }, ws)
            
            async def update_state_with_streaming_output(value):
                # self.log_event("STREAM UPDATE", {
                #     "type": "stream_output",
                #     "stream_id": random_streaming_id,
                #     "value": value
                # })
                await self.send_websocket_msg({
                    "type": "stream_output",
                    "stream_id": random_streaming_id,
                    "value": value
                }, ws)
                for route_direct in new_routes:
                    # previous_state_tmp = deepcopy(self.state)
                    
                    self.state = insert_in_static_route_global(self.state, route_direct, value, **state_args, append=True)
                    # self.log_event("STREAM UPDATE AT ROUTE", {
                    #     "type": "stream_output",
                    #     "stream_id": random_streaming_id,
                    #     "route": route_direct,
                    #     "value": value,
                    #     "previous_state": previous_state_tmp,
                    #     "new_state": self.state
                    # })
            
            # Use the first route index as the id
            # Assuming only top-level results are streamed.
            callable_index = feed_map.getFromOutputs.route[0]
            
            streaming_callables[callable_index] = update_state_with_streaming_output

        self.log_event("STREAM CALLABLES CREATED", {
            "streaming_callables": streaming_callables,
            "state": self.state
        })
        
        return self.state, streaming_callables
    
    def construct_node_run_inputs(self, 
                                  node : toolchainNode, 
                                  node_arguments : dict, 
                                  user_provided_arguments : dict, 
                                  system_args: dict,
                                  use_firing_queue : bool = True) -> dict:
        """
        Construct the inputs for running a node.
        """
        node_inputs = {}
        self.log_event("CONSTRUCTING NODE INPUTS WITH", {
            "node_id": node.id,
            "node_arguments": node_arguments,
            "user_provided_arguments": user_provided_arguments,
            "system_args": system_args,
            "state_args": self.state,
        })
        
        state_copied_reference = deepcopy(self.state)
        
        if (use_firing_queue) and (not node.is_event) and (self.firing_queue[node.id] != False):
            node_inputs.update(self.firing_queue[node.id].copy())
        
        for node_input_arg in node.input_arguments:
            
            if not node_input_arg.initalValue is None:
                node_inputs[node_input_arg.key] = node_input_arg.initalValue
                self.log_event("SET INITIAL VALUE", {
                    "value": node_inputs[node_input_arg.key]
                })
            
            # If active, this give priority to the firing queue initial values.
            # elif node_input_arg.key in node_inputs:
            #     continue
            
            elif node_input_arg.from_user == True:
                self.log_event("SEARCHING FOR USER INPUT ARG", {
                    "node_id": node.id,
                    "node_input_arg": node_input_arg,
                    "user_args": user_provided_arguments
                })
                
                assert (node_input_arg.key in user_provided_arguments) or node_input_arg.optional, f"Required input argument \'{node_input_arg.key}\' in node \'{node.id}\' not found in function parameters"
                
                if node_input_arg.key in user_provided_arguments:
                    node_inputs[node_input_arg.key] = user_provided_arguments[node_input_arg.key]
                    self.log_event("FOUND USER INPUT ARG", {
                        "value": node_inputs[node_input_arg.key]
                    })
            
            elif not node_input_arg.from_state is None:
                # For now, optionality will not be supported for state arguments. May change in the future.
                self.log_event("SEARCHING FOR STATE INPUT ARG", {
                    "node_id": node.id,
                    "node_input_arg": node_input_arg,
                    "state": self.state
                })
                assert node_input_arg.key in state_copied_reference, f"State argument \'{node_input_arg.key}\' not provided while firing node {node.id} \n{state_copied_reference}"
                node_inputs[node_input_arg.key] = state_copied_reference[node_input_arg.key]
                self.log_event("FOUND STATE INPUT ARG", {
                    "value": node_inputs[node_input_arg.key]
                })
                
                
            elif node_input_arg.from_server:
                assert node_input_arg.key in system_args, f"Server argument \'{node_input_arg.key}\' not provided while firing node {node.id}"
                node_inputs[node_input_arg.key] = system_args[node_input_arg.key]
            
            else:
                self.log_event("SEARCHING FOR NODE INPUT ARG", {
                    "node_id": node.id,
                    "node_input_arg": node_input_arg,
                    "node_arguments": node_arguments
                })
                assert node_input_arg.key in node_arguments, f"Node argument \'{node_input_arg.key}\' not provided while firing node {node.id}"
                node_inputs[node_input_arg.key] = node_arguments[node_input_arg.key]
                self.log_event("FOUND NODE INPUT ARG", {
                    "value": node_inputs[node_input_arg.key]
                })
                
        self.log_event("CREATED NODE INPUTS", {
            "node_id": node.id,
            "node_inputs": node_inputs
        })
        
        return node_inputs
    
    async def run_feed_map_on_object(self,
                               feed_map : feedMapping,
                               target_object : dict,
                               node_arguments : dict, 
                               user_provided_arguments : dict, 
                               system_args: dict,
                               node_outputs : dict,
                               ws : WebSocket = None) -> dict:
        """
        Execute a feed mapping on an object.
        """
        
        state_args = {
            "toolchain_state": self.state, 
            "node_inputs_state": node_arguments, 
            "node_outputs_state": node_outputs
        }
        
        if hasattr(feed_map, "value") or "value" in feed_map:
            initial_value = feed_map.value
            print("\n\n\n\n\n\n\n\n\n\n GOT VALUE \n\n\n\n\n\n\n\n\n\n")
            self.log_event("INITIAL VALUE RETRIEVED", {
                "value": initial_value,
            })
        else:
            
            # TODO: I think this is the source of our error. The get_value_obj_global function is limited in scope.
            value_obj_not_run = [getattr(feed_map, attr_check) for attr_check in [
                "getFromOutputs", "getFrom", "getFromState", "getFromInputs"
            ] if hasattr(feed_map, attr_check)]     # There are multiple flexible types here for syntax.
            
            self.log_event("SEARCHING FOR INITIAL VALUE WITH", {
                "value_obj": value_obj_not_run[0],
                "state_kwargs": state_args
            })
            
            initial_value = get_value_obj_global(value_obj_not_run[0], **state_args)
            
            self.log_event("INITIAL VALUE EVALUATED", {
                "type": "node_execution_got_val_obj",
                "value_obj_type": str(type(value_obj_not_run[0])),
                "value": initial_value,
            })
        
        if not feed_map.route is None:
            target_object = insert_in_static_route_global(target_object, feed_map.route, initial_value, **state_args)
            
            self.log_event("INSERT IN ROUTE PERFORMED", {
                "target_result": target_object,
            })
            
        else:
            target_object = run_sequence_action_on_object(target_object, sequence=feed_map.sequence, provided_object=initial_value, **state_args)

            self.log_event("SEQUENCE ACTIONS PERFORMED", {
                "target_result": target_object,
            })
        
        
        return target_object
    
    async def notify_ws_state_difference(self, previous_state, current_state, ws : WebSocket = None):
        """
        Notify the websocket of the difference between the previous and current state.
        """
        self.log_event("STATE DIFF CALL ENTERED", {})
        
        append_routes, append_state, update_state = dict_diff_append_and_update(current_state, previous_state)
        delete_state = dict_diff_deleted(previous_state, current_state)
        
        update_state_values = {
            "append_routes": append_routes,
            "append_state": append_state,
            "update_state": update_state,
            "delete_state": delete_state
        }
        
        self.log_event("STATE DIFFERENCE NOTIFICATION", {
            "state_diff": update_state_values
        })
        
        if any([(len(v) > 0) for v in update_state_values.values()]):
            await self.send_websocket_msg({
                "type": "state_diff",
                **{k: v for k, v in update_state_values.items() if len(v) > 0}
            }, ws)
            
    
    async def run_node(self,
                       node : toolchainNode, 
                       node_arguments : dict,
                       system_args : dict,
                       user_provided_arguments : dict, # These persist through propagation from an event call.
                       ws : WebSocket = None,
                       user_existing_return_arguments : dict = {},
                       use_firing_queue : bool = True) -> Tuple[dict, List[Tuple[str, dict]]]:
        """
        TODO
        This has to collect the arguments for the node, then run the node and/or forward them.
        It also has to manage forwarding in the firing queue.
        
        
        # IMPORTANT CAVEATS
        
        *   It should return a list of node ids that need to be fired based on the feed mappings,
            ommitting if `store` is True.
        *   Handle stream outputs by creating a local function to send to the target api function
            if `stream` is true.
        *   Run sequenceActions in feedmappings.
        """
        
        await self.send_websocket_msg({
            "type": "node_execution_start",
            "node_id": node.id
        }, ws)
        
        early_state_reference = deepcopy(self.state)
        
        node_outputs, user_return_arguments, firing_targets = {}, {}, []
        node_argument_templates : Dict[str, dict] = {}
        node_follow_up_firing_queue : List[Tuple[str, dict]] = []
        
        node_inputs = copy(self.construct_node_run_inputs(node, node_arguments, user_provided_arguments, system_args, use_firing_queue))
        
        await self.send_websocket_msg({
            "type": "node_execution_created_inputs",
            "node_id": node.id,
            "inputs": node_inputs,
        }, ws)
        
        state_kwargs = {
            "toolchain_state" : early_state_reference,
            "node_inputs_state" : node_inputs,
            "node_outputs_state" : {}
            # "branching_state" : branching_state       # Figure out branches later
        }
        
        # Fire the node or assert that it is an event node and the mappings are valid.
        
        if node.is_event:
            assert all([
                not any([
                    hasattr(feed_map, "getFromOutputs"),
                    hasattr(feed_map, "getFrom") and feed_map.getFrom.type == "getNodeOutput", 
                ]) \
                for feed_map in node.feed_mappings
            ]), f"Event node \'{node.id}\' has mappings grabbing outside node outputs, which do not exist since it is an event."
            node_outputs = {}
        else:
            self.state, stream_callables = await self.create_streaming_callables(node, state_kwargs, ws)
            
            self.log_event("STATE DIFF CALL STARTED", {})
            await self.notify_ws_state_difference(early_state_reference, self.state, ws)
            
            get_function = self.function_getter(node.api_function)
            node_outputs = await run_function_safe(get_function, {**node_inputs, "stream_callables": stream_callables})
        
        state_kwargs.update({"node_outputs_state" : node_outputs})
        
        # TODO: is shallow copy sufficient?
        
        for feed_map in node.feed_mappings:
            
            feed_map_args = {
                "feed_map": feed_map,
                "node_arguments": node_inputs, # Changed from node_arguments
                "user_provided_arguments": user_provided_arguments,
                "system_args": system_args,
                "node_outputs": node_outputs,
                "ws": ws
            }
            
            # Feed to user case.
            if feed_map.destination == "<<USER>>":
                user_return_arguments = await self.run_feed_map_on_object(target_object=user_return_arguments, **feed_map_args)
            
            # Feed to state case.
            elif feed_map.destination == "<<STATE>>":
                
                if feed_map.stream:
                    continue
                self.state = await self.run_feed_map_on_object(target_object=self.state, **feed_map_args)
                
                # append_state, update_state = dict_diff_append_and_update(self.state.copy(), previous_state.copy())
                # delete_state = dict_diff_deleted(previous_state.copy(), self.state.copy())
                
                
                self.log_event("UPDATED STATE WITH FEED MAP", {
                    "node_id": node.id,
                    "toolchain_state": self.state,
                })
            else:
                # TODO: Revisit this later. I'm not clear on the logic and there is massive room for bugs here.
                
                
                if feed_map.store:
                    # Initialize firing queue args if they're not already used.
                    if self.firing_queue[feed_map.destination] is False and isinstance(self.firing_queue[feed_map.destination], bool):
                        self.firing_queue[feed_map.destination] = {}
                    self.firing_queue[feed_map.destination] = await self.run_feed_map_on_object(target_object=self.firing_queue[feed_map.destination], **feed_map_args)
                else:
                    node_argument_templates[feed_map.destination] = await self.run_feed_map_on_object(target_object=node_argument_templates.get(feed_map.destination, {}), **feed_map_args)
                    
                    
                    # TODO: Need to handle split inputs here.
        
        for node_target_id, node_target_args in node_argument_templates.items():
            node_follow_up_firing_queue.append((node_target_id, node_target_args))
        
        self.firing_queue[node.id] = False

        await self.send_websocket_msg({
            "type": "node_completion",
            "node_id": node.id,
            **state_kwargs
        }, ws)
        
        user_existing_return_arguments.update(user_return_arguments)
        
        return user_existing_return_arguments, node_follow_up_firing_queue #, split_targets

    async def run_node_then_forward(self, 
                                    node : Union[str, toolchainNode], 
                                    node_arguments : dict,
                                    user_provided_arguments : dict,
                                    system_args : dict, 
                                    ws : WebSocket = None, 
                                    # no_split_inputs : bool = False,
                                    clear_firing_queue : bool = True,
                                    user_return_arguments : dict = {}):
        
        
        self.log_event("RUN NODE & FORWARD", {
            "node_id": node.id,
            "node_arguments": node_arguments,
            "user_provided_arguments": user_provided_arguments,
            "system_args": system_args,
            "clear_firing_queue": clear_firing_queue,
            "user_return_arguments": user_return_arguments
        })
        
        if isinstance(node, str):
            node : toolchainNode = self.node_carousel[node]
        
        await self.send_websocket_msg({
            "type": "node_execution_start",
            "node_id": node.id
        }, ws)
        
        user_return_arguments, firing_targets = await self.run_node(
            node=node, 
            node_arguments=node_arguments,
            system_args=system_args,
            user_provided_arguments=user_provided_arguments,
            ws=ws,
            user_existing_return_arguments=user_return_arguments,
            use_firing_queue=True
        )
        
        self.log_event("RUN NODE & FORWARD FIRST STEP RESULTS", {
            "node": node,
            "user_return_args": user_return_arguments,
            "firing_targets": firing_targets
        })
        
        await self.send_websocket_msg({
            "type": "node_execution_start_firing_targets",
            "targets": firing_targets
        }, ws)
        for (node_id_target, node_target_arguments) in firing_targets:
            
            # firing_target_input_arguments = self.firing_queue[node_id_target]
            firing_target_input_arguments = node_target_arguments
            
            self.log_event("RUN NODE & FORWARD FIRST STEP RESULTS", {
                "node": node,
                "firing_target_node_id": node_id_target,
                "firing_target_input_arguments": firing_target_input_arguments
            })
            
            user_return_arguments = await self.run_node_then_forward(self.nodes_dict[node_id_target], 
                                                                     firing_target_input_arguments, 
                                                                     user_provided_arguments,
                                                                     system_args,
                                                                     ws,
                                                                     user_return_arguments=user_return_arguments)
        
        return user_return_arguments
        
        
        # for entry in node["feed_to"]:

    async def event_prop(self, 
                         event_id : str,
                         input_parameters : dict, 
                         system_args : dict,
                         ws : WebSocket = None):
        """
        Activate an event node with parameters by id, then propagate forward.
        """
        # print("event_prop")
        target_event = self.nodes_dict[event_id]
        
        
        
        result = await self.run_node_then_forward(target_event,
                                                  {}, 
                                                  input_parameters, 
                                                  system_args,
                                                  ws=ws)
        await self.send_websocket_msg({
            "type": "event_completion",
            "event_id": event_id,
            "output": result
        }, ws)
        
        return result
    
    async def update_active_node(self, 
                                 node_id,
                                 ws : WebSocket = None, 
                                 arguments : dict = None):
        """
        Send the state of the session as a json.
        """
        # print("update_active_node")
        send_information = {
            "type": "active_node_update",
            "active_node": node_id
        }
        if not arguments is None:
            send_information.update({"arguments": arguments})
        # if not self.session_state_generator is None:
        await self.send_websocket_msg(send_information, ws)

    def dump(self):
        
        return {
            "title": self.state["title"],
            "toolchain_id": self.toolchain_id,
            "state_arguments": self.state,
            "session_hash_id": self.session_hash,
            "firing_queue": self.firing_queue
        }
        
    def write_logs(self):
        if not os.path.exists("toolchain_sessions_logs"):
            os.mkdir("toolchain_sessions_logs")
        
        print("DUMPING LOGS WITH LENGTH", len(self.log))
        
        with open("toolchain_sessions_logs/%.2f_%s.txt" % (time(), self.session_hash), "w") as f:
            f.write("\n\n".join(self.log))
            f.close()
    
    def load(self, data, toolchains_available : Dict[str, ToolChain]):
        
        
        if type(data) is str:
            data = json.loads(data)
        
        self.toolchain_id = data["toolchain_id"]
        self.toolchain = deepcopy(toolchains_available[data["toolchain_id"]])
        self.nodes_dict : Dict[str, toolchainNode] = { node.id : node for node in self.toolchain.nodes }
        
        self.session_hash = data["session_hash_id"]
        self.reset_everything()
        # self.assert_split_inputs()
        self.state = data["state_arguments"]
        self.state["title"] = data["title"]
        self.firing_queue = data["firing_queue"]
