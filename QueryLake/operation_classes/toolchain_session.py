import os, json

os.chdir(os.path.dirname(os.path.realpath(__file__)))

from QueryLake.typing.toolchains import *
from ..api.single_user_auth import get_user
from sqlmodel import Session, select, and_, not_
from sqlalchemy.sql.operators import is_
# from ..database import sql_db_tables
# from ..api import *

from ..models.langchain_sse import ThreadedGenerator
from copy import deepcopy
from time import sleep
from ..api.hashing import random_hash
# from ..toolchain_functions import toolchain_node_functions
# from fastapi import UploadFile
# from sse_starlette.sse import EventSourceResponse
# import asyncio
# from threading import Thread
# from chromadb.api import ClientAPI
# import time
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

    def reset_everything(self):
        """
        Reset everything, as if the toolchain session had just been created.
        Only the toolchain object itself is unaffected.
        """
        self.entry_called = False
        self.node_carousel = {}
        # self.state_arguments = {"session_hash": self.session_hash}
        self.state : dict = self.toolchain.initial_state
        self.firing_queue : Dict[str, Union[Literal[False], dict]] = {}
        self.reset_firing_queue()
    
    async def send_state_notification(self, message, ws : WebSocket = None):
        if not ws is None:
            await ws.send_text(safe_serialize({"state_notification": message}))

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
    
    def create_streaming_callables(self, 
                                   node : toolchainNode, 
                                   assigned_id : str,
                                   ws : WebSocket = None) -> Dict[str, Awaitable[Callable[[Any], None]]]:
        """
        Create a dictionary of callables for streaming outputs to be passed to an API function.
        """
        pass
    
    async def run_node(self,
                       node : toolchainNode, 
                       node_arguments : dict,
                       system_args : dict,
                       user_arguments_from_propagation : dict,
                       ws : WebSocket = None,
                       user_existing_return_arguments : dict = {},
                       use_firing_queue : bool = True) -> Tuple[dict, dict, List[str]]:
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
        
        await self.send_state_notification({
            "type": "node_execution_start",
            "node_id": node.id,
            "inputs": node_arguments,
            "system_args": system_args,
            "user_args": user_arguments_from_propagation,
            "existing_user_return_args": user_existing_return_arguments,
        }, ws)
        
        
        node_inputs, node_outputs, user_return_arguments, firing_targets = {}, {}, {}, []
        
        
        # Initialize the node inputs with existing firing queue if it exists.
        if (use_firing_queue) and (not node.is_event) and (self.firing_queue[node.id] != False):
            node_inputs.update(self.firing_queue[node.id])
        
        
        # Construct the node inputs.
        
        for node_input_arg in node.input_arguments:
            
            if not node_input_arg.initalValue is None:
                node_inputs[node_input_arg.key] = node_input_arg.initalValue
            
            # If active, this give priority to the firing queue initial values.
            # elif node_input_arg.key in node_inputs:
            #     continue
            
            elif node_input_arg.from_user == True:
                assert (node_input_arg.key in user_arguments_from_propagation) or node_input_arg.optional, f"Required input argument \'{node_input_arg.key}\' in node \'{node.id}\' not found in function parameters"
                
                if node_input_arg.key in user_arguments_from_propagation:
                    node_inputs[node_input_arg.key] = user_arguments_from_propagation[node_input_arg.key]
            
            elif not node_input_arg.from_state is None:
                # For now, optionality will not be supported for state arguments. May change in the future.
                assert node_input_arg.key in self.state, f"State argument \'{node_input_arg.key}\' not provided while firing node {node.id} \n{self.state}"
                node_inputs[node_input_arg.key] = self.state[node_input_arg.key]
            
            elif node_input_arg.from_server:
                assert node_input_arg.key in system_args, f"Server argument \'{node_input_arg.key}\' not provided while firing node {node.id}"
                node_inputs[node_input_arg.key] = system_args[node_input_arg.key]
            
            else:
                assert node_input_arg.key in node_arguments, f"Node argument \'{node_input_arg.key}\' not provided while firing node {node.id}"
                node_inputs[node_input_arg.key] = node_arguments[node_input_arg.key]
                print("Got input key from node arguments: ", node_input_arg.key, node_arguments[node_input_arg.key])
        
        await self.send_state_notification({
            "type": "node_execution_created_inputs",
            "node_id": node.id,
            "inputs": node_inputs,
        }, ws)
        
        
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
            node_firing_id = random_hash()
            stream_callables = self.create_streaming_callables(node, node_firing_id, ws)
            get_function = self.function_getter(node.api_function)
            node_outputs = await run_function_safe(get_function, {**node_inputs, "stream_callables": stream_callables})
        
        state_kwargs = {
            "toolchain_state" : self.state,
            "node_inputs_state" : node_inputs,
            "node_outputs_state" : node_outputs
            # "branching_state" : branching_state       # Figure out branches later
        }
        
        # TODO: is shallow copy sufficient?
        previous_state = self.state.copy()
        
        
        
        
        await self.send_state_notification({"type": "1"}, ws)
        
        for feed_map in node.feed_mappings:
            
            await self.send_state_notification({"type": "2"})
            await self.send_state_notification({
                "type": "node_execution_running_feed_map",
                "node_id": node.id,
                "feed_map": feed_map
            }, ws)
            
            if hasattr(feed_map, "value") or "value" in feed_map:
                initial_value = feed_map.value
                print("\n\n\n\n\n\n\n\n\n\n GOT VALUE \n\n\n\n\n\n\n\n\n\n")
                await self.send_state_notification({
                    "type": "node_execution_got_init_value",
                    "value": initial_value,
                }, ws)
            else:
                
                # TODO: I think this is the source of our error. The get_value_obj_global function is limited in scope.
                value_obj_not_run = [getattr(feed_map, attr_check) for attr_check in [
                    "getFromOutputs", "getFrom", "getFromState", "getFromInputs"
                ] if hasattr(feed_map, attr_check)]     # There are multiple flexible types here for syntax.
                
                print("\n\n\n\n\n\n\n\n\n\n GOT VALUE \n\n\n\n\n\n\n\n\n\n")
                print("\n\n\n\n\n\n\n\n\n\n value_obj_not_run  ", value_obj_not_run, "\n\n\n\n\n\n\n\n\n\n")
                
                
                initial_value = get_value_obj_global(value_obj_not_run[0], **state_kwargs)
                
                await self.send_state_notification({
                    "type": "node_execution_got_val_obj",
                    "value_obj_type": str(type(value_obj_not_run[0])),
                    "value": initial_value,
                }, ws)

            if feed_map.destination == "<<USER>>":
                user_return_arguments = run_sequence_action_on_object(user_return_arguments, sequence=feed_map.sequence, provided_object=initial_value, **state_kwargs)
            elif feed_map.destination == "<<STATE>>":
                self.state = run_sequence_action_on_object(self.state, sequence=feed_map.sequence, provided_object=initial_value, **state_kwargs)
                
                append_state, update_state = dict_diff_append_and_update(self.state.copy(), previous_state.copy())
                delete_state = dict_diff_deleted(previous_state.copy(), self.state.copy())
                
                
                if any([(delete_state != []), (append_state != {}), (update_state != {})]):
                    await self.send_state_notification({
                        "type": "state_update",
                        "delete": delete_state,
                        "append": append_state,
                        "update": update_state
                    }, ws)
                
            else:
                # TODO: Revisit this later. I'm not clear on the logic and there is massive room for bugs here.
                
                if self.firing_queue[feed_map.destination] is False and isinstance(self.firing_queue[feed_map.destination], bool):
                    print("Previous firing queue entry was False. Creating new entry.", self.firing_queue[feed_map.destination])
                    
                    self.firing_queue[feed_map.destination] = {}
                
                if not feed_map.sequence is None:
                    await self.send_state_notification({
                        "type": "node_execution_running_sequence_action",
                        "node_id": node.id,
                        "initial_value": initial_value,
                        "sequence": feed_map.sequence,
                        "target": self.firing_queue[feed_map.destination],
                        **state_kwargs
                    }, ws)
                
                result_of_sequence = run_sequence_action_on_object(
                    self.firing_queue[feed_map.destination].copy(), 
                    sequence=feed_map.sequence, 
                    provided_object=initial_value, 
                    **state_kwargs
                )
                
                print(result_of_sequence)
                
                await self.send_state_notification({"type": "node_execution_running_sequence_action_result", "result": result_of_sequence}, ws)
                self.firing_queue[feed_map.destination].update(result_of_sequence)
                self.firing_queue[feed_map.destination].update({"bogus_entry": "bogus_value"})
                
                
                if not feed_map.sequence is None:
                    await self.send_state_notification({
                        "type": "node_execution_running_sequence_action_complete",
                        "node_id": node.id,
                        "sequence": feed_map.sequence,
                        "target": self.firing_queue[feed_map.destination]
                    }, ws)
                
                if not feed_map.store:
                    firing_targets.append(feed_map.destination)
        
        self.firing_queue[node.id] = False

        await self.send_state_notification({
            "type": "node_completion",
            "node_id": node.id,
            **state_kwargs
        }, ws)
        
        user_existing_return_arguments.update(user_return_arguments)
        
        return user_existing_return_arguments, firing_targets #, split_targets

    async def run_node_then_forward(self, 
                                    node : Union[str, toolchainNode], 
                                    node_arguments : dict,
                                    user_provided_arguments : dict,
                                    system_args : dict, 
                                    ws : WebSocket = None, 
                                    # no_split_inputs : bool = False,
                                    clear_firing_queue : bool = True,
                                    user_return_arguments : dict = {}):
        
        if isinstance(node, str):
            node : toolchainNode = self.node_carousel[node]
        
        await self.send_state_notification({
            "type": "node_execution_start",
            "node_id": node.id
        }, ws)
        
        user_return_arguments, firing_targets = await self.run_node(
            node=node, 
            node_arguments=node_arguments,
            system_args=system_args,
            user_arguments_from_propagation=user_provided_arguments,
            ws=ws,
            user_existing_return_arguments=user_return_arguments,
            use_firing_queue=True
        )
        
        for node_id in firing_targets:
            user_return_arguments = await self.run_node_then_forward(self.nodes_dict[node_id], 
                                                                     self.firing_queue[node_id], 
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
        await self.send_state_notification({
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
        await self.send_state_notification(send_information, ws)

    def dump(self):
        return {
            "title": self.state["title"],
            "toolchain_id": self.toolchain_id,
            "state_arguments": self.state,
            "session_hash_id": self.session_hash,
            "firing_queue": self.firing_queue
        }
    
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
