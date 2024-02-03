import os, json
from ..api.single_user_auth import get_user
from sqlmodel import Session, select, and_, not_
from sqlalchemy.sql.operators import is_
from ..database import sql_db_tables
from ..models.langchain_sse import ThreadedGenerator
from copy import deepcopy
from time import sleep
from ..api.hashing import random_hash
from ..toolchain_functions import toolchain_node_functions
from fastapi import UploadFile
from sse_starlette.sse import EventSourceResponse
import asyncio
from threading import Thread
from chromadb.api import ClientAPI
import time
from ..api.document import get_file_bytes, get_document_secure
from ..api.user_auth import get_user_private_key
from ..database.encryption import aes_encrypt_zip_file, aes_decrypt_zip_file
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi import WebSocket
import zipfile
from ..function_run_clean import run_function_safe
from ..typing.config import AuthType, getUserType
from typing import Callable, Any, List, Dict, Union, Awaitable
from ..typing.toolchains import *




server_dir = "/".join(os.path.dirname(os.path.realpath(__file__)).split("/")[:-2])
upper_server_dir = "/".join(os.path.dirname(os.path.realpath(__file__)).split("/")[:-2])+"/"
user_db_path = server_dir+"/user_db/files/"


def retrieve_value_from_dict(input_dict : dict, 
                             directory : Union[str, List[str]]):
    try:
        if isinstance(directory, str):
            return input_dict[directory]
        else:
            current_dict = input_dict
            for entry in directory:
                # print("Checking:", [entry], json.dumps(current_dict))
                current_dict = current_dict[entry]
            # print("Returning:", current_dict)
            return current_dict
    except KeyError:
        # print("Returning KeyError")
        raise KeyError
    except:
        # print("Returning None")
        return None

def dict_path_is_valid(input_dict : dict, directory : Union[str, List[str]]):
    try:
        if isinstance(directory, str):
            return input_dict[directory]
        else:
            current_dict = input_dict
            for entry in directory:
                # print("Checking:", [entry], json.dumps(current_dict))
                current_dict = current_dict[entry]
            # print("Returning:", current_dict)
            return True
    except:
        # print("Returning False")
        return False

def safe_serialize(obj):
  default = lambda o: f"<<non-serializable: {type(o).__qualname__}>>"
  return json.dumps(obj, default=default)




default_toolchain = "chat_session_normal"


class ToolchainSession():
    SPECIAL_NODE_FUNCTIONS = ["<<ENTRY>>", "<<EVENT>>"]
    SPECIAL_ARGUMENT_ORIGINS = ["<<SERVER_ARGS>>", "<<USER>>", "<<STATE>>"]

    def __init__(self, 
                 toolchain_id, 
                 toolchain_pulled : Union[dict, ToolChain], 
                 function_getter : Callable[[Any], Union[Awaitable[Callable], Callable]],
                 session_hash, 
                 author : str) -> None:
        self.author = author
        self.session_hash = session_hash
        self.toolchain_id = toolchain_id
        # print("toolchain_id:", toolchain_id)
        
        if isinstance(toolchain_pulled, dict):
            self.toolchain = ToolChain(**toolchain_pulled)
        
        self.toolchain : ToolChain = deepcopy(toolchain_pulled)
        self.function_getter = function_getter
        self.reset_everything()
        self.assert_split_inputs()

    def reset_everything(self):
        """
        Reset everything, as if the toolchain had just been loaded.
        Only the toolchain json itself is unaffected.
        """
        self.entry_called = False
        self.node_carousel = {}
        self.state_arguments = {"session_hash": self.session_hash}
        self.entry_point = None
        self.entry_point_id = None
        self.event_points = []
        self.event_point_ids = []
        self.processing_lock = False
        self.layer_outputs = {}
        self.return_to_user = {}
        self.queued_node_inputs = {}
        self.firing_queue = {}
        self.reset_firing_queue()


        # Verify the toolchain
        # for entry in self.toolchain["pipeline"]:
            
        #     self.node_carousel[entry["id"]] = entry

        #     self.node_carousel[entry["id"]]["requires_event"] = False
        #     self.node_carousel[entry["id"]]["outputs_stream_to_user"] = False

        #     tmp_stream_outputs = {}
        #     if "output_arguments" in entry:
        #         for output_variable_entry in entry["output_arguments"]:
        #             if "stream" in output_variable_entry and output_variable_entry["stream"] == True:
        #                 tmp_stream_outputs[output_variable_entry["id"]] = True

        #     for feed_mapping in entry["feed_to"]:
        #         break_flag = False
        #         if feed_mapping["destination"] == "<<USER>>":
        #             for sub_mapping in feed_mapping["input"]:
        #                 if sub_mapping["output_argument_id"] in tmp_stream_outputs:
        #                     self.node_carousel[entry["id"]]["outputs_stream_to_user"] = sub_mapping["output_argument_id"]
        #                     break_flag = True
        #                     break
        #         if break_flag:
        #             break
            
        #     # print("Loaded node id %s with \'outputs_stream_to_user\': %s" % (entry["id"], self.node_carousel[entry["id"]]["outputs_stream_to_user"]))

        #     if entry["function"] == "<<ENTRY>>":
        #         self.entry_point = entry
        #         self.entry_point_id = entry["id"]
        #     elif entry["function"] == "<<EVENT>>":
        #         self.event_points.append(entry)
        #         self.event_point_ids.append(entry["id"])
        #     # else:
        #     #     self.function_getter(entry["function"])
        
        # for event_point in self.event_points:
        #     self.assert_event_path(event_point)
        
        # for key, value in self.toolchain["state_initialization"].items():
        #     self.state_arguments[key] = value
        
        # if "title" not in self.state_arguments:
        #     self.state_arguments["title"] = self.toolchain["name"]
        
        # self.session_state_generator = None
        # self.session_state_generator_log = []
    
    async def send_state_notification(self, message, ws : WebSocket = None):
        if not ws is None:
            await ws.send_text(safe_serialize({"state_notification": message}))

    def reset_firing_queue(self):
        for entry in self.toolchain["pipeline"]:
            if not (entry["function"] == "<<ENTRY>>" or entry["function"] == "<<EVENT>>"):
                self.queued_node_inputs[entry["id"]] = {}
                self.firing_queue[entry["id"]] = False
    
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

    def get_node_argument_requirements(self, node : dict):
        return deepcopy(node["arguments"])

    async def run_node_with_dependencies(self,
                                         node : dict, 
                                         function_parameters : dict, 
                                         system_args : dict,
                                         ws : WebSocket = None,
                                         use_previous_outputs : bool = False):
        
        # print("Running node %s with args %s" % (node["id"], safe_serialize(function_parameters)))
        node_arguments_possible = self.get_node_argument_requirements(node)
        node_argument_ids = [entry["argument_name"] for entry in node_arguments_possible]
        node_arguments_required = [entry for entry in node_arguments_possible if not ("optional" in entry and entry["optional"])]
        node_arguments_required_ids = [entry["argument_name"] for entry in node_arguments_required]

        filter_args = {}

        sum_list = 0
        if node["function"] not in ToolchainSession.SPECIAL_NODE_FUNCTIONS:
            for key, value in self.queued_node_inputs[node["id"]].items():
                filter_args[key] = value
                sum_list += 1
        
        for key, value in function_parameters.items():
            if key in node_argument_ids:
                filter_args[key] = value
                if key in node_arguments_required_ids:
                    sum_list += 1

        for key, value in system_args.items():
            if key in node_argument_ids:
                filter_args[key] = value
                sum_list += 1

        # print("Filter args:", filter_args)

        for argument in node_arguments_possible:
            if argument["argument_name"] in filter_args:
                continue
            argument["satisfied"] = False
            if argument["argument_name"] in filter_args:
                argument["satisfied"] = True
            elif "origin" in argument and argument["origin"] == "<<STATE>>":
                assert argument["argument_name"] in self.state_arguments, f"Required state argument {argument['argument_name']} could not be found."
                arg_name = argument["argument_name"] if not ("rename_to" in argument) else argument["rename_to"]
                filter_args[arg_name] = self.state_arguments[argument["argument_name"]]
            else:
                if "origin" in argument and argument["origin"] in self.layer_outputs and use_previous_outputs:
                    filter_args[argument["argument_name"]] = self.layer_outputs[argument["origin"]][argument["argument_name"]]
                # elif use_previous_outputs:
                #     node_source = self.node_carousel[argument["origin"]]
                #     run_arguments = self.run_node_with_dependencies(node_source, function_parameters, use_previous_outputs=use_previous_outputs)
                #     filter_args[argument["argument_name"]] = run_arguments[argument["argument_name"]]
            sum_list += 1
        
        assert sum_list >= len(node_arguments_required), f"Failed to satisfy required arguments for node {node['id']}"

        if node["function"] not in ToolchainSession.SPECIAL_NODE_FUNCTIONS:
            # print("Calling %s" % (node["function"]))
            function_target = self.function_getter(node["function"])
            run_arguments = await run_function_safe(function_target, filter_args)
            # run_arguments = function_target(**filter_args)
            if not type(run_arguments) is dict:
                run_arguments = {"result": run_arguments}

            self.layer_outputs[node["id"]] = run_arguments
        else:
            run_arguments = function_parameters

        user_return_args = {}

        split_targets = []

        recombination_targets = []

        if self.node_carousel[node["id"]]["outputs_stream_to_user"] != False:
            run_arguments[self.node_carousel[node["id"]]["outputs_stream_to_user"]] = self.await_generator_completion(run_arguments[self.node_carousel[node["id"]]["outputs_stream_to_user"]] )

        for entry in node["feed_to"]:
            destination_merge_mapping_flag = False

            if entry["destination"] == "<<STATE>>":
                state_return = {}

            for feed_map in entry["input"]:
                assert "value" in feed_map or \
                        ("output_argument_id" in feed_map and (dict_path_is_valid(run_arguments, feed_map["output_argument_id"]) or \
                         dict_path_is_valid(function_parameters, feed_map["output_argument_id"]))) or \
                        ("optional" in feed_map and feed_map["optional"]), \
                        f"argument {feed_map['output_argument_id']} could not be found"+json.dumps(run_arguments)+", "+json.dumps(function_parameters)
                
                target_arg = feed_map["target_argument"]
                input_value = None

                if "value" in feed_map:
                    input_value = feed_map["value"]
                elif "input_argument_id" in feed_map:
                    if dict_path_is_valid(function_parameters, feed_map["input_argument_id"]):
                        input_value = retrieve_value_from_dict(function_parameters, feed_map["input_argument_id"])
                elif dict_path_is_valid(run_arguments, feed_map["output_argument_id"]):
                    # if feed_map["output_argument_id"] in run_arguments:
                    input_value = retrieve_value_from_dict(run_arguments, feed_map["output_argument_id"])
                elif dict_path_is_valid(function_parameters, feed_map["output_argument_id"]):
                    # if feed_map["output_argument_id"] in function_parameters:
                    # input_value = function_parameters[feed_map["output_argument_id"]]
                    input_value = retrieve_value_from_dict(function_parameters, feed_map["output_argument_id"])
                if input_value is None:
                    continue
                # print("Target Arg", target_arg, type(input_value), str(type(input_value)))

                if entry["destination"] == "<<STATE>>":
                    # Need to await generator here.
                    # if type(input_value) is ThreadedGenerator:
                    #     input_value = self.await_generator_completion(input_value)
                    state_return[target_arg] = input_value
                elif entry["destination"] == "<<USER>>":
                    # In the case of a generator, run create_generator. No need to wait.
                    if type(input_value) is ThreadedGenerator:
                        # await self.create_generator(node["id"], target_arg, input_value)
                        # user_return_args[target_arg] = self.await_generator_completion(input_value)
                        pass
                    else:
                        user_return_args[target_arg] = input_value
                else:
                    destination_merge_mapping_flag = destination_merge_mapping_flag or self.check_if_input_combination_required(entry["destination"], feed_map["target_argument"])

                    # In the case of a generator, await the output and use that instead.
                    # if type(input_value) is ThreadedGenerator:
                    #     input_value = self.await_generator_completion(input_value)
                    if destination_merge_mapping_flag:
                        if feed_map["target_argument"] not in self.queued_node_inputs[entry["destination"]]:
                            self.queued_node_inputs[entry["destination"]][feed_map["target_argument"]] = []
                        self.queued_node_inputs[entry["destination"]][feed_map["target_argument"]].append(input_value)
                    else:
                        self.queued_node_inputs[entry["destination"]][feed_map["target_argument"]] = input_value  

                    if "split_outputs" in entry and entry["split_outputs"]:
                        self.queued_node_inputs[entry["destination"]][feed_map["target_argument"]] = [e for e in input_value]
                        if not destination_merge_mapping_flag:
                            # print("%s attempts to add via split target %s" % (node["id"], entry["destination"]))
                            split_targets.append({"node_id": entry["destination"], "split_argument_name": feed_map["target_argument"]})
                        else:
                            recombination_targets.append(entry["destination"])

            if entry["destination"] == "<<STATE>>":
                await self.special_state_action(node["id"], entry["action"], entry["target_value"], state_return)

            if not destination_merge_mapping_flag and \
                entry["destination"] not in ToolchainSession.SPECIAL_ARGUMENT_ORIGINS and \
                not ("store" in entry and entry["store"]):
                # print("%s attempts to add %s" % (node["id"], entry["destination"]))
                self.attempt_to_add_to_firing_queue(entry["destination"])
                    # else:
                        
        self.firing_queue[node["id"]] = False

        await self.send_state_notification({
            "type": "node_completion",
            "node_id": node["id"],
            "outputs": run_arguments
        }, ws)
        return run_arguments, user_return_args, split_targets, recombination_targets

    def attempt_to_add_to_firing_queue(self, node_id):
        # print("Attempting to add %s" % (node_id))
        required_args = []
        for entry in self.node_carousel[node_id]["arguments"]:
            if entry["origin"] not in ToolchainSession.SPECIAL_ARGUMENT_ORIGINS and not ("optional" in entry and entry["optional"]):
                required_args.append(entry["argument_name"])
        
        required_args = sorted(list(set(required_args)))
        available_args = sorted(list(set(list(self.queued_node_inputs[node_id].keys()))))
        # print(required_args)
        # print(available_args)
        for arg in required_args:
            if arg not in required_args:
                # print("Input args not satisfied")
                # print(available_args, "vs", required_args)
                return
        # if available_args != required_args:

        self.firing_queue[node_id] = True

    def check_if_input_combination_required(self, node_id, input_argument) -> bool:
        relevant_argument = None
        for arg_entry in self.node_carousel[node_id]["arguments"]:
            if arg_entry["argument_name"] == input_argument:
                relevant_argument = arg_entry
                break
        if relevant_argument is None:
            return False
        return ("merge_parallel_outputs" in relevant_argument and relevant_argument["merge_parallel_outputs"])

    async def special_state_action(self, 
                                   action_origin_node_id, 
                                   state_action, 
                                   target_arg, 
                                   input_args,
                                   ws : WebSocket = None):
        """
        Router function for state actions. Multiple are necessary because of chat
        history appending and formatting.
        """
        assert state_action in [
            "append_dict",
            "set_state_value"
        ], f"state action \'{state_action}\' not found"

        key_get = list(input_args.keys())[0]
        if key_get == "<<VALUE_OVERWRITE_DICT>>":
            input_args = input_args[key_get]
        
        action = getattr(self, state_action)
        result = action(**{
            "target_arg": target_arg,
            "input_args": input_args
        })
        await self.notify_state_update(action_origin_node_id, target_arg, result, ws)

    def append_dict(self, target_arg, input_args : dict):
        """
        Append provided content to state argument with OpenAI's message format, with the role of assistant.
        """
        # print("append_model_response_to_chat_history")
        assert target_arg in self.state_arguments, f"State argument \'{target_arg}\' not found"
        self.state_arguments[target_arg].append(input_args)
        return input_args

    def set_state_value(self, target_arg, input_args : dict):
        """
        Set a provided state value to the input.
        """
        self.state_arguments[target_arg] = input_args

    async def run_node_then_forward(self, 
                                    node, 
                                    parameters : dict,
                                    system_args : dict, 
                                    ws : WebSocket = None,
                                    user_return_args = {}, 
                                    is_event : bool = False, 
                                    no_split_inputs : bool = False,
                                    clear_firing_queue : bool = True,
                                    display_tab_count : int = 0):
        # print("  "*display_tab_count + "Running node: %s" % (node["id"]))
        await self.send_state_notification({
            "type": "node_execution_start",
            "node_id": node["id"]
        }, ws)
        # if node["outputs_stream_to_user"] != False:
        #     self.queued_node_inputs[node["id"]].update(parameters)
        #     relevant_mappings = []
        #     for feed_mapping in node["feed_to"]:
        #         if feed_mapping["destination"] == "<<STATE>>":
        #             relevant_mappings.append(feed_mapping)
        #     await self.send_state_notification(json.dumps({
        #         "type": "stream_output_pause",
        #         "node_id": node["id"],
        #         "stream_argument": node["outputs_stream_to_user"],
        #         "mapping": relevant_mappings
        #     }))
        #     return None
        functions_fed_to = []
        for feed_mapping in node["feed_to"]:
            if feed_mapping["destination"] not in ToolchainSession.SPECIAL_NODE_FUNCTIONS:
                functions_fed_to.append(feed_mapping["destination"])

        run_arguments, user_return_args_get, split_targets, recombo_targets = await self.run_node_with_dependencies(node, parameters, system_args, ws)
        # self.send_state_notification(json.dumps({
        #     "type": "node_completion",
        #     "node_id": node["id"],
        #     "outputs": run_arguments
        # }))
        
        user_return_args.update(user_return_args_get)
        recombo_targets_review = {}
        for split_entry in split_targets:
            # print("Got split target:", split_entry)
            get_split_arguments = deepcopy(self.queued_node_inputs[split_entry["node_id"]][split_entry["split_argument_name"]])
            # print("Got split argument:", get_split_arguments)


            for value in get_split_arguments:
                # print("Firing node id from split entry %s" % (split_entry["node_id"]))
                self.queued_node_inputs[split_entry["node_id"]][split_entry["split_argument_name"]] = value
                # run_arguments
                _, recombo_targets = await self.run_node_then_forward(self.node_carousel[split_entry["node_id"]], 
                                                                      parameters, 
                                                                      system_args,
                                                                      ws,
                                                                      no_split_inputs=True, 
                                                                      clear_firing_queue=False, 
                                                                      display_tab_count=display_tab_count+1)
                for recombo_id in recombo_targets:
                    recombo_targets_review[recombo_id] = True

        for recombo_id, _ in recombo_targets_review.items():
            self.attempt_to_add_to_firing_queue(recombo_id)

        # if len(next_nodes) > 0:
        #     for i in range(len(next_nodes)):
        #         if next_nodes[i]["function"] not in ["<<ENTRY>>", "<<EVENT>>"] and (not next_nodes[i]["requires_event"] or is_event):
        #             _ = self.run_node_then_forward(next_nodes[i], next_node_args[i], user_return_args=user_return_args, is_event=is_event)

        for node_id, value in self.firing_queue.items():
            if value and \
                ((not self.node_carousel[node_id]["requires_split_inputs"]) or (not no_split_inputs)) and \
                node_id in functions_fed_to:
                # print("  "*display_tab_count + "Firing node id from firing queue %s" % (node_id))
                tmp_args = deepcopy(parameters)
                tmp_args.update(self.queued_node_inputs[node_id])
                
                
                await self.run_node_then_forward(self.node_carousel[node_id], 
                                                 tmp_args, 
                                                 system_args, 
                                                 ws,
                                                 user_return_args=user_return_args, 
                                                 display_tab_count=display_tab_count+1)
                if clear_firing_queue:
                    self.firing_queue[node["id"]] = False
        # print("  "*display_tab_count + "Returning RNTF %s" % (node["id"]))
        # print("  "*display_tab_count + "Got user return args", user_return_args)
        return user_return_args, recombo_targets
        # for entry in node["feed_to"]:

    # def run_stream_node(self, node_id, system_args : dict):

    #     user_return_args, _, _, _ = self.run_node_with_dependencies(self.node_carousel[node_id], 
    #                                                                 self.queued_node_inputs[node_id],
    #                                                                 system_args)
    #     # loop = asyncio.get_event_loop()
    #     asyncio.run(self.fire_node_mappings_follow_up(node_id, system_args, user_return_args=user_return_args))
    
    async def fire_node_mappings_follow_up(self, 
                                           node_id, 
                                           system_args : dict, 
                                           ws : WebSocket = None,
                                           user_return_args = {}):
        node = self.node_carousel[node_id]
        functions_fed_to = []
        for feed_mapping in node["feed_to"]:
            if feed_mapping["destination"] not in ToolchainSession.SPECIAL_NODE_FUNCTIONS:
                functions_fed_to.append(feed_mapping["destination"])
        for node_id, _ in self.firing_queue.items():
            if node_id in functions_fed_to:
                await self.run_node_then_forward(self.node_carousel[node_id], 
                                                 self.queued_node_inputs[node_id], 
                                                 system_args, 
                                                 ws,
                                                 user_return_args=user_return_args)
        await self.send_state_notification({
            "type": "completed_propagation",
            "entry_node": node_id,
            "outputs": user_return_args
        }, ws)
        return user_return_args

    async def fire_queued_nodes_without_entry(self, no_split_inputs : bool = False):
        user_return_args = {}
        for node_id, value in self.firing_queue.items():
            if value and ((not self.node_carousel[node_id]["requires_split_inputs"]) or (not no_split_inputs)):
                # print("Firing node id %s" % (node_id))
                await self.run_node_then_forward(self.node_carousel[node_id], 
                                                 self.queued_node_inputs[node_id],
                                                 {},
                                                 None,
                                                 user_return_args=user_return_args)
        return user_return_args

    async def entry_prop(self, input_parameters, system_args : dict):
        """
        Activate the entry node with parameters, then propagate forward.
        """
        # print("entry_prop")
        self.entry_called = True
        if self.entry_point is None:
            return None
        result, _ = await self.run_node_then_forward(self.entry_point, input_parameters, system_args)
        return result
    
    async def event_prop(self, event_id, input_parameters, system_args : dict):
        """
        Activate an event node with parameters by id, then propagate forward.
        """
        # print("event_prop")
        target_event = self.node_carousel[event_id]
        result, _ = await self.run_node_then_forward(target_event, input_parameters, system_args, is_event=True)
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

    def notify_node_completion(self, node_id, result_arguments : dict):
        """
        Upon completion of a node, notify globally of the result of the node.
        """
        pass
    
    def await_generator_completion(self, generator : ThreadedGenerator, join_array : bool = True):
        """
        Wait for a threaded generator to finish, then return the completed sequence.
        """
        # print("await_generator_completion")
        # print("Awaiting generator")
        if type(generator) is str:
            return generator
        while not generator.done:
            sleep(0.01)
        # print("Got generator")
        if join_array:
            return "".join(generator.sent_values)
        return generator.sent_values

    async def notify_state_update(self, 
                                  action_origin_node_id, 
                                  state_arg_id,
                                  add_value = None,
                                  ws : WebSocket = None):
        # print("notify_state_update")
        if not add_value is None:
            # add_value = await add_value
            # print("Sending Add Value:", add_value)
            # print("Dumping", state_arg_id)
            # json.dumps({"state_arg_id": state_arg_id})
            # print("Dumping", add_value, type(add_value))
            # json.dumps({"add_value": add_value})
            # print("Actually dumping")
            dump_get = json.dumps({
                "type": "state_update",
                "state_arg_id": state_arg_id,
                "content_subset": "append",
                "content": add_value
            })
            # print("Sending dump")
            if not ws is None:
                await ws.send_text(dump_get)
        else:
            # print("Sending Value:", self.state_arguments[state_arg_id])
            if not ws is None:
                await ws.send_text(json.dumps({
                    "type": "state_update",
                    "state_arg_id": state_arg_id,
                    "content_subset": "full",
                    "content": self.state_arguments[state_arg_id]
                }))

    def dump(self):
        return {
            "title": self.state_arguments["title"],
            "toolchain_id": self.toolchain_id,
            "state_arguments": self.state_arguments,
            "session_hash_id": self.session_hash,
            "queue_inputs": self.queued_node_inputs,
            "firing_queue": self.firing_queue
        }
    
    def load(self, data, toolchains_available : Dict[str, dict]):
        if type(data) is str:
            data = json.loads(data)
        self.toolchain_id = data["toolchain_id"]
        self.toolchain = deepcopy(toolchains_available[data["toolchain_id"]])
        self.session_hash = data["session_hash_id"]
        self.reset_everything()
        self.assert_split_inputs()
        self.state_arguments["title"] = data["title"]
        self.state_arguments = data["state_arguments"]
        self.queued_node_inputs = data["queue_inputs"]
        self.firing_queue = data["firing_queue"]

# def