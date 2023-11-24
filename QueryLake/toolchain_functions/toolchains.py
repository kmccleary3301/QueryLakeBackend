import os, json
from ..models.model_manager import LLMEnsemble
from ..api.user_auth import get_user
from sqlmodel import Session, select, and_
from ..database import sql_db_tables
from ..models.langchain_sse import ThreadedGenerator
from copy import deepcopy
from time import sleep
from ..api.hashing import random_hash
from . import toolchain_node_functions
from fastapi import UploadFile
from sse_starlette.sse import EventSourceResponse

server_dir = "/".join(os.path.dirname(os.path.realpath(__file__)).split("/")[:-1])
upper_server_dir = "/".join(os.path.dirname(os.path.realpath(__file__)).split("/")[:-2])+"/"
user_db_path = server_dir+"/user_db/files/"

TOOLCHAINS = {}

default_toolchain = "document_q_and_a_test"

toolchain_files_list = os.listdir(upper_server_dir+"toolchains")
for toolchain_file in toolchain_files_list:
    if not toolchain_file.split(".")[-1] == "json":
        continue
    with open(upper_server_dir+"toolchains/"+toolchain_file, 'r', encoding='utf-8') as f:
        toolchain_retrieved = json.loads(f.read())
        f.close()
    TOOLCHAINS[toolchain_retrieved["id"]] = toolchain_retrieved


TOOLCHAIN_SESSION_CAROUSEL = { # SSE generators will need to be called in single form. Use this to retrieve them.
    # "abcd..." (random session hash): {
    #   "table_target" 
    #
    # }
}

class ToolchainSession():
    SPECIAL_NODE_FUNCTIONS = ["<<ENTRY>>", "<<EVENT>>"]
    SPECIAL_ARGUMENT_ORIGINS = ["<<SERVER_ARGS>>", "<<USER>>", "<<STATE>>"]

    def __init__(self, toolchain_id, function_getter, session_hash) -> None:
        self.session_hash = session_hash
        self.toolchain_id = toolchain_id
        self.toolchain = deepcopy(TOOLCHAINS[toolchain_id])
        self.function_getter = function_getter
        self.reset_everything()
        self.assert_split_inputs()

    def reset_everything(self):
        """
        Reset everything, as if the toolchain had just been loaded.
        Only the toolchain json itself is unaffected.
        """

        self.node_carousel = {}
        self.generators = {}
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

        for entry in self.toolchain["pipeline"]:
            
            self.node_carousel[entry["id"]] = entry

            self.node_carousel[entry["id"]]["requires_event"] = False

            if entry["function"] == "<<ENTRY>>":
                self.entry_point = entry
                self.entry_point_id = entry["id"]
            elif entry["function"] == "<<EVENT>>":
                self.event_points.append(entry)
                self.event_point_ids.append(entry["id"])
            else:
                self.function_getter(entry["function"])
        
        for event_point in self.event_points:
            self.assert_event_path(event_point)
        
        for key, value in self.toolchain["state_initialization"].items():
            self.state_arguments[key] = value
        
        self.session_state_generator = ThreadedGenerator(encode_hex=False)
    
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

    def run_node_with_dependencies(self,
                                   node : dict, 
                                   function_parameters : dict, 
                                   use_previous_outputs : bool = True):
        
        # print("Running node %s with args %s" % (node["id"], json.dumps(function_parameters)))
        node_arguments_required = self.get_node_argument_requirements(node)
        node_argument_ids = [entry["argument_name"] for entry in node_arguments_required]

        filter_args = {}
        sum_list = 0
        if node["function"] not in ToolchainSession.SPECIAL_NODE_FUNCTIONS:
            for key, value in self.queued_node_inputs[node["id"]].items():
                filter_args[key] = value
                sum_list += 1
        for key, value in function_parameters.items():
            if key in node_argument_ids:
                filter_args[key] = value
                sum_list += 1

        # print("Filter args:", filter_args)

        for argument in node_arguments_required:
            if argument["argument_name"] in filter_args:
                continue
            argument["satisfied"] = False
            if argument["argument_name"] in filter_args:
                argument["satisfied"] = True
                sum_list += 1
            elif "origin" in argument and argument["origin"] == "<<STATE>>":
                assert argument["argument_name"] in self.state_arguments, f"Required state argument {argument['argument_name']} could not be found."
                filter_args[argument["argument_name"]] = self.state_arguments[argument["argument_name"]]
                sum_list += 1
            else:
                if "origin" in argument and argument["origin"] in self.layer_outputs and use_previous_outputs:
                    filter_args[argument["argument_name"]] = self.layer_outputs[argument["origin"]][argument["argument_name"]]
                    sum_list += 1
                elif use_previous_outputs:
                    node_source = self.node_carousel[argument["origin"]]
                    run_arguments = self.run_node_with_dependencies(node_source, function_parameters, use_previous_outputs=use_previous_outputs)
                    filter_args[argument["argument_name"]] = run_arguments[argument["argument_name"]]
                    sum_list += 1
        assert sum_list >= len(node_arguments_required), f"Failed to satisfy required arguments for node {node['id']}"

        if node["function"] not in ToolchainSession.SPECIAL_NODE_FUNCTIONS:
            function_target = self.function_getter(node["function"])
            run_arguments = function_target(**filter_args)
            self.layer_outputs[node["id"]] = run_arguments
        else:
            run_arguments = function_parameters

        user_return_args = {}

        split_targets = []

        for entry in node["feed_to"]:
            
            for feed_map in entry["input"]:
                assert "value" in feed_map or \
                        feed_map["output_argument_id"] in run_arguments or \
                        feed_map["output_argument_id"] in function_parameters, "argument could not be found"
                
                target_arg = feed_map["target_argument"]

                if "value" in feed_map:
                    input_value = feed_map["value"]
                elif "input_argument_id" in feed_map:
                    input_value = run_arguments[feed_map["input_argument_id"]]
                elif feed_map["output_argument_id"] in run_arguments:
                    input_value = run_arguments[feed_map["output_argument_id"]]
                elif feed_map["output_argument_id"] in function_parameters:
                    input_value = function_parameters[feed_map["output_argument_id"]]

                # print("Target Arg", target_arg, type(input_value), str(type(input_value)))

                if entry["destination"] == "<<STATE>>":
                    # Need to await generator here.
                    if type(input_value) is ThreadedGenerator:
                        input_value = self.await_generator_completion(input_value)
                    self.special_state_action(entry["action"], target_arg, input_value)
                elif entry["destination"] == "<<USER>>":
                    # In the case of a generator, run create_generator. No need to wait.
                    if type(input_value) is ThreadedGenerator:
                        self.create_generator(node["id"], target_arg, input_value)
                    user_return_args[target_arg] = input_value
                else:
                    # In the case of a generator, await the output and use that instead.
                    if type(input_value) is ThreadedGenerator:
                        input_value = self.await_generator_completion(input_value)
                    if self.check_if_input_combination_required(entry["destination"], feed_map["target_argument"]):
                        if feed_map["target_argument"] not in self.queued_node_inputs[entry["destination"]]:
                            self.queued_node_inputs[entry["destination"]][feed_map["target_argument"]] = []
                        self.queued_node_inputs[entry["destination"]][feed_map["target_argument"]].append(input_value)
                    else:
                        self.queued_node_inputs[entry["destination"]][feed_map["target_argument"]] = input_value

                    if "split_outputs" in entry and entry["split_outputs"]:
                        self.queued_node_inputs[entry["destination"]][feed_map["target_argument"]] = [e for e in input_value]
                        split_targets.append({"node_id": entry["destination"], "split_argument_name": feed_map["target_argument"]})
                    self.attempt_to_add_to_firing_queue(entry["destination"])
        self.firing_queue[node["id"]] = False
        return run_arguments, user_return_args, split_targets


    def attempt_to_add_to_firing_queue(self, node_id):


        print("Attempting to add %s" % (node_id))
        required_args = [entry["argument_name"] for entry in self.node_carousel[node_id]["arguments"] if entry["origin"] not in ToolchainSession.SPECIAL_ARGUMENT_ORIGINS]
        required_args = sorted(list(set(required_args)))
        available_args = sorted(list(set(list(self.queued_node_inputs[node_id].keys()))))
        # print(required_args)
        # print(available_args)
        if available_args != required_args:
            print("Input args not satisfied")
            return

        self.firing_queue[node_id] = True

    def check_if_input_combination_required(self, node_id, input_argument):
        relevant_argument = None
        for arg_entry in self.node_carousel[node_id]["arguments"]:
            if arg_entry["argument_name"] == input_argument:
                relevant_argument = arg_entry
                break
        return ("merge_parallel_outputs" in relevant_argument and relevant_argument["merge_parallel_outputs"])

    def special_state_action(self, state_action, target_arg, input_arg):
        """
        Router function for state actions. Multiple are necessary because of chat
        history appending and formatting.
        """
        assert state_action in [
            "append_model_response_to_chat_history",
            "append_user_query_to_chat_history",
        ], f"state action \'{state_action}\' not found"
        action = getattr(self, state_action)
        action(**{
            "target_arg": target_arg,
            "input_arg": input_arg
        })

    def append_model_response_to_chat_history(self, target_arg, input_arg):
        """
        Append provided content to state argument with OpenAI's message format, with the role of assistant.
        """
        assert target_arg in self.state_arguments, f"State argument \'{target_arg}\' not found"
        new_addition = {
            "role": "assistant",
            "content": input_arg
        }
        self.state_arguments[target_arg].append(new_addition)
        self.notify_state_update(target_arg, new_addition)

    def append_user_query_to_chat_history(self, target_arg, input_arg):
        """
        Append provided content to state argument with OpenAI's message format, with the role of user.
        """
        assert target_arg in self.state_arguments, f"State argument \'{target_arg}\' not found"
        new_addition = {
            "role": "user",
            "content": input_arg
        }
        self.state_arguments[target_arg].append(new_addition)
        self.notify_state_update(target_arg, new_addition)


    def run_node_then_forward(self, node, parameters, user_return_args = {}, is_event : bool = False, no_split_inputs : bool = False):
        # print("Running node: %s" % (node["id"]))

        run_arguments, user_return_args_get, split_targets = self.run_node_with_dependencies(node, parameters)
        user_return_args.update(user_return_args_get)
        for split_entry in split_targets:
            # print("Got split target:", split_entry)
            get_split_arguments = deepcopy(self.queued_node_inputs[split_entry["node_id"]][split_entry["split_argument_name"]])
            # print("Got split argument:", get_split_arguments)

            for value in get_split_arguments:
                self.queued_node_inputs[split_entry["node_id"]][split_entry["split_argument_name"]] = value
                run_arguments
                self.run_node_then_forward(self.node_carousel[split_entry["node_id"]], {}, no_split_inputs=True)
        # if len(next_nodes) > 0:
        #     for i in range(len(next_nodes)):
        #         if next_nodes[i]["function"] not in ["<<ENTRY>>", "<<EVENT>>"] and (not next_nodes[i]["requires_event"] or is_event):
        #             _ = self.run_node_then_forward(next_nodes[i], next_node_args[i], user_return_args=user_return_args, is_event=is_event)

        for node_id, value in self.firing_queue.items():
            if value and ((not self.node_carousel[node_id]["requires_split_inputs"]) or (not no_split_inputs)):
                # print("Firing node id %s" % (node_id))
                self.run_node_then_forward(self.node_carousel[node_id], self.queued_node_inputs[node_id], user_return_args=user_return_args)
        return user_return_args
        # for entry in node["feed_to"]:

    def entry_prop(self, input_parameters):
        """
        Activate the entry node with parameters, then propagate forward.
        """
        return self.run_node_then_forward(self.entry_point, input_parameters)
    
    def event_prop(self, event_id, input_parameters):
        """
        Activate an event node with parameters by id, then propagate forward.
        """
        target_event = self.node_carousel[event_id]
        return self.run_node_then_forward(target_event, input_parameters, is_event=True)
    
    def update_active_node(self, node_id, arguments : dict = None):
        """
        Send the state of the session as a json.
        """
        send_information = {
            "type": "active_node_update",
            "active_node": node_id
        }
        if not arguments is None:
            send_information.update({"arguments": arguments})
        self.session_state_generator.send(json.dumps(send_information))

    def notify_node_completion(self, node_id, result_arguments : dict):
        """
        Upon completion of a node, notify globally of the result of the node.
        """
        pass

    def create_generator(self, node_origin_id, argument_name, generator):
        """
        On the creation of a ThreadedGenerator by a node, appends the generator
        to the session state, then notifies the client via the global generator
        of its creation.
        """
        entry = {
            "generator_id": node_origin_id+"|||"+argument_name,
            "origin": node_origin_id,
            "argument_name": argument_name
        }
        send_message = {"type": "generator_creation"}
        send_message.update(entry)
        # print("Creating generator:", json.dumps(send_message))
        self.session_state_generator.send(json.dumps(send_message))
        entry.update({"generator": generator})
        self.generators[entry["generator_id"]] = entry

    def get_generator(self, generator_id):
        return self.generators[generator_id]["generator"]

    def await_generator_completion(self, generator : ThreadedGenerator, join_array : bool = True):
        """
        Wait for a threaded generator to finish, then return the completed sequence.
        """
        # print("Awaiting generator")
        while not generator.done:
            sleep(0.01)
        # print("Got generator")
        if join_array:
            return "".join(generator.sent_values)
        return generator.sent_values

    def notify_state_update(self, state_arg_id, add_value : None):
        if not add_value is None:
            self.session_state_generator.send(json.dumps({
                "type": "state_update",
                "state_arg_id": state_arg_id,
                "content_subset": "append",
                "content": add_value
            }))
        else:
            self.session_state_generator.send(json.dumps({
                "type": "state_update",
                "state_arg_id": state_arg_id,
                "content_subset": "full",
                "content": self.state_arguments[state_arg_id]
            }))

    def dump(self):
        return json.dumps({
            "state_arguments": self.state_arguments,
            "toolchain_id": self.toolchain_id,
            "session_hash_id": self.session_hash
        })
    
    def load(self, data):
        if type(data) is str:
            data = json.loads(data)
        self.toolchain_id = data["toolchain_id"]
        self.toolchain = TOOLCHAINS[data["toolchain_id"]]
        self.session_hash = data["session_hash_id"]
        self.state_arguments = data["state_arguments"]

        

        
    # def entry_call(self, parameters):
    #     for output in self.entry_point:

def toolchain_function_caller(function_name):
    return getattr(toolchain_node_functions, function_name)



def get_available_toolchains(database : Session,
                             username : str, 
                             password_prehash : str):
    """
    Returns available toolchains with chat window settings and all.
    Will find organization locked
    If there are organization locked toolchains, they will be added to the database.
    """
    user = get_user(database, username, password_prehash)
    toolchains_available = {}
    for key, toolchain in TOOLCHAINS.items():
        if toolchain["category"] not in toolchains_available:
            toolchains_available[toolchain["category"]] = []
        toolchains_available[toolchain["category"]].append({
            "name": toolchain["name"],
            "id": toolchain["id"],
            "category": toolchain["category"],
            "chat_window_settings": toolchain["chat_window_settings"]
        })
    result = {
        "toolchains": [{"category": key, "entries": value} for key, value in toolchains_available.items()],
        "default": default_toolchain
    }
    return {"success": True, "result": result}

def create_toolchain_session(database : Session,
                             username : str, 
                             password_prehash : str,
                             toolchain_id : str):
    """
    Initiate a toolchain session with a random access token.
    This token is the session ID, and the session will be stored in the
    database accordingly.
    """
    user = get_user(database, username, password_prehash)
    session_hash = random_hash()
    toolchain_get = TOOLCHAINS[toolchain_id]
    TOOLCHAIN_SESSION_CAROUSEL[session_hash] = {
        "session": ToolchainSession(toolchain_get, toolchain_function_caller, session_hash),
        "author": username
    }
    return {"success": True, "session_id": session_hash}

def get_session_global_generator(database : Session,
                                username : str, 
                                password_prehash : str,
                                session_id : str):
    """
    Get the session global generator for a given toolchain.
    """
    assert session_id in TOOLCHAIN_SESSION_CAROUSEL, "Session not found"
    user = get_user(database, username, password_prehash)
    return EventSourceResponse(TOOLCHAIN_SESSION_CAROUSEL[session_id]["session"].session_state_generator)

def get_session_state(database : Session,
                        username : str, 
                        password_prehash : str,
                        session_id : str):
    """
    Get the session state of a given toolchain.
    """
    assert session_id in TOOLCHAIN_SESSION_CAROUSEL, "Session not found"
    user = get_user(database, username, password_prehash)
    return {"success": True, "result": TOOLCHAIN_SESSION_CAROUSEL[session_id]["session"].state_arguments}

def get_generator_by_id(database : Session,
                        username : str, 
                        password_prehash : str,
                        session_id : str,
                        generator_id : str):
    """
    Get a specific generator id in a given session.
    """
    assert session_id in TOOLCHAIN_SESSION_CAROUSEL, "Session not found"
    user = get_user(database, username, password_prehash)
    assert generator_id in TOOLCHAIN_SESSION_CAROUSEL[session_id]["session"].generators, "Generator not found"
    generator_retrieved = TOOLCHAIN_SESSION_CAROUSEL[session_id]["session"].generators[generator_id]
    return EventSourceResponse(generator_retrieved)


# def append_file_to_session()

def entry_call(database : Session,
                username : str, 
                password_prehash : str,
                session_id : str,
                entry_parameters : dict):
    """
    Call entry point in toolchain and propagate forward.
    entry parameters can be provided, however there must be special cases for
    things like files.
    """
    assert session_id in TOOLCHAIN_SESSION_CAROUSEL, "Session not found"
    user = get_user(database, username, password_prehash)
    return {"success": True, "result": TOOLCHAIN_SESSION_CAROUSEL[session_id]["session"].entry_prop(entry_parameters)}

def event_call(database : Session,
                username : str, 
                password_prehash : str,
                session_id : str,
                event_node_id : str,
                entry_parameters : dict):
    """
    Call event point in toolchain and propagate forward.
    entry parameters can be provided, however there must be special cases for
    things like files.
    """
    assert session_id in TOOLCHAIN_SESSION_CAROUSEL, "Session not found"
    user = get_user(database, username, password_prehash)
    return {"success": True, "result": TOOLCHAIN_SESSION_CAROUSEL[session_id]["session"].event_prop(event_node_id, entry_parameters)}



