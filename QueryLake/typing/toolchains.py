from typing import List, Dict, Optional, Union, Tuple, Literal, Any
from pydantic import BaseModel
from .toolchain_interface import DisplaySection

"""
Below is the typing scheme for object traversal and manipulation within ToolChains.
It allows for full control over objects and their values, which is necessary for the
argument propagation without using redundant nodes and API functions to accomplish
this instead.

The sequenceAction type can be thought of as a bash command, and the object
can be thought of as a file system which is being traversed and/or changed.
"""

class staticValue(BaseModel):
    """
    Just a static value to be used in a sequenceAction.
    """
    type : Optional[Literal["staticValue"]] = "staticValue"
    value : Any

class stateValue(BaseModel):
    """
    Used to retrieve a value from the toolchain state via traversal.
    """
    type : Optional[Literal["stateValue"]] = "stateValue"
    route: "staticRoute"

# class nodeInputArgObject(BaseModel):
class getNodeInput(BaseModel):
    type : Optional[Literal["getNodeInput"]] = "getNodeInput"
    route: "staticRoute"

# class nodeOutputArgObject(BaseModel):
class getNodeOutput(BaseModel):
    type : Optional[Literal["getNodeOutput"]] = "getNodeOutput"
    route: "staticRoute"

class getFile(BaseModel):
    """
    There is another equivalent to state in the toolchain, which is set of files uploaded to the session.
    """
    type : Optional[Literal["getFile"]] = "getFile"
    route: Optional["staticRoute"] = None
    routes: Optional[List["staticRoute"]] = None
    # getText: Optional[bool] = False # If True, then we get the text of the file instead of the file object.

# Add alternatives for direct referencing of valueObjs in getFrom

# The following 3 are used to shorten the syntax in getting values in feedMappings.
# Doing it the normal way works fine too, as their retrieval fields are all different.
class indexRouteRetrievedStateValue(BaseModel):
    # type : Optional[Literal["indexRouteRetrieved"]] = "indexRouteRetrieved"
    getFromState: stateValue

class indexRouteRetrievedInputArgValue(BaseModel):
    # type : Optional[Literal["indexRouteRetrieved"]] = "indexRouteRetrieved"
    getFromInputs: getNodeInput

class indexRouteRetrievedOutputArgValue(BaseModel):
    # type : Optional[Literal["indexRouteRetrieved"]] = "indexRouteRetrieved"
    getFromOutputs: getNodeOutput
    
class indexRouteRetrievedFile(BaseModel):
    # type : Optional[Literal["indexRouteRetrieved"]] = "indexRouteRetrieved"
    getFiles: getFile

class indexRouteRetrieved(BaseModel):
    """
    This is specifically for pulling a `valueObj` and using it as a route element as if it was statically coded.
    """
    type : Optional[Literal["indexRouteRetrieved"]] = "indexRouteRetrieved"
    getFrom: "valueObj"

indexRouteRetrievedBasic = Union[
    indexRouteRetrieved, 
    staticValue, 
    indexRouteRetrievedStateValue, 
    indexRouteRetrievedInputArgValue, 
    indexRouteRetrievedOutputArgValue,
    indexRouteRetrievedFile
]

class getLengthValue(BaseModel):
    """
    This is specifically for pulling a `valueObj` and using it as a route element as if it was statically coded.
    """
    type : Optional[Literal["getLengthValue"]] = "getLengthValue"
    getLength: indexRouteRetrievedBasic

indexRouteRetrievedNew = Union[
    indexRouteRetrievedBasic,
    getLengthValue
]




# Branching states are deprecated (Actually, they were never implemented)
# class valueFromBranchingState(BaseModel):
#     """
#     In toolchains, there is a concept of a "branching state", where you have 
#     a node that outputs a list or iterable of some kind. There is a subgraph
#     of the toolchain that this is fed to which is then executed for each item.
#     The branching state is used as a temporary state during this execution to
#     specify the route and index of the current item in the list.
#     """
#     type : Optional[Literal["valueFromBranchingState"]] = "valueFromBranchingState"
#     route: "staticRoute"
#     # dimension: int



class rootActionType(BaseModel):
    """
    This is the parent class for all the action types.
    It's only here for reusage of the condition field.
    """
    condition: Optional[Union["Condition", "conditionBasic"]] = None

class createAction(rootActionType):
    """
    This will create a new object in the toolchain state, or modify an existing one.
    This will be created at the given route within the current working directory/route.
    
    
    The logic on this is probably one of the craziest ones, because you can construct the object
    with `valueObj` strewn about wherever via the `insertion_values` and `insertions` fields.
    """
        
    type : Optional[Literal["createAction"]] = "createAction"
    initialValue: Optional["valueObj"] = None # If None, then we use the given value.
    insertion_values: Optional[List[Union["valueObj", Literal[None]]]] = [] # Use None when you want to use the given value.
    insertions: Optional[List[List["staticRouteElementType"]]] = []
    route: Optional["staticRoute"] = [] # If None, assert the current object in focus is a list and append to it.

class deleteAction(rootActionType):
    """
    Delete object at route or multiple routes.
    
    Using `routes` doesn't work at the moment.
    """
    type : Optional[Literal["deleteAction"]] = "deleteAction"
    route: Optional["staticRoute"] = None
    routes: Optional[List["staticRoute"]] = None
    
class updateAction(rootActionType):
    type : Optional[Literal["updateAction"]] = "updateAction"
    route: "staticRoute"
    value: Optional["valueObj"] = None
    
class appendAction(rootActionType):
    """
    Value is alread provided, but we can initialize an object to make changes to in place of it.
    """
    type : Optional[Literal["appendAction"]] = "appendAction"
    initialValue: Optional["valueObj"] = None # If None, then we use the given value.
    insertion_values: Optional[List[Union["valueObj", Literal[None]]]] = [] # Use None when you want to use the given value.
    insertions: Optional[List[List["staticRouteElementType"]]] = []
    route: Optional["staticRoute"] = [] # If None, assert the current object in focus is a list and append to it.

    
class operatorAction(rootActionType):
    """
    Value is alread provided, but we can initialize an object to make changes to in place of it.
    """
    type : Optional[Literal["operatorAction"]] = "operatorAction"
    action: Literal["+=", "-="]
    value: Optional["valueObj"] = None # If None, then we use the given value.
    route: "staticRoute" 

class backOut(rootActionType):
    """
    Within sequenceActions, this is equivalent to a "cd ../" command in bash.
    The count represents the number of times to go back.
    """
    type : Optional[Literal["backOut"]] = "backOut"
    count : Optional[int] = 1

class insertAction(rootActionType):
    """
    TODO: Logic not quite clear. Needs some of the actions of `sequenceAction`, but
    functionality is different as the value is already known and provided.
    For now, it uses staticRoute.
    
    Used by `feedMapping` to insert a value into a target object, either toolchain
    state or node inputs. 
    """
    type : Optional[Literal["insertSequenceAction"]] = "insertSequenceAction"
    route : "staticRoute"
    replace: Optional[bool] = True

valueObj = Union[staticValue, stateValue, getNodeInput, getNodeOutput, getFile]
staticRouteBasicElementType = Union[int, str]
staticRouteBasic = List[staticRouteBasicElementType]

# It actually seems to matter that int be before string, because otherwise it converts the int to a string.
staticRouteElementType = Union[int, str, indexRouteRetrievedNew]
staticRoute = List[staticRouteElementType]
sequenceAction = Union[staticRouteElementType, createAction, updateAction, appendAction, deleteAction, operatorAction, backOut]

class conditionBasic(BaseModel):
    """
    Detail a condition on which a feedMapping should be executed.
    """
    variableOne: Optional["valueObj"] = None # Use a provided variable or value.
    variableTwo: "valueObj"
    operator: Literal["==", "!=", ">", "<", ">=", "<=", "in", "not in", "is", "is not"]

class Condition(BaseModel):
    type: Optional[Literal["singular", "and", "or", "not"]] = "singular"
    statement: Union[conditionBasic, List[Union["Condition", conditionBasic]]]

class feedMappingAtomic(BaseModel):
    """
    TODO: Add conditional feeding.
    Output mapping from a node to a destination.
    
    Note: if using stream, you must initialize the value (list or str) first, usually via createAction.
    """
    destination: str # Either a Node ID, "<<STATE>>", or "<<USER>>"
    sequence: Optional[List[sequenceAction]] = [] # This operates in the firing queue inputs if a node id is provided above.
    route: Optional[staticRoute] = None # If not None, we simply store the given value at this route, and ignore the sequence.
    stream: Optional[bool] = False
    stream_initial_value: Optional[Any] = None # Must be provided if stream is True. Pretty much always an empty string or list.
    
    store: Optional[bool] = False # If True, then we store the inputs into the mapped node without firing it.
    
    # If defined, we take the target arguments and split them at the given route. 
    # The route in the stored function inputs is already a list or some iterable, but
    # The actual queue has this copied with the value replaced with an element in each copy.
    iterate: Optional[bool] = False 

    condition: Optional[Union[Condition, conditionBasic]] = None


# The following five classes are used to shorten the syntax in getting values in feedMappings.
# They all have different retrieval field names, so the logic is non-ambiguous and allows for flexibility in choices.

class feedMappingOriginal(feedMappingAtomic):
    # type : Optional[Literal["indexRouteRetrieved"]] = "indexRouteRetrieved"
    getFrom: valueObj

class feedMappingStaticValue(feedMappingAtomic):
    # type : Optional[Literal["indexRouteRetrieved"]] = "indexRouteRetrieved"
    value: Any

class feedMappingStateValue(feedMappingAtomic):
    # type : Optional[Literal["indexRouteRetrieved"]] = "indexRouteRetrieved"
    getFromState: stateValue

class feedMappingInputValue(feedMappingAtomic):
    # type : Optional[Literal["indexRouteRetrieved"]] = "indexRouteRetrieved"
    getFromInputs: getNodeInput

class feedMappingOutputValue(feedMappingAtomic):
    # type : Optional[Literal["indexRouteRetrieved"]] = "indexRouteRetrieved"
    getFromOutputs: getNodeOutput


# The ordering here actually caused a massive headache. It seems that the order of the classes
# is a defaulting order, and it's checking is limited.
feedMapping = Union[
    feedMappingOutputValue,
    feedMappingInputValue, 
    feedMappingStateValue, 
    feedMappingStaticValue, 
    feedMappingOriginal, 
]

    
class nodeInputArgument(BaseModel):
    """
    This is an input kwarg of a node, similar to feedMapping in that it takes a value.
    Can modify it before passing via sequence.
    Value is statically provided.
    """
    key: str
    
    initialValue: Optional[Any] = None # If None, then we use the given value and perform sequence on it.
    from_user: Optional[bool] = False # If True, then we use the key value from user args on the propagation call, sequence and initialValue are ignored.
    from_server: Optional[bool] = False # If True, then we use the key value from server args, sequence and initialValue are ignored.
    from_state: Optional[stateValue] = None
    from_files: Optional[getFile] = None
    
    sequence: Optional[List[sequenceAction]] = []
    
    optional: Optional[bool] = False
 

class chatWindowMapping(BaseModel):
    """
    This defines a correspondance with an argument in the toolchain state and a
    display element within the chat window.
    """
    route: "staticRoute"
    display_as: Literal["chat", "markdown", "file", "image"]
    
class eventButton(BaseModel):
    """
    A button that can be clicked in the chat window.
    """
    type: Optional[Literal["eventButton"]] = "eventButton"
    event_node_id: str
    return_file_response: Optional[bool] = False
    feather_icon: str
    display_text: Optional[str] = None



class inputPointer(BaseModel):
    event : str
    event_parameter : str
    
class chatBarProperties(BaseModel):
    text: Optional[inputPointer] = None
    file: Optional[inputPointer] = None

class chatBar(BaseModel):
    type: Optional[Literal["chat_bar"]] = "chat_bar"
    properties: chatBarProperties
    display_text: Optional[str] = ""



class toggleInput(inputPointer):
    type: Optional[Literal["toggle"]] = "toggle"
    display_text: Optional[str] = ""

class nativeApplicationParameter(inputPointer):
    type: Optional[Literal["native_application_parameter"]] = "native_application_parameter"
    origin : str

class llmModelChoice(inputPointer):
    type: Optional[Literal["llm_model_choice"]] = "llm_model_choice"

inputConfigType = Union[
    chatBar,
    toggleInput,
    nativeApplicationParameter,
    llmModelChoice
]


# class displayConfiguration(BaseModel):
#     """
#     This is the display and usage configuration of the toolchain.
#     """
#     display_mappings: List[Union[chatWindowMapping, eventButton]]
#     max_files: Optional[int] = 0
#     # enable_rag: Optional[bool] = False,
#     input_config : Optional[List[inputConfigType]] = []
    

class toolchainNode(BaseModel):
    """
    This is a node within the toolchain. It has a unique ID and a sequence of actions.
    """
    id: str
    
    # One of the following must be provided.
    is_event: Optional[bool] = False
    api_function: Optional[str] = None
    
    # arguments: Optional[List[str]] = []
    
    input_arguments: Optional[List[nodeInputArgument]] = []
    
    feed_mappings: Optional[List[feedMapping]] = []
    
    

class startScreenSuggestion(BaseModel):
    """
    This defines a displayed suggestion in the window when the user starts a new session.
    It defines display text, an event node id, and a dictionary of event parameters for the event.
    """
    display_text: str
    event_id: str
    event_parameters: Optional[Dict[str, Any]] = {}

    
class ToolChain(BaseModel):
    """
    This is the main object that defines the toolchain.
    """
    name: str
    id: str
    category: str
    display_configuration: Optional[DisplaySection] = None
    
    suggestions: Optional[List[startScreenSuggestion]] = []
    
    initial_state: Dict[str, Any]
    
    nodes: List[toolchainNode]

class ToolChainSessionFile(BaseModel):
    """
    This is effectively a pointer to a file in the database so that it can be retrieved.
    """
    name: str
    document_hash_id: str

stateValue.update_forward_refs()
getNodeInput.update_forward_refs()
getNodeOutput.update_forward_refs()
getFile.update_forward_refs()
indexRouteRetrieved.update_forward_refs()
# valueFromBranchingState.update_forward_refs()

# valueObj.update_forward_refs()

createAction.update_forward_refs()
deleteAction.update_forward_refs()
updateAction.update_forward_refs()
appendAction.update_forward_refs()
# deleteListElementsAction.update_forward_refs()
# insertListElementAction.update_forward_refs()
operatorAction.update_forward_refs()
insertAction.update_forward_refs()

chatWindowMapping.update_forward_refs()


if __name__ == "__main__":
    # print("Test Class Dict:", testClassWithDictType(dict_in={"a": 1}))
    
    
    # print(createAction(**{"type": "createAction", "init": {"a": 1}}))
    # print(deleteAction(route=["a", "b"]))
    
    toolchain_create = ToolChain.parse_file('/home/kyle_m/QueryLake_Development/QueryLakeBackend/toolchains/chat_session_normal_new_scheme.json')
    
    print(toolchain_create)
    
    # print(createAction(type="createActionNope", initialValue=[1, 2, 3]))
    # print(createAction(type="createAction", init=1))
    # print(createAction(type="createAction", init="1"))