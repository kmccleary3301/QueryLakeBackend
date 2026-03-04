
// """
// Below is the typing scheme for object traversal and manipulation within ToolChains.
// It allows for full control over objects and their values, which is necessary for the
// argument propagation without using redundant nodes and API functions to accomplish
// this instead.

import { displaySection } from "./toolchain-interface";

// The sequenceAction type can be thought of as a bash command, and the object
// can be thought of as a file system which is being traversed and/or changed.
// """

export type substituteAny = string | number | boolean | null | undefined | Array<substituteAny> | {[key : string] : substituteAny} | Map<string, substituteAny>;
// export type compositionType = Array<substituteAny> | { [key: string]: substituteAny; } | Map<string, substituteAny>;
// export type compositionObjectType = { [key: string]: substituteAny; } | Map<string, substituteAny>;

export type compositionGenericType<T> = Array<T | compositionGenericType<T>> | { [key: string]: T | compositionGenericType<T>; } | Map<string, T | compositionGenericType<T>>;
export type compositionObjectGenericType<T> = { [key: string]: T | compositionObjectGenericType<T>; } | Map<string, T | compositionObjectGenericType<T>>;

export type compositionType = compositionGenericType<substituteAny>;
export type compositionObjectType = compositionObjectGenericType<substituteAny>;

export interface staticValue {
    type?: "staticValue";
    value: substituteAny;
}

export interface stateValue {
    type?: "stateValue";
    route: staticRoute;
}

export interface getNodeInput {
    type?: "getNodeInput";
    route: staticRoute;
}

export interface getNodeOutput {
    type?: "getNodeOutput";
    route: staticRoute;
}

export interface getFiles {
    type?: "getFiles";
    route?: staticRoute;
    routes?: staticRoute[];
    getText?: boolean;
}

export type indexRouteRetrievedStateValue = {
    getFromState: stateValue;
}

export interface indexRouteRetrievedInputArgValue {
    getFromInputs: getNodeInput;
}

export interface indexRouteRetrievedOutputArgValue {
    getFromOutputs: getNodeOutput;
}

export interface indexRouteRetrievedFile {
    getFiles: getFiles;
}

export type indexRouteRetrieved = {
    type?: "indexRouteRetrieved";
    getFrom: valueObj;
}

export type indexRouteRetrievedBasic = indexRouteRetrieved | staticValue | indexRouteRetrievedStateValue | indexRouteRetrievedInputArgValue | indexRouteRetrievedOutputArgValue | indexRouteRetrievedFile;

export interface getLengthValue {
    type?: "getLengthValue";
    getLength: indexRouteRetrievedBasic;
}

export type indexRouteRetrievedNew = indexRouteRetrievedBasic | getLengthValue;

export interface rootActionType {
    condition?: Condition | conditionBasic;
}

export interface createAction extends rootActionType {
    type: "createAction";
    initialValue?: valueObj;
    insertion_values?: (valueObj | null)[];
    insertions?: staticRouteElementType[][];
    route?: staticRoute;
}

export interface deleteAction extends rootActionType {
    type: "deleteAction";
    route?: staticRoute | null;
    routes?: staticRoute[] | null;
}

export interface updateAction extends rootActionType {
    type: "updateAction";
    route: staticRoute;
    value?: valueObj | null;
}

export interface appendAction extends rootActionType {
    type: "appendAction";
    initialValue?: valueObj | null;
    insertion_values?: (valueObj | null)[];
    insertions?: staticRouteElementType[][];
    route?: staticRoute | null;
}

export interface operatorAction extends rootActionType {
    type: "operatorAction";
    action: "+" | "-";
    value?: valueObj | null;
    route: staticRoute;
}

export interface backOut extends rootActionType {
    type: "backOut";
    count?: number;
}

export interface insertAction extends rootActionType {
    type: "insertSequenceAction";
    route: staticRoute;
    replace?: boolean;
}

export type valueObj = staticValue | stateValue | getNodeInput | getNodeOutput | getFiles;

export type staticRouteBasicElementType = number | string;
export type staticRouteBasic = Array<staticRouteBasicElementType>;

export type staticRouteElementType = number | string | indexRouteRetrievedNew;
export type staticRoute = Array<staticRouteElementType>;

export type sequenceAction = staticRouteElementType | createAction | updateAction | appendAction | deleteAction | operatorAction | backOut;
export type sequenceActionNonStatic = createAction | updateAction | appendAction | deleteAction | operatorAction | backOut;

export interface conditionBasic {
    variableOne?: valueObj | null;
    variableTwo: valueObj;
    operator: "==" | "!=" | ">" | "<" | ">=" | "<=" | "in" | "not in" | "is" | "is not";
}

export interface Condition {
    type?: "singular" | "and" | "or" | "not";
    statement: conditionBasic | Array<Condition | conditionBasic>;
}

export interface feedMappingAtomic {
    destination: string;
    sequence?: Array<sequenceAction>;
    route?: staticRoute | null;
    stream?: boolean;
    stream_initial_value?: substituteAny;
    store?: boolean;
    iterate?: boolean;
    condition?: Condition | conditionBasic;
}

export interface feedMappingOriginal extends feedMappingAtomic {
    getFrom: valueObj;
}

export interface feedMappingStaticValue extends feedMappingAtomic {
    value: substituteAny;
}

export interface feedMappingStateValue extends feedMappingAtomic {
    getFromState: stateValue;
}

export interface feedMappingInputValue extends feedMappingAtomic {
    getFromInputs: getNodeInput;
}

export interface feedMappingOutputValue extends feedMappingAtomic {
    getFromOutputs: getNodeOutput;
}

export type feedMapping = feedMappingOutputValue | feedMappingInputValue | feedMappingStateValue | feedMappingStaticValue | feedMappingOriginal;

export interface nodeInputArgument {
    key: string;
    value?: substituteAny;
    from_user?: boolean;
    from_server?: boolean;
    from_state?: stateValue;
    from_files?: getFiles;
    sequence?: sequenceAction[];
    optional?: boolean;
    type_hint?: string;
}

export interface chatWindowMapping {
    route: staticRoute;
    display_as: "chat" | "markdown" | "file" | "image";
}

export interface eventButton {
    type: "eventButton";
    event_node_id: string;
    return_file_response?: boolean;
    feather_icon: string;
    display_text?: string;
}


export interface inputPointer {
    event: string;
    event_parameter: string;
}

export interface chatBarProperties {
    text?: inputPointer;
    file?: inputPointer;
}

export interface chatBar {
    type?: "chat_bar";
    properties: chatBarProperties;
    display_text?: string;
}

export interface toggleInput extends inputPointer {
    type?: "toggle";
    display_text?: string;
}

export interface nativeApplicationParameter extends inputPointer {
    type?: "native_application_parameter";
    origin: string;
}

export type inputConfigType = chatBar | toggleInput | nativeApplicationParameter;



export interface displayConfiguration {
    display_mappings: (chatWindowMapping | eventButton)[];
    max_files?: number;
    enable_rag?: boolean;
    input_config: inputConfigType[];
}

export interface toolchainNode {
    id: string;
    api_function?: string;
    input_arguments?: nodeInputArgument[];
    feed_mappings?: feedMapping[];
}

// class startScreenSuggestion(BaseModel):
//     """
//     This defines a displayed suggestion in the window when the user starts a new session.
//     It defines display text, an event node id, and a dictionary of event parameters for the event.
//     """
//     display_text: str
//     event_id: str
//     event_parameters: Optional[Dict[str, Any]] = {}
 
export interface ToolChain {
    name: string;
    id: string;
    category: string;
    display_configuration: displaySection;
    first_event_follow_up?: string;
    // suggestions?: startScreenSuggestion[];
    initial_state: substituteAny;
    nodes: toolchainNode[];
}

export interface ToolChainSessionFile {
    name: string;
    document_hash_id: string;
}

export interface toolchainPointer {
    title: string;
    category: string;
    id: string;
}
