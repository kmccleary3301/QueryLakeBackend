"use client";
import { Dispatch } from "react";
import { 
  compositionObjectGenericType,
  // compositionGenericType,
  compositionObjectType, 
  compositionType,
  substituteAny 
} from "@/types/toolchains";
// import { toast } from "sonner";

export type toolchainStateType = {title?: string, [key : string]: substituteAny};

export type CallbackOrValue<T> = T | ((prevState: T) => T);
// type stateSetGeneric<T> = Dispatch<React.SetStateAction<T>>;
// type stateSet = stateSetGeneric<toolchainStateType>

type stateSet = (value: CallbackOrValue<toolchainStateType>) => void;
// type titleSet = Dispatch<React.SetStateAction<string>> | ((new_state : string) => void)
type sessionCallback = (session : ToolchainSession) => void;
// type deleteStateElements = string | number | Array<string | number | compositionObjectGenericType<substituteAny>>;
type deleteStateListType = Array<string | number | compositionObjectGenericType<substituteAny>>;
type deleteStateGenericType = string | number | compositionObjectGenericType<substituteAny>;

export interface ToolchainSessionMessage {
  toolchain_session_id?: string;
  [key: string]: any;
}

export default class ToolchainSession {
	public  onStateChange: stateSet;
	public  onCallEnd: sessionCallback;
	private onMessage: (message: object) => void;
  private onSend: (message: object) => void;
	private onOpen: (session : ToolchainSession) => void;
	private onError: (message: object) => void;
	private onClose: () => void;
	private onFirstCallEnd: () => void;
  private onCurrentEventChange: (event: string | undefined) => void;
	public  socket: WebSocket | undefined; // Add socket as a type
	private stream_mappings: Map<string, (string | number)[][]>;
  private message_queue: Object[];
  public currently_running: boolean;
  public current_event: undefined | string;

	// private stateChangeCounter: number;
  
	constructor ( { onStateChange = undefined, 
									onCallEnd = undefined,
									onMessage = undefined,
                  onSend = undefined,
								  onOpen = undefined,
									onError = undefined,
									onClose = undefined,
                  onFirstCallEnd = undefined,
                  onCurrentEventChange = undefined,
                }:{ 
                  onStateChange?: stateSet, 
								  onCallEnd?: sessionCallback,
									onMessage?: (message: object) => void,
                  onSend?: (message: object) => void,
								  onOpen?: (session : ToolchainSession) => void,
									onError?: (message: object) => void,
									onClose?: () => void,
                  onFirstCallEnd?: () => void,
                  onCurrentEventChange?: (event: string | undefined) => void,
                } ) {
		
		this.onStateChange = onStateChange || (() => {});
		this.onCallEnd = onCallEnd || (() => {});
		this.onMessage = onMessage || (() => {});
    this.onSend = onSend || (() => {});
		this.onOpen = onOpen || (() => {});
		this.onError = onError || (() => {});
		this.onClose = onClose || (() => {});
    this.onFirstCallEnd = onFirstCallEnd || (() => {});
    this.onCurrentEventChange = onCurrentEventChange || (() => {});
    this.message_queue = [];
    this.currently_running = false;
    this.current_event = undefined;
		
		this.socket = undefined;
		
		// this.state = {};
		this.stream_mappings = new Map<string, (string | number)[][]>();
    

		

		const onMsgCallback = async (event: MessageEvent) => {
			try {
				const message_text : Blob | string = event.data;

				// console.log("Message received data  :", typeof message_text, message_text)
				// const message_text_string : string = 
				// console.log("Message received string:", typeof message_text_string, message_text_string)
				const message = (typeof message_text === "string") ? 
													JSON.parse(message_text) : 
													JSON.parse(await message_text.text());
				
				// console.log("Message received parsed:", typeof message, message)
				// this.onStateChange(() => this.handle_message(message))
				this.onMessage(message);
        this.handle_message(message);
			} catch (error) {
				console.error("Error parsing message:", error);
			}
		};

		
		const onOpenCallback = () => {
			console.log("Connected to server");
			this.onOpen(this);
		}
		
		const onCloseCallback = () => {
      this.cleanup();
      reconnect();
      this.onClose();
    };

		const onErrorCallback = (error : Event) => {
			console.error("WebSocket error:", error);
			this.cleanup();
			this.onError({ error: "Connection error" });
		};
		
		const set_up_socket_handlers = () => {
			if (this.socket === undefined) return;
			this.socket.onmessage = onMsgCallback;
			this.socket.onopen = onOpenCallback;
			this.socket.onclose = onCloseCallback;
			this.socket.onerror = onErrorCallback;
		}


    // Add reconnection logic
    let reconnectAttempts = 0;
    const maxReconnectAttempts = 3;
    
    const reconnect = () => {
      if (reconnectAttempts < maxReconnectAttempts) {
        reconnectAttempts++;
        setTimeout(() => {
          console.log(`Attempting reconnect ${reconnectAttempts}/${maxReconnectAttempts}`);
          this.socket = new WebSocket(`/toolchain`);
          set_up_socket_handlers();
        }, 1000 * reconnectAttempts);
      }
    };

		reconnect();
	}
  
	

	handle_message(data : { [key : string] : substituteAny }) {
		if (this.socket === undefined) return;
		if ("state" in data) {
			const state = data["state"] as toolchainStateType;
			// this.state = state;
			this.onStateChange(state);
		}

    if (Object.prototype.hasOwnProperty.call(data, "type") && data.type === "node_execution_start") {
      this.current_event = data.node_id as string;
      this.onCurrentEventChange(this.current_event);
    }

		if (Object.prototype.hasOwnProperty.call(data, "ACTION") && get_value(data, "ACTION") === "END_WS_CALL") {
			// Reset this.stream_mappings to an empty map
			this.stream_mappings = new Map<string, (string | number)[][]>();
      this.message_queue = this.message_queue.slice(1);
      if (this.message_queue.length > 0) {
        this.onSend(this.message_queue[0]);
        this.socket.send(JSON.stringify(this.message_queue[0]));
      } else {
        this.currently_running = false;
        this.current_event = undefined;
        this.onCurrentEventChange(this.current_event);
        this.onCallEnd(this);
      }
		} else if (Object.prototype.hasOwnProperty.call(data, "trace") || Object.prototype.hasOwnProperty.call(data, "error")) {
			// console.log(data);
			this.onError(data);
		} else if ("type" in data) {
      
			if (data["type"] === "streaming_output_mapping") {
				const data_typed = data as {"type" : "streaming_output_mapping", "stream_id" : string, "routes" : (string | number)[][]};
				this.stream_mappings.set(data_typed["stream_id"], data_typed["routes"]);
			} else if (data["type"] === "state_diff") {
				const data_typed = data as {
					type : "state_diff",
					append_routes?: Route[];
					append_state?: compositionObjectType;
					update_state?: compositionObjectType;
					delete_state?: deleteStateListType;
				};

				// eslint-disable-next-line @typescript-eslint/no-unused-vars
				const { type , ...data_typed_type_removed } = data_typed;
        
				this.onStateChange(
          (prevState) => runStateDiff(prevState, data_typed_type_removed) as toolchainStateType
        );
			}
		} else if (checkKeys(["s_id", "v"], Object.keys(data))) {
			const data_typed = data as { s_id : string, v : substituteAny };
			const routes_get = this.stream_mappings.get(data_typed["s_id"]) as (string | number)[][];
			for (const route_get of routes_get) {
        this.onStateChange(
          (prevState) => appendInRoute(prevState, route_get, data_typed["v"]) as toolchainStateType
        )
			}
		} else if (checkKeys(["event_result"], Object.keys(data))) {
			const data_typed = data as { event_result : compositionObjectType };
			console.log("Event result:", data_typed["event_result"]);
		}
	}

	send_message(message : { [key : string] : substituteAny }) {
		if (this.socket === undefined) return;
    this.message_queue.push(message);
    if (this.currently_running) {
      return;
    }
    this.currently_running = true;
    this.onSend(message);
		this.socket.send(JSON.stringify(message));
	}

	cleanup() {
    this.currently_running = false;
    this.current_event = undefined;
    this.message_queue = [];
    this.stream_mappings.clear();
		if (this.socket !== undefined) {
			this.socket.onclose = () => {};
			this.socket.close();
			this.socket = undefined;
		}
  }

	// TODO: turn send_message into a queue system.
  
}


type Route = Array<string | number>;
// type streamMappingType = {
//     stream_id: string,
//     routes: (string | number)[][]
// }

export type handleToolchainMessageReturnType = {
  counter?: number,
  streamMappings?: Map<string, (string | number)[][]>,
  endWsCall?: boolean,
  sessionId?: string,
}

// export function handleToolchainMessage(
//   data : { [key : string] : substituteAny },
//   streamMappings: Map<string, (string | number)[][]>,
//   stateInitial: toolchainStateType,
//   onStateChange: stateSet,
//   counter: number,
// ) : handleToolchainMessageReturnType {

//   const returnObject : handleToolchainMessageReturnType = {};

//   if ("state" in data) {
//     const state = data["state"] as toolchainStateType;
//     onStateChange(state, counter);
//     counter += 1;
//     returnObject.counter = counter;
//   }

//   if (Object.prototype.hasOwnProperty.call(data, "ACTION") && get_value(data, "ACTION") === "END_WS_CALL") {
//     // Rest this.stream_mappings to an empty map
//     returnObject.streamMappings = new Map<string, (string | number)[][]>();
//     returnObject.endWsCall = true;
//   } else if (Object.prototype.hasOwnProperty.call(data, "trace")) {
//     console.log(data);
//   } else if ("type" in data) {
//     // console.log("Data type:", data["type"]);

//     if (data["type"] === "streaming_output_mapping") {
//       const data_typed = data as {"type" : "streaming_output_mapping", "stream_id" : string, "routes" : (string | number)[][]};
//       returnObject.streamMappings = streamMappings;
//       returnObject.streamMappings.set(data_typed["stream_id"], data_typed["routes"]);
//     } else if (data["type"] === "state_diff") {
//       const data_typed = data as {
//         type : "state_diff",
//         append_routes?: Route[];
//         append_state?: compositionObjectType;
//         update_state?: compositionObjectType;
//         delete_state?: deleteStateListType;
//       };

//       // eslint-disable-next-line @typescript-eslint/no-unused-vars
//       const { type , ...data_typed_type_removed } = data_typed;
      
//       // console.log("State before diff:", JSON.parse(JSON.stringify(this.state)));
//       // this.state = runStateDiff(this.state, data_typed_type_removed) as typeof this.state;
//       // console.log("State after diff:", JSON.parse(JSON.stringify(this.state)));

//       onStateChange(
//         runStateDiff(stateInitial, data_typed_type_removed) as typeof stateInitial, 
//         counter
//       );
//       returnObject.counter = counter + 1;
//       // this.onTitleChange(get_value(this.state, "title") as string);
//     }
//   } else if (checkKeys(["s_id", "v"], Object.keys(data))) {
//     console.log("Stream Token:", data);

//     const data_typed = data as { s_id : string, v : substituteAny };

//     // const routes_get: Array<Array<string | number>> = stream_mappings[parsedResponse["s_id"]];
//     const routes_get = streamMappings.get(data_typed["s_id"]) as (string | number)[][];


//     returnObject.counter = counter;
    
//     for (const route_get of routes_get) {


//       stateInitial = appendInRoute(stateInitial, route_get, data_typed["v"]) as typeof stateInitial;

//       onStateChange(
//         stateInitial,
//         returnObject.counter
//       );
//       returnObject.counter += 1;
//     }
//   } else if (checkKeys(["event_result"], Object.keys(data))) {
//     // TODO: Find a way to return the event result.

//     const data_typed = data as { event_result : compositionObjectType };
//     // final_output = parsedResponse["event_result"];
//     console.log("Event result:", data_typed["event_result"]);
//   }

//   if (checkKeys(["toolchain_session_id"], Object.keys(data))) {
//     returnObject.sessionId = data.toolchain_session_id as string;
//   }

//   return returnObject;
// }


export function get_value(data: compositionType, 
                          index: number | string): substituteAny {
	if (Array.isArray(data)) {
    const indexWrapped = ((index as number >= 0)?index:(data.length + (index as number))) as number;
		return data[indexWrapped];
	} else if (data instanceof Map) {
		if (typeof index !== "string") {
			throw new Error("Index over a map must be a string");
		}
		return data.get(index);
	} else if (typeof data === "object") {
		return data[index as string];
	} else {
		throw new Error("Invalid data type used in get_value");
	}
}


/**
 * Sets the value of an element in the given data structure.
 * 
 * @param data - The data structure to modify.
 * @param index - The index or key of the element to set.
 * @param value - The new value to assign to the element.
 * @returns The modified data structure.
 * @throws {Error} If the data type is invalid or the index is not of the correct type.
 */
export function set_value(
    data: compositionType,
    index: number | string,
    value: substituteAny
): typeof data {
	if (Array.isArray(data)) {
		data[index as number] = value;
	} else if (data instanceof Map) {
		if (typeof index !== "string") {
			throw new Error("Index over a map must be a string");
		}
		data.set(index, value);
	} else if (typeof data === "object") {
		data[index as string] = value;
	} else {
		throw new Error("Invalid data type used in set_value");
	}

	return data;
}

export function delete_value(
    data: compositionType,
    index: number | string
): typeof data {
	if (Array.isArray(data)) {
		data.splice(index as number, 1);
	} else if (data instanceof Map) {
		if (typeof index !== "string") {
			throw new Error("Index over a map must be a string");
		}
		data.delete(index);
	} else if (typeof data === "object") {
		delete data[index as string];
	} else {
		throw new Error("Invalid data type used in delete_value");
	}

	return data;
}


export function appendInRoute(
    objectForStaticRoute: substituteAny,
    route: Route,
    value: substituteAny,
    onlyAdd: boolean = false
): compositionType {
	if (route.length > 0) {
		set_value(objectForStaticRoute as compositionType, route[0], appendInRoute(
			get_value(objectForStaticRoute as compositionType, route[0]) as compositionType,
			route.slice(1),
			value,
			onlyAdd
		));
	} else {
		if (onlyAdd) {
			if (Array.isArray(objectForStaticRoute)) {
        // TODO: This is causing issues. It is supposed to be equivalent to the following python:
        // `objectForStaticRoute += value`;
        // console.log("APPEND CONDITION 1", objectForStaticRoute, value);
        objectForStaticRoute = objectForStaticRoute.concat(value);
        // console.log("AFTER:", objectForStaticRoute)
        // if (Array.isArray(value)) {
        //   objectForStaticRoute.concat(value);
        // } else {
        //   objectForStaticRoute.push(value);
        // }
			} else if (typeof objectForStaticRoute === "string") {
				objectForStaticRoute += value as string;
			} else {
				// Throw error
				throw new Error("Invalid data type used in appendInRoute");
			}
		} else if (Array.isArray(objectForStaticRoute)) {

      // console.log("APPEND CONDITION 2", objectForStaticRoute, value);
			objectForStaticRoute.push(value);
      // console.log("AFTER:", objectForStaticRoute)
		} else {

			if (typeof objectForStaticRoute === "string") {
			
				objectForStaticRoute += value;
			} else if (Array.isArray(objectForStaticRoute)) {
				objectForStaticRoute.push(value);
			}
		}
	}

	return objectForStaticRoute as compositionType;
}

export function retrieveValueFromObj(
	input: compositionType,
	directory: string | number | (string | number)[]
): substituteAny {
	try {
		if (typeof directory === "string" || typeof directory === "number") {
			return get_value(input, directory);
		} else {
			let currentDict : compositionType = input;
			for (const entry of directory) {
					currentDict = get_value(currentDict, entry) as compositionType;
			} 
			return currentDict;
		}
	} catch (error) {
		// throw new Error("Key not found");
    return undefined;
	}
}

export function runDeleteState(
    stateInput: compositionType,
    deleteStates: deleteStateGenericType
): typeof stateInput {
	if (Array.isArray(deleteStates)) {
		for (const deleteState of deleteStates) {
			stateInput = runDeleteState(stateInput, deleteState);
			// console.log("State input after array delete with pair", deleteState, ": ", JSON.stringify(stateInput))
		}
	// deleteStates is a map
	// } else if (deleteStates instanceof Map) {
			// for (const key of deleteStates.keys()) {
			//     // stateInput = delete_value(stateInput, key) as typeof stateInput;
			//     stateInput = runDeleteState(stateInput, key) as typeof stateInput;
			// }
	// deleteStates is an object
	} else if (typeof deleteStates === "object") {

		for (const [key, value] of Object.entries(deleteStates)) {
			// stateInput = delete_value(stateInput, key) as typeof stateInput;
			stateInput = set_value(
				stateInput, 
				key, 
				runDeleteState(
					get_value(stateInput, key) as compositionType, value
				)
			) as typeof stateInput;
			// console.log("State input after object delete with pair", key, value, ": ", JSON.stringify(stateInput))

		}
	// } else if (typeof deleteStates === "string" || typeof deleteStates === "number") {
			
	} else {
		stateInput = delete_value(stateInput, deleteStates) as typeof stateInput;
		// console.log("State input after delete of key", deleteStates, ": ", JSON.stringify(stateInput))
	}

	return stateInput;
}




/*
 * This function updates the stateInput with the updateStateInput, same as python dict.update()
 * Needs to account for mixed types of objects and maps.
 */
export function updateObjects(
    stateInput: compositionObjectType,
    updateStateInput: compositionObjectType
) : substituteAny {
	if (typeof updateStateInput === "object" && typeof stateInput === "object") {
		for (const [key, value] of Object.entries(updateStateInput)) {

			if (typeof value === "object" && typeof get_value(stateInput, key) === "object") {
				stateInput = set_value(
					stateInput,
					key,
					updateObjects(
						get_value(stateInput, key) as compositionObjectType,
						value as compositionObjectType
					)
				) as typeof stateInput;
			} else {
				stateInput = set_value(stateInput, key, value) as typeof stateInput;
			}
		}
	} else {
			const updateStateMap = updateStateInput as Map<string, substituteAny>;
			for (const key of updateStateMap.keys()) {
					const updateValue = updateStateMap.get(key) as substituteAny;
					if (typeof updateValue as substituteAny !== "object" && !(updateValue instanceof Map)) {
							stateInput = set_value(
									stateInput, 
									key, 
									updateObjects(
											get_value(stateInput, key) as compositionObjectType,
											updateValue as compositionObjectType
									)
							) as typeof stateInput;
					}
			}
	}
	return stateInput;
}

export function runStateDiff(
    stateInput: compositionObjectType,
    stateDiffSpecs: {
        append_routes?: Route[];
        append_state?: compositionObjectType;
        update_state?: compositionObjectType;
        delete_state?: deleteStateListType;
    }
): typeof stateInput {
	const appendRoutes = stateDiffSpecs.append_routes || [];
	const appendState = stateDiffSpecs.append_state || {};
	const updateState = stateDiffSpecs.update_state || {};
	const deleteStates = stateDiffSpecs.delete_state || [];

	for (const route of appendRoutes) {
		const valGet = retrieveValueFromObj(appendState, route);
		stateInput = appendInRoute(stateInput, route, valGet, true) as typeof stateInput;
	}

	stateInput = updateObjects(stateInput, updateState) as typeof stateInput;

	for (const deleteState of deleteStates) {
		stateInput = runDeleteState(stateInput, deleteState) as typeof stateInput;
	}

	return stateInput;
}

export function checkKeys(keys1: string[], keys2: string[]): boolean {
	return keys1.sort().join() === keys2.sort().join();
}


export function runUnitTestForDiffFunctions() {
	const dict1 = {'a': 1, 'b': {'x': 'hello!!', 'y': [1, 2, 3]}, 'c': 'world_2', 'e': {'x': 2, 'y': 3}, 'f': {'x': 2}, 'z': 6};
	
	const dict2 = {'b': {'x': 'hello'}, 'c': 'world', 'd': 4, 'f': {'x': 2, 'y': 3}, 'z': 6};

	const dict2Map = new Map(Object.entries(dict2));

	const diff_append_routes = [['b', 'x'], ['c']] ;
	const diff_append = {'b': {'x': '!!'}, 'c': '_2'} ;
	const diff_update = {'a': 1, 'b': {'y': [1, 2, 3]}, 'e': {'x': 2, 'y': 3}} ;
	const diff_delete = ['d', {'f': ['y']}] ;

	const result = runStateDiff(dict2Map, {
		append_routes: diff_append_routes, 
		append_state: diff_append, 
		update_state: diff_update, 
		delete_state: diff_delete
	});

	if (result === dict1) {
		console.log("Test passed");
	} else {
		console.error("Test failed");
		console.error("Expected:", JSON.stringify(dict1));
		console.error("Got:", result);
	}

}