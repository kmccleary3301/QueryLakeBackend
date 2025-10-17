import json
import logging
import traceback
from copy import deepcopy
from typing import Any, Awaitable, Callable
from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect

from QueryLake.typing.config import AuthType
from QueryLake.typing.toolchains import *
from QueryLake.operation_classes.toolchain_session import ToolchainSession
from QueryLake.api.single_user_auth import process_input_as_auth_type, get_user

logger = logging.getLogger(__name__)

async def toolchain_websocket_handler(
    umbrella_class, 
    ws: WebSocket,
    clean_function_arguments_for_api: Callable[[dict, dict, dict], dict],
    fetch_toolchain_session: Callable[[Any], ToolchainSession],
    create_toolchain_session: Callable[[Any], ToolchainSession],
    toolchain_file_upload_event_call: Callable[[Any], Awaitable[dict]],
    save_toolchain_session: Callable[[Any], Awaitable[None]],
    toolchain_event_call: Callable[[Any], Awaitable[dict]],
):
    """
    Toolchain websocket API point.
    On connection, there is no session.
    All client messages must decode as a JSON with the fields `command`, `arguments`, and `auth`.
    The `auth` field is a dictionary with the fields `username` and `password_prehash`.
    The client can send the following to load an existing toolchain:
    ```json
    {
        "command": "toolchain/load",
        "arguments": {
            "session_id": "..."
        }
    }
    ```
    Or, the client can send the following to create one:
    ```json
    {
        "command": "toolchain/create",
        "arguments": {
            "toolchain_id": "..."
        }
    }
    ```
    Some other commands include:
    * `toolchain/retrieve_files` - Retrieve files belonging to the toolchain session.
    * `toolchain/file_upload_event_call` - Call the file upload event node of the toolchain.
    * `toolchain/entry` - Call the entry node of the toolchain.
    * `toolchain/event` - Call an event node of the toolchain.
    
    All session state updates will be sent back to the client.
    
    For nodes with stream output, the websocket will send back a mapping id to the state variable/index.
    The client will then be sent each piece of the generated output as a json, which also includes the mapping id.
    
    For file uploads, the client must make a separate POST request, then do the following:
    On upload completion, the file will be added to the database, where the client must then send
    the new id to the websocket via `toolchain/file_upload_event_call`.
    """
    
    system_args = {
        **umbrella_class.default_function_arguments,
        "ws": ws,
    }
    
    await ws.accept()
    
    toolchain_session : ToolchainSession = None
    old_toolchain_state = None
    
    await ws.send_text((json.dumps({"success": True})).encode("utf-8"))
    
    async def reset_session_state():
        """
        For when there is an error, restore the toolchain 
        session to its previous state before an event was called.
        """
        nonlocal toolchain_session, ws, old_toolchain_state
        if not toolchain_session is None and not old_toolchain_state is None:
            toolchain_session.state = deepcopy(old_toolchain_state)
            await ws.send_text(json.dumps({"state": toolchain_session.state}))
            old_toolchain_state = None
    
    
    try:
        while True:
            text = await ws.receive_text()
            # print("Got text:", text)
            try:
                arguments_websocket = json.loads(text)
                assert "auth" in arguments_websocket, "No auth provided"
                assert "command" in arguments_websocket, "No command provided"
                command : str = arguments_websocket["command"]
                auth : AuthType = arguments_websocket["auth"]
                
                auth = process_input_as_auth_type(auth)
                
                arguments : dict = arguments_websocket["arguments"]
                
                (_, _) = get_user(umbrella_class.database, auth)
                
                arguments.update({"auth": auth})
                
                assert command in [
                    "toolchain/load",
                    "toolchain/create",
                    "toolchain/file_upload_event_call",
                    "toolchain/event",
                ], "Invalid command"
                
                if command == "toolchain/load":
                    if not toolchain_session is None and toolchain_session.first_event_fired:
                        await save_toolchain_session(umbrella_class.database, toolchain_session)
                        toolchain_session = None
                    true_args = clean_function_arguments_for_api(system_args, arguments, function_object=fetch_toolchain_session)
                    toolchain_session : ToolchainSession = fetch_toolchain_session(**true_args)
                    result = {
                        "success": True,
                        "loaded": True,
                        "toolchain_session_id": toolchain_session.session_hash,
                        "toolchain_id": toolchain_session.toolchain_id,
                        "state": toolchain_session.state,
                        "first_event_fired": toolchain_session.first_event_fired
                    }
                
                elif command == "toolchain/create":
                    if not toolchain_session is None and toolchain_session.first_event_fired:
                        await save_toolchain_session(umbrella_class.database, toolchain_session)
                        toolchain_session = None
                    true_args = clean_function_arguments_for_api(system_args, arguments, function_object=create_toolchain_session)
                    toolchain_session : ToolchainSession = create_toolchain_session(**true_args)
                    result = {
                        "success": True,
                        "toolchain_session_id": toolchain_session.session_hash,
                        "toolchain_collection_id": toolchain_session.collection_id,
                        "state": toolchain_session.state,
                    }
                
                elif command == "toolchain/file_upload_event_call":
                    old_toolchain_state = deepcopy(toolchain_session.state)
                    true_args = clean_function_arguments_for_api(system_args, arguments, function_object=toolchain_file_upload_event_call)
                    result = await toolchain_file_upload_event_call(**true_args, session=toolchain_session)
                # Entries are deprecated.
                # elif command == "toolchain/entry":
                #     true_args = clean_function_arguments_for_api(system_args, arguments, function_object=api.toolchain_entry_call)
                #     result = await api.toolchain_entry_call(**true_args, session=toolchain_session)

                elif command == "toolchain/event":
                    old_toolchain_state = deepcopy(toolchain_session.state)
                    true_args = clean_function_arguments_for_api(system_args, arguments, function_object=toolchain_event_call)
                    event_result = await toolchain_event_call(**true_args, system_args=system_args, session=toolchain_session)
                    result = {"event_result": event_result}
                    toolchain_session.first_event_fired = True
                    
                elif command == "toolchain/update_local_cache":
                    assert "local_cache" in arguments, "No local cache provided"
                    toolchain_session.local_cache = arguments["local_cache"]
                    result = {"success": True}
                    
                if toolchain_session.first_event_fired:
                    logger.debug(
                        "Persisting toolchain session %s after event",
                        getattr(toolchain_session, "session_id", "<unknown>"),
                    )
                    await save_toolchain_session(umbrella_class.database, toolchain_session)
                
                await ws.send_text((json.dumps(result)).encode("utf-8"))
                await ws.send_text((json.dumps({"ACTION": "END_WS_CALL"})).encode("utf-8"))
                
                del result
                
                old_toolchain_state = None
                # await api.save_toolchain_session(self.database, toolchain_session)
            
            except WebSocketDisconnect:
                logger.info("Toolchain websocket disconnected by client")
                raise WebSocketDisconnect
            except Exception as e:
                logger.exception("Error during toolchain websocket handling: %s", e)
                umbrella_class.database.rollback()
                umbrella_class.database.flush()
                
                error_message = str(e)
                stack_trace = traceback.format_exc()
                await ws.send_text(json.dumps({"error": error_message, "trace": stack_trace}))
                await ws.send_text((json.dumps({"ACTION": "END_WS_CALL_ERROR"})))
                await reset_session_state()
    except WebSocketDisconnect as e:
        logger.info("Toolchain websocket disconnected")
        if not toolchain_session is None:
            logger.debug(
                "Unloading toolchain session %s after disconnect",
                getattr(toolchain_session, "session_id", "<unknown>"),
            )
            
            if toolchain_session.first_event_fired:
                await save_toolchain_session(umbrella_class.database, toolchain_session)
                
            toolchain_session.write_logs()
            toolchain_session = None
            del toolchain_session
