from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from fastapi import Request
from ..api.single_user_auth import get_user
from ray.serve.handle import DeploymentHandle
from fastapi.responses import StreamingResponse, JSONResponse
import json
from pydantic import BaseModel
from typing import Any

MESSAGE_PREPENDS = {
    "stream": ">>>>>>>>>>>STREAM",
    "error": ">>>>>>>>>>>ERROR",
    "standard": ">>>>>>>>>>>STANDARD"
}

class MockRequest(BaseModel):
    headers: Any
    state: Any


# @with_cancellation
async def openai_chat_completion(
    umbrella_class,
    request: ChatCompletionRequest,
    raw_request : Request
):
    """
    Chat completions endpoint for the UmbrellaClass deployment.
    
    TODO: Add token tracking similar to the legacy endpoint.
    """
    
    request_body = await raw_request.json()
    request_headers = raw_request.headers
    auth_header, auth_header_type = request_headers.get("authorization", ""), None
    for prefix, auth_type in [("Bearer ", "api_key"), ("Authorization: ", "oauth2")]:
        if auth_header.startswith(prefix):
            auth_header = auth_header[len(prefix):]
            auth_header_type = auth_type
            break
    
    if auth_header_type == "api_key":
        auth_header = {"api_key": auth_header}
    
    (_, auth_type) = get_user(umbrella_class.database, auth_header)
    
    model_choice : str = request_body.get("model", None)
    assert not model_choice is None, "Model choice not specified"
    
    assert model_choice in umbrella_class.llm_handles, "Model choice not found"
    llm_handle : DeploymentHandle = umbrella_class.llm_handles[model_choice]
    
    # We disguise the raw_request for the second argument.
    # The endpoint only cares about the raw request's headers and state.
    mock_request = MockRequest(headers=raw_request.headers, state=raw_request.state)
    
    
    # generator = llm_handle.create_chat_completion_new.remote(request, request)
    generator = llm_handle.create_chat_completion_original.remote(request, mock_request)
    
    # The endpoint's first yield is an indicator of response type
    # We assume a generator, but it may return a static type delivered via generator.
    # This workaround was necessary because ray forces us to choose whether we're calling
    # for a generator or a static type before making the call.
    first_yield_value = await generator.__anext__() 
    
    if isinstance(first_yield_value, str):
        if first_yield_value.startswith(MESSAGE_PREPENDS["stream"]):
            return StreamingResponse(content=generator, media_type="text/event-stream")
        
        elif first_yield_value.startswith(MESSAGE_PREPENDS["error"]):
            message_content = first_yield_value[len(MESSAGE_PREPENDS["error"])+3:]
            error = json.loads(message_content)
            return JSONResponse(content=error, status_code=error.get("code", 500))
        
        elif first_yield_value.startswith(MESSAGE_PREPENDS["standard"]):
            message_content = first_yield_value[len(MESSAGE_PREPENDS["standard"])+3:]
            standard_result = ChatCompletionResponse(**json.loads(message_content))
            return JSONResponse(content=standard_result.model_dump())

    return JSONResponse(content={"error": "First yield value not a string"})