from fastapi import Request
from ..api.single_user_auth import get_user, AuthType2
from ray.serve.handle import DeploymentHandle
from fastapi.responses import StreamingResponse, JSONResponse
import json
from pydantic import BaseModel
from typing import Any, List
from .misc_models import embedding_call

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
    raw_request : Request
):
    """
    Chat completions endpoint for the UmbrellaClass deployment.
    
    TODO: Add token tracking similar to the legacy endpoint.
    """
    
    try:
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
        
        
        generator = llm_handle.create_chat_completion_original.remote(request_body, mock_request)
        
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
                try:
                    error = json.loads(message_content)
                    return JSONResponse(content=error, status_code=error.get("code", 500))
                except json.JSONDecodeError:
                    return JSONResponse(
                        content={
                            "object": "error",
                            "type": "UpstreamError",
                            "message": message_content,
                        },
                        status_code=500,
                    )
            
            elif first_yield_value.startswith(MESSAGE_PREPENDS["standard"]):
                message_content = first_yield_value[len(MESSAGE_PREPENDS["standard"])+3:]
                try:
                    standard_result = json.loads(message_content)
                    return JSONResponse(content=standard_result)
                except json.JSONDecodeError:
                    return JSONResponse(
                        content={
                            "object": "error",
                            "type": "UpstreamError",
                            "message": message_content,
                        },
                        status_code=500,
                    )

        return JSONResponse(content={"error": "First yield value not a string"})
    
    except Exception as e: # Yield an OpenAI Error Object
        return JSONResponse(content={
            "object": "error",
            "type": type(e).__name__,
            "message": str(e)
        }, status_code=500)


async def openai_create_embedding(
    umbrella_class,
    raw_request : Request
):
    """
    Embedding endpoint for the UmbrellaClass deployment.
    """
    
    
    try:
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
        
        
        model_choice = request_body.get("model")
        
        assert not model_choice is None, "Model choice not specified"
    
        all_strings: List[str] = []
        if "messages" in request_body:
            messages = request_body.get("messages") or []
            for msg in messages:
                if not isinstance(msg, dict):
                    continue
                content = msg.get("content")
                if isinstance(content, str):
                    all_strings.append(content)
                elif isinstance(content, list):
                    for part in content:
                        if not isinstance(part, dict):
                            continue
                        if part.get("type") == "text" and isinstance(part.get("text"), str):
                            all_strings.append(part["text"])
        elif "input" in request_body:
            raw_input = request_body.get("input")
            if isinstance(raw_input, list):
                all_strings = [str(e) for e in raw_input]
            elif raw_input is not None:
                all_strings = [str(raw_input)]
        else:
            raise ValueError("No embedding input provided (expected `input` or `messages`).")
        
        (all_embeddings, total_tokens) = await embedding_call(
            umbrella_class, auth_header, all_strings, model_choice, return_tokens_usage=True
        )
        
        final_response = {
            "object": "list",
            "model": model_choice,
            "data": [
                {
                    "object": "embedding",
                    "embedding": embedding,
                    "index": emb_i,
                }
                for emb_i, embedding in enumerate(all_embeddings)
            ],
            "usage": {
                "prompt_tokens": total_tokens,
                "total_tokens": total_tokens,
            },
        }
        
        return JSONResponse(content=final_response)
        
    except Exception as e:
        return JSONResponse(content={
            # "object": "error",
            "error": type(e).__name__,
            "message": str(e)
        }, status_code=500)
        
        
    
    
    
    
    
    
    
