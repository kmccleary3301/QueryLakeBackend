from sqlmodel import Session
from ..typing.config import AuthType
from ..api.single_user_auth import get_user
from ..api.user_auth import get_user_external_providers_dict
import openai
from typing import AsyncIterator, AsyncGenerator, Dict, List, Any, Callable
import tiktoken
from openai import AzureOpenAI

def num_tokens_from_string_tiktoken(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def count_tokens_openai(text: str, model: str) -> int:
    if any([model.startswith(prefix) for prefix in [
        "gpt-4o", "o1"
    ]]):
        return num_tokens_from_string_tiktoken(text, "o200k_base")
    elif any([model.startswith(prefix) for prefix in [
        "gpt-4", "gpt-3.5"
    ]]):
        return num_tokens_from_string_tiktoken(text, "cl100k_base")


async def stream_openai_response(
    messages: List[Dict[str, str]],
    model: str,
    api_key: str,
    set_input_token_count: Callable[[int], None],
    model_parameters: Dict[str, Any] = None,
    organization_id: str = None,
    base_url: str = None,
) -> AsyncGenerator[str, None]:
    """
    Stream responses from OpenAI models in a standardized format.
    """
    auth_args = {"api_key": api_key}
    if organization_id:
        auth_args["organization"] = organization_id

    model_parameters = model_parameters or {}
    
    client = openai.OpenAI(
        api_key=api_key,
        base_url=base_url
    )
    
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
        stream_options={"include_usage": True},
        **model_parameters
    )
    
    input_tokens_called = False
    
    for chunk in stream:
        if not input_tokens_called and chunk.usage is not None and chunk.usage.prompt_tokens is not None:
            set_input_token_count(chunk.usage.prompt_tokens)
            
        if chunk.choices is not None and \
            len(chunk.choices) > 0 and \
            chunk.choices[0].delta.content is not None:
            
            yield chunk.choices[0].delta.content
        
            
async def stream_azure_response(
    messages: List[Dict[str, str]],
    model: str,
    endpoint: str,
    api_key: str,
    model_parameters: Dict[str, Any] = None,
    organization_id: str = None,
) -> AsyncGenerator[str, None]:
    """
    Stream responses from OpenAI models in a standardized format.
    """
    auth_args = {"api_key": api_key}
    if organization_id:
        auth_args["organization"] = organization_id

    model_parameters = model_parameters or {}
    
    client = AzureOpenAI(
        api_key=api_key,
        api_version="2024-06-01",
        azure_endpoint=endpoint
    )
    
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
        **model_parameters
    )
    
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            yield chunk.choices[0].delta.content


async def openai_llm_generator(
    request_dict: dict,
    model: str,
    external_api_key: str,
    set_input_token_count: Callable[[int], None],
    external_organization_id: str = None,
    base_url: str = None,
) -> AsyncIterator[str]:
    """
    Create a generator for openAI language model.
    """
    if "question" in request_dict:
        messages = [{"role": "user", "content": request_dict.pop("question")}]
    elif "chat_history" in request_dict:
        messages = request_dict.pop("chat_history")
    
    model_parameters = request_dict.pop("model_parameters", {})
    
    async for content in stream_openai_response(
        messages=messages,
        model=model,
        api_key=external_api_key,
        set_input_token_count=set_input_token_count,
        model_parameters=model_parameters,
        base_url=base_url,
        organization_id=external_organization_id,
    ):
        yield content


async def azure_llm_generator(
    request_dict: dict,
    model: str,
    endpoint: str,
    external_api_key: str,
    external_organization_id: str = None,
) -> AsyncIterator[str]:
    """
    Create a generator for openAI language model.
    """
    if "question" in request_dict:
        messages = [{"role": "user", "content": request_dict.pop("question")}]
    elif "chat_history" in request_dict:
        messages = request_dict.pop("chat_history")

    model_parameters = request_dict.pop("model_parameters", {})
    
    async for content in stream_azure_response(
        messages=messages,
        model=model,
        endpoint=endpoint,
        api_key=external_api_key,
        model_parameters=model_parameters,
        organization_id=external_organization_id
    ):
        yield content

provider_base_url_lookups = {
    "openai": [None, "OpenAI"],
    "deepinfra": ["https://api.deepinfra.com/v1/openai", "DeepInfra"],
}

def external_llm_generator(
    database : Session,
    auth : AuthType,
    provider : str,
    model : str,
    request_dict : dict,
    set_input_token_count: Callable[[int], None],
    additional_auth: dict = None,
) -> AsyncIterator:
    """
    Create a generator for an external language model provider.
    """
    
    (user, user_auth) = get_user(database, auth)
    
    assert not user.external_providers_encrypted is None, "No external provider credentials on account."
    
    external_providers_user_credentials = get_user_external_providers_dict(database, auth)
    
    # print("Ext LLM 2 external_providers:", external_providers)
    
    if provider in ["openai", "deepinfra"]:
        provider_specs = provider_base_url_lookups.get(provider)
        return openai_llm_generator(
            request_dict,
            model,
            external_providers_user_credentials[provider_specs[1]],
            base_url=provider_specs[0],
            set_input_token_count=set_input_token_count,
            **(additional_auth or {}),
        )
    elif provider == "azure": 
        # TODO: This currently doesn't work
        # We need a clean way of specifying an endpoint and model
        model_split = model.split("/")
        assert len(model_split) >= 2, "Invalid model format for azure call. Must be in the format `endpoint/model`"
        model_split = [model_split[0], "/".join(model_split[1:])]
        endpoint_sub, model_sub = model_split[0], model_split[1]
        return azure_llm_generator(
            request_dict,
            model=model_sub,
            endpoint=endpoint_sub,
            external_api_key=external_providers_user_credentials['Azure'],
            **(additional_auth or {})
        )
    else:
        raise ValueError("Invalid provider.")
    
def external_llm_count_tokens(text: str, model: str):
    model_specified = model.split("/")
    if model_specified[0] == "openai":
        return count_tokens_openai(text, model_specified[1])
    
    else:
        raise Exception(f"Invalid provider specified ({model_specified[0]}).")