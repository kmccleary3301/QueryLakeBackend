from sqlmodel import Session, select, and_, not_
from ..typing.config import AuthType
from ..api.single_user_auth import process_input_as_auth_type, get_user
from ..database import encryption
from ..api.hashing import hash_function
from ..api.user_auth import get_user_external_providers_dict
import openai
from typing import AsyncIterator, AsyncGenerator, Dict, List, Any
import tiktoken
import json

EXTERNAL_PROIVDERS = [
    "openai"
]


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
    
    client = openai.OpenAI(
        api_key=api_key,
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
    
    async for content in stream_openai_response(
        messages=messages,
        model=model,
        api_key=external_api_key,
        model_parameters=model_parameters,
        organization_id=external_organization_id
    ):
        yield content


def external_llm_generator(
    database : Session,
    auth : AuthType,
    provider : str,
    model : str,
    request_dict : dict,
    additional_auth: dict = None
) -> AsyncIterator:
    """
    Create a generator for an external language model provider.
    """
    
    assert provider in EXTERNAL_PROIVDERS, f"Invalid external provider. Valid providers are: {EXTERNAL_PROIVDERS}"
    
    (user, user_auth) = get_user(database, auth)
    
    assert not user.external_providers_encrypted is None, "No external provider credentials on account."
    
    external_providers = get_user_external_providers_dict(database, auth)
    
    print("Ext LLM 2 external_providers:", external_providers)
    
    if provider == "openai":
        return openai_llm_generator(
            request_dict,
            model,
            external_providers['OpenAI'],
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