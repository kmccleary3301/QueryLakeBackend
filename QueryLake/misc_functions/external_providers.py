from sqlmodel import Session, select, and_, not_
from ..typing.config import AuthInputType
from ..api.single_user_auth import process_input_as_auth_type, get_user
from ..database import encryption
from ..api.hashing import hash_function
from ..api.user_auth import get_user_external_providers_dict
import openai
from typing import AsyncIterator, List
import inspect
import json


EXTERNAL_PROIVDERS = [
    "openai"
]


async def openai_llm_generator(
    request_dict : dict,
    model : str,
    external_api_key : str,
    external_organization_id: str = None,
) -> AsyncIterator[str]:
    """
    Create a generator for openAI language model.
    """
    
    auth_args = {"api_key": external_api_key}
    if not external_organization_id is None:
        auth_args["organization"] = external_organization_id

    if "question" in request_dict:
        question = request_dict.pop("question")
        messages = [{"role": "user", "content": question}]
    elif "chat_history" in request_dict:
        chat_history = request_dict.pop("chat_history")
        messages = chat_history

    model_parameters = request_dict.pop("model_parameters", {})
    
    async for chunk in openai.ChatCompletion.create(
        model=model,
        messages=messages,
        **model_parameters,
        stream=True,
    ): 
        try:
            content = chunk.choices[0].delta.content
            yield content
        except:
            pass
        

def external_llm_generator(
    database : Session,
    auth : AuthInputType,
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
    
    external_providers = get_user_external_providers_dict(user_auth, user.external_providers_encrypted)
    
    if provider == "openai":
        return openai_llm_generator(
            request_dict,
            model,
            external_providers["openai"]["api_key"],
            **(additional_auth or {})
        )
    else:
        raise ValueError("Invalid provider.")
    