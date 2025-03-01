import json
from copy import deepcopy

from QueryLake.api import api
from QueryLake.database import sql_db_tables
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask
from typing import Union, List, Callable, Dict, Awaitable
from ray.serve.handle import DeploymentHandle, DeploymentResponseGenerator


from QueryLake.typing.config import AuthType
from QueryLake.operation_classes.ray_vllm_class import format_chat_history
from QueryLake.typing.function_calling import FunctionCallDefinition

from QueryLake.misc_functions.external_providers import external_llm_generator, external_llm_count_tokens
from QueryLake.misc_functions.server_class_functions import stream_results_tokens, find_function_calls, basic_stream_results
from sqlmodel import select


async def llm_call(
    self, # Umbrella class, can't type hint because of circular imports
    auth : AuthType, 
    question : str = None,
    model_parameters : dict = {},
    model : str = None,
    sources : List[dict] = [],
    chat_history : List[dict] = None,
    stream_callables: Dict[str, Awaitable[Callable[[str], None]]] = None,
    functions_available: List[Union[FunctionCallDefinition, dict]] = None,
    only_format_prompt: bool = False
):
    """
    Call an LLM model, possibly with parameters.
    """
    (_, user_auth, original_auth, auth_type) = api.get_user(self.database, auth, return_auth_type=True)
    
    if not question is None:
        chat_history = [{"role": "user", "content": question}]
    
    if not chat_history is None:
        model_parameters["chat_history"] = chat_history
    
    on_new_token = None
    if not stream_callables is None and "output" in stream_callables:
        on_new_token = stream_callables["output"]
    
    if not model is None:
        model_choice = model
    else:
        model_choice : str = model_parameters.pop("model", self.config.default_models.llm)
    
    model_specified = model_choice.split("/")
    return_stream_response = model_parameters.pop("stream", False)
    
    input_token_count = -1
    def set_input_token_count(input_token_value: int):
        nonlocal input_token_count
        input_token_count = input_token_value
    
    if len(model_specified) > 1:
        # External LLM provider (OpenAI, Anthropic, etc)
        
        new_chat_history = format_chat_history(
            chat_history,
            sources=sources,
            functions_available=functions_available
        )
        model_parameters.update({
            "messages": new_chat_history,
            "chat_history": new_chat_history
        })
        gen = external_llm_generator(self.database, 
                                        auth, 
                                        provider=model_specified[0],
                                        model="/".join(model_specified[1:]),
                                        request_dict=model_parameters,
                                        set_input_token_count=set_input_token_count)
    else:
        model_entry : sql_db_tables.model = self.database.exec(select(sql_db_tables.model)
                                            .where(sql_db_tables.model.id == model_choice)).first()
        
        model_parameters_true = {
            **json.loads(model_entry.default_settings),
        }
        
        model_parameters_true.update(model_parameters)
        
        
        
        stop_sequences = model_parameters_true["stop"] if "stop" in model_parameters_true else []
        
        assert self.config.enabled_model_classes.llm, "LLMs are disabled on this QueryLake Deployment"
        assert model_choice in self.llm_handles, f"Model choice [{model_choice}] not available for LLMs"
        
        llm_handle : DeploymentHandle = self.llm_handles[model_choice]
        gen : DeploymentResponseGenerator = (
            llm_handle.get_result_loop.remote(deepcopy(model_parameters_true), sources=sources, functions_available=functions_available)
        )
        # print("GOT LLM REQUEST GENERATOR WITH %d SOURCES" % len(sources))
        
        if self.llm_configs[model_choice].engine == "exllamav2":
            async for result in gen:
                print("GENERATOR OUTPUT:", result)
        
        
        # input_token_count = self.llm_count_tokens(model_choice, model_parameters_true["text"])
        generated_prompt = await self.llm_handles_no_stream[model_choice].generate_prompt.remote(
            deepcopy(model_parameters_true), 
            sources=sources, 
            functions_available=functions_available
        )
        if only_format_prompt:
            return generated_prompt
        input_token_count = generated_prompt["tokens"]
    
    if "n" in model_parameters and model_parameters["n"] > 1:
        results = []
        async for result in gen:
            results = result
        # print(results)
        return [e.text for e in results.outputs]
    
    increment_usage_args = {
        "database": self.database, 
        "auth": user_auth,
        **({"api_key_id": original_auth} if auth_type == 2 else {})
    }
    
    total_output_tokens = 0
    def increment_token_count():
        nonlocal total_output_tokens
        total_output_tokens += 1
    
    def on_finish():
        nonlocal total_output_tokens, input_token_count, user_auth, original_auth, auth_type
        api.increment_usage_tally(self.database, user_auth, {
            "llm": {
                model_choice: {
                    "input_tokens": input_token_count,
                    "output_tokens": total_output_tokens
                }
            }
        }, **({"api_key_id": original_auth} if auth_type == 2 else {}))
    
    if return_stream_response:
        if len(model_specified) == 1:
            return StreamingResponse(
                stream_results_tokens(
                    gen, 
                    self.llm_configs[model_choice],
                    on_new_token=on_new_token,
                    increment_token_count=increment_token_count,
                    encode_output=False, 
                    stop_sequences=stop_sequences
                ),
                background=BackgroundTask(on_finish)
            )
        else:
            return StreamingResponse(basic_stream_results(gen, on_new_token=on_new_token))
    else:
        results = []
        if len(model_specified) == 1:
            async for result in stream_results_tokens(
                gen, 
                self.llm_configs[model_choice],
                on_new_token=on_new_token,
                increment_token_count=increment_token_count,
                stop_sequences=stop_sequences
            ):
                results.append(result)
            on_finish()
        else:
            async for result in basic_stream_results(gen, on_new_token=on_new_token):
                results.append(result)
    
    
    text_outputs = "".join(results)
    
    call_results = {}
    if not functions_available is None:
        calls = find_function_calls(text_outputs)
        calls_possible = [
            e.name if isinstance(e, FunctionCallDefinition) else e["name"] 
            for e in functions_available 
        ]
        
        calls = [e for e in calls if ("function" in e and e["function"] in calls_possible)]
        call_results = {"function_calls": calls}
    
    return {"output": text_outputs, "output_token_count": len(results), "input_token_count": input_token_count, **call_results}
