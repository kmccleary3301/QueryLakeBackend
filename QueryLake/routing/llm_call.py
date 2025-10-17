import asyncio
import json
import logging
from copy import deepcopy

from QueryLake.api import api
from QueryLake.database import sql_db_tables
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask
from typing import Union, List, Callable, Dict, Awaitable, Optional
from ray.serve.handle import DeploymentHandle, DeploymentResponseGenerator


from QueryLake.typing.config import AuthType
from QueryLake.operation_classes.ray_vllm_class import format_chat_history
from QueryLake.typing.function_calling import FunctionCallDefinition

from QueryLake.misc_functions.external_providers import external_llm_generator, external_llm_count_tokens
from QueryLake.misc_functions.server_class_functions import stream_results_tokens, find_function_calls, basic_stream_results
from QueryLake.runtime.signals import JobSignal
from sqlmodel import select

logger = logging.getLogger(__name__)


async def llm_call(
    self, # Umbrella class, can't type hint because of circular imports
    auth : AuthType, 
    question : str = None,
    model_parameters : dict = {},
    model : str = None,
    lora_id : str = None,
    sources : List[dict] = [],
    chat_history : List[dict] = None,
    stream_callables: Dict[str, Awaitable[Callable[[str], None]]] = None,
    functions_available: List[Union[FunctionCallDefinition, dict]] = None,
    only_format_prompt: bool = False,
    job_signal: Optional[JobSignal] = None,
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
    # When invoked from the Toolchains v2 runtime, we receive `stream_callables` so that tokens
    # can be fanned out via SSE. In this case, do not return a FastAPI StreamingResponse;
    # instead, consume the generator here and trigger the provided callbacks.
    if stream_callables is not None:
        return_stream_response = False
    
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
        gen = external_llm_generator(
            self.database,
            auth,
            provider=model_specified[0],
            model="/".join(model_specified[1:]),
            request_dict=model_parameters,
            set_input_token_count=set_input_token_count,
        )
        if job_signal is not None:
            await job_signal.set_request_id(f"external:{model_choice}")
    else:
        model_entry : sql_db_tables.model = self.database.exec(select(sql_db_tables.model)
                                            .where(sql_db_tables.model.id == model_choice)).first()
        
        
        assert not model_entry is None, f"Model choice [{model_choice}] not found in database"
        assert model_entry.id == model_choice, "Model choice and model ID do not match"
        
        model_parameters_true = {
            **json.loads(model_entry.default_settings),
        }
        
        model_parameters_true.update(model_parameters)
        
        
        
        stop_sequences = model_parameters_true["stop"] if "stop" in model_parameters_true else []
        
        assert self.config.enabled_model_classes.llm, "LLMs are disabled on this QueryLake Deployment"
        assert model_choice in self.llm_handles, f"Model choice [{model_choice}] not available for LLMs"
        
        llm_handle : DeploymentHandle = self.llm_handles[model_choice]
        
        gen : DeploymentResponseGenerator = (
            llm_handle.get_result_loop.remote(
                deepcopy(model_parameters_true), 
                sources=sources, 
                functions_available=functions_available,
                lora_id=lora_id,
                job_id=job_signal.job_id if job_signal else None,
            )
        )
        # print("GOT LLM REQUEST GENERATOR WITH %d SOURCES" % len(sources))
        
        if self.llm_configs[model_choice].engine == "exllamav2":
            async for result in gen:
                logger.debug("exllamav2 generator output: %s", result)
        
        
        # input_token_count = self.llm_count_tokens(model_choice, model_parameters_true["text"])
        generated_prompt = await self.llm_handles_no_stream[model_choice].generate_prompt.remote(
            deepcopy(model_parameters_true), 
            sources=sources, 
            functions_available=functions_available
        )
        if only_format_prompt:
            return generated_prompt
        input_token_count = generated_prompt["tokens"]

        if job_signal is not None:
            handle_no_stream = self.llm_handles_no_stream[model_choice]

            async def abort_remote() -> None:
                await handle_no_stream.cancel_job.remote(job_signal.job_id)

            await job_signal.on_stop(abort_remote)

            request_id: Optional[str] = None
            for _ in range(5):
                request_id = await handle_no_stream.get_request_id.remote(job_signal.job_id)
                if request_id:
                    break
                await asyncio.sleep(0)
            if request_id:
                await job_signal.set_request_id(request_id)
    
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
