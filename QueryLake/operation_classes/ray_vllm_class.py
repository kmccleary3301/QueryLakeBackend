import asyncio
import logging
import time

import re, json

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid
# from aphrodite import AsyncAphrodite, AsyncEngineArgs, SamplingParams
# from aphrodite.common.utils import random_uuid
from QueryLake.misc_functions.vllm_lmformating_modifed_banned_tokens import build_vllm_token_enforcer_tokenizer_data
from ray import serve

from QueryLake.misc_functions.grammar_sampling_functions import get_token_id, get_logits_processor_from_grammar_options

from QueryLake.typing.config import Padding, Model
from QueryLake.misc_functions.prompt_construction import construct_chat_history
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from typing import List, Union, AsyncGenerator, AsyncIterator
from QueryLake.typing.function_calling import FunctionCallDefinition
from QueryLake.misc_functions.server_class_functions import construct_functions_available_prompt
from huggingface_hub import snapshot_download
import os

from typing import List, Union, Optional, Dict, Any
from vllm.entrypoints.llm import ChatCompletionMessageParam, cast, is_list_of, TokensPrompt, TextPrompt, parse_chat_messages, MistralTokenizer
from vllm.entrypoints.llm import apply_mistral_chat_template, apply_hf_chat_template
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.config import ModelConfig
from pydantic import BaseModel
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    BeamSearchParams,
    RequestResponseMetadata,
)
from vllm.outputs import RequestOutput
from vllm.entrypoints.openai.serving_chat import (
    OpenAIServingChat,
    maybe_serialize_tool_calls,
    truncate_tool_call_ids
)
from vllm.entrypoints.openai.serving_models import OpenAIServingModels, BaseModelPath
from fastapi import Request
from vllm.usage.usage_lib import UsageContext
import traceback
from vllm.lora.request import LoRARequest
from QueryLake.operation_classes.runtime_introspection import RuntimeIntrospectionMixin

logger = logging.getLogger(__name__)

def encode_chat(
    llm_engine: AsyncLLMEngine,
    tokenizer: AnyTokenizer,
	messages: Union[List[ChatCompletionMessageParam],
                        List[List[ChatCompletionMessageParam]]],
	chat_template: Optional[str] = None,
	add_generation_prompt: bool = True,
	continue_final_message: bool = False,
	tools: Optional[List[Dict[str, Any]]] = None,
	mm_processor_kwargs: Optional[Dict[str, Any]] = None,
):
    
    if chat_template is None:
        raise ValueError("chat_template must be provided.")
    
    list_of_messages: List[List[ChatCompletionMessageParam]]

    # Handle multi and single conversations
    if is_list_of(messages, list):
        # messages is List[List[...]]
        list_of_messages = cast(List[List[ChatCompletionMessageParam]],
                                messages)
    else:
        # messages is List[...]
        list_of_messages = [
            cast(List[ChatCompletionMessageParam], messages)
        ]

    prompts: List[Union[TokensPrompt, TextPrompt]] = []

    for msgs in list_of_messages:

        # NOTE: _parse_chat_message_content_parts() currently doesn't
        # handle mm_processor_kwargs, since there is no implementation in
        # the chat message parsing for it.
        model_config: ModelConfig = llm_engine.engine.get_model_config()

        conversation, mm_data = parse_chat_messages(
            msgs, model_config, tokenizer, "openai")

        prompt_data: Union[str, List[int]]
        if isinstance(tokenizer, MistralTokenizer):
            prompt_data = apply_mistral_chat_template(
                tokenizer,
                messages=msgs,
                chat_template=chat_template,
                add_generation_prompt=add_generation_prompt,
                continue_final_message=continue_final_message,
                tools=tools,
            )
        else:
            
            logger.debug("Conversation parsed for chat template: %s", conversation)
            
            prompt_data = apply_hf_chat_template(
                tokenizer,
                conversation=conversation,
                chat_template=chat_template,
                add_generation_prompt=add_generation_prompt,
                continue_final_message=continue_final_message,
                tools=tools,
            )

        prompt: Union[TokensPrompt, TextPrompt]
        if is_list_of(prompt_data, int):
            prompt = TokensPrompt(prompt_token_ids=prompt_data)
        else:
            prompt = TextPrompt(prompt=prompt_data)

        if mm_data is not None:
            prompt["multi_modal_data"] = mm_data

        if mm_processor_kwargs is not None:
            prompt["mm_processor_kwargs"] = mm_processor_kwargs

        prompts.append(prompt)
	
    return prompts

def format_chat_history(chat_history : List[dict],
                        sources : List[dict] = [],
                        functions_available: List[Union[FunctionCallDefinition, dict]] = None):
    
    for i in range(len(sources)):
        if isinstance(sources[i], BaseModel):
            sources[i] = sources[i].model_dump()
    
    # Turn chat history entry content fields into lists.
    chat_history = [
        e if not isinstance(e["content"], str) else {
            **e, 
            "content": [
                {"type": "text", "text": e["content"]}
            ]
        }
        for e in chat_history
    ]
    
    padding = 2 if (functions_available is None and len(sources) == 0) else (100 + 300 * len(sources) + 300 * (len(functions_available) if isinstance(functions_available, list) else 0))
    if len(sources) > 0:
        new_entry = {
            "type": "text",
            "text": ("SYSTEM MESSAGE - PROVIDED SOURCES\n" +
                     "Cite these sources in your response with the following notation " + 
                     "for inline citations: {cite:source_number} (i.e. {cite:3})\n" +
                     "<SOURCES>\n" +
            '\n\n'.join(['[%d] Source %d\n\n%s' % (i+1, i+1, e['text']) for i, e in enumerate(sources)]) +
            f"\n</SOURCES>\nEND SYSTEM MESSAGE\n")
        }
        chat_history[-1]["content"] = [new_entry] + chat_history[-1]["content"]
    if not functions_available is None:
        new_entry = {
            "type": "text",
            "text": f"SYSTEM MESSAGE - AVAILABLE FUNCTIONS\n<FUNCTIONS>{construct_functions_available_prompt(functions_available)}" + \
            f"\n</FUNCTIONS>\nEND SYSTEM MESSAGE\n\n"
        }
        chat_history[-1]["content"] = [new_entry] + chat_history[-1]["content"]
    
    
    stripped_chat_history = [{**e, "content": "\n".join([c["text"] for c in e["content"] if c["type"] == "text"])} for e in chat_history]
    
    # If it's all text, just return the stripped chat history.
    if all([p["type"] == "text" for e in chat_history for p in e["content"]]):
        chat_history = stripped_chat_history
    
    return chat_history


# @serve.deployment(
#     ray_actor_options={"num_gpus": 0.001}, 
#     # max_replicas_per_node=1
#     autoscaling_config={
#         "min_replicas": 0,
#         "max_replicas": 3,
#         "target_num_ongoing_requests_per_replica": 5,
#     },
# )
class VLLMDeploymentClass(RuntimeIntrospectionMixin, OpenAIServingChat):
    def __init__(
        self,
        model_config : Model,
        **kwargs
    ):
        """
        Construct a VLLM deployment.

        Refer to https://github.com/vllm-project/vllm/blob/main/vllm/engine/arg_utils.py
        for the full list of arguments.

        Args:
            model: name or path of the huggingface model to use
            download_dir: directory to download and load the weights,
                default to the default cache dir of huggingface.
            use_np_weights: save a numpy copy of model weights for
                faster loading. This can increase the disk usage by up to 2x.
            use_dummy_weights: use dummy values for model weights.
            dtype: data type for model weights and activations.
                The "auto" option will use FP16 precision
                for FP32 and FP16 models, and BF16 precision.
                for BF16 models.
            seed: random seed.
            worker_use_ray: use Ray for distributed serving, will be
                automatically set when using more than 1 GPU
            pipeline_parallel_size: number of pipeline stages.
            tensor_parallel_size: number of tensor parallel replicas.
            block_size: token block size.
            swap_space: CPU swap space size (GiB) per GPU.
            gpu_memory_utilization: the percentage of GPU memory to be used for
                the model executor
            max_num_batched_tokens: maximum number of batched tokens per iteration
            max_num_seqs: maximum number of sequences per iteration.
            disable_log_stats: disable logging statistics.
            engine_use_ray: use Ray to start the LLM engine in a separate
                process as the server process.
            disable_log_requests: disable logging requests.
        """
        self.model_config_t = model_config
        self.padding_t : Padding = model_config.padding
        self.default_model_args_t = self.model_config_t.default_parameters
        self._runtime_role = "llm"
        self._runtime_model_id = model_config.id
        self._runtime_extra_metadata = {
            "model_name": getattr(model_config, "name", model_config.id),
            "engine": getattr(model_config, "engine", "vllm"),
        }
        
        engine_args_make = {
            **kwargs,
            "gpu_memory_utilization": 0.4, # This should match the ray GPU allocation
            "enforce_eager": True,
            "disable_log_requests": True,  # Had to mute this thing because it was spamming the logs.
            **(self.model_config_t.engine_args if not self.model_config_t.engine_args is None else {})
        }
        
        # --- LoRA Configuration Start ---
        self.loras_map_t: Dict[str, Dict[str, Any]] = {}
        enable_lora = False
        if self.model_config_t.loras and len(self.model_config_t.loras) > 0:
            logger.info("Found %s LoRA adapters in configuration", len(self.model_config_t.loras))
            enable_lora = True
            # Assign unique integer IDs and store paths
            for i, lora_config in enumerate(self.model_config_t.loras):
                if lora_config.id and lora_config.system_path:
                    # Basic check if path exists (can be enhanced)
                    if not os.path.exists(lora_config.system_path):
                        logger.warning(
                            "LoRA path does not exist for adapter %s at %s",
                            lora_config.id,
                            lora_config.system_path,
                        )
                    
                    self.loras_map_t[lora_config.id] = {
                        "path": lora_config.system_path,
                        "int_id": i + 1 # VLLM uses 1-based integer IDs
                    }
                    logger.info(
                        "Registered LoRA adapter %s (int_id=%s) at %s",
                        lora_config.id,
                        i + 1,
                        lora_config.system_path,
                    )
                else:
                    logger.warning("Skipping LoRA config with missing id or system_path: %s", lora_config)

            # Set engine args for LoRA, allowing overrides from model_config.engine_args
            engine_args_make.setdefault("enable_lora", enable_lora)
            # Default max_loras to number of configured LoRAs, min 1 if enabled
            engine_args_make.setdefault("max_loras", max(1, len(self.loras_map_t)))
            # Default max_lora_rank, common value is 64
            engine_args_make.setdefault("max_lora_rank", 64)
            # Default max_cpu_loras, often same as max_loras
            engine_args_make.setdefault("max_cpu_loras", engine_args_make["max_loras"])
            
            logger.debug(
                "LoRA engine args: enable_lora=%s, max_loras=%s, max_lora_rank=%s, max_cpu_loras=%s",
                engine_args_make["enable_lora"],
                engine_args_make["max_loras"],
                engine_args_make["max_lora_rank"],
                engine_args_make["max_cpu_loras"],
            )

        elif "enable_lora" in engine_args_make and engine_args_make["enable_lora"]:
             # Handle case where enable_lora is true in engine_args but no loras defined in config
             enable_lora = True
             logger.warning("LoRA enabled via engine_args, but no adapters configured; using defaults")
             engine_args_make.setdefault("max_loras", 1) # Default to 1 if enabled but none specified
             engine_args_make.setdefault("max_lora_rank", 64)
             engine_args_make.setdefault("max_cpu_loras", engine_args_make["max_loras"])
             logger.debug(
                "LoRA engine args: enable_lora=%s, max_loras=%s, max_lora_rank=%s, max_cpu_loras=%s",
                engine_args_make["enable_lora"],
                engine_args_make["max_loras"],
                engine_args_make["max_lora_rank"],
                engine_args_make["max_cpu_loras"],
            )
        # --- LoRA Configuration End ---

        args = AsyncEngineArgs( 
            **engine_args_make,
            model=self.model_config_t.system_path,
        )
        self.context_size_t = args.max_model_len
        
        if os.path.exists(os.path.join(self.model_config_t.system_path, "tokenizer_config.json")):
            with open(os.path.join(self.model_config_t.system_path, "tokenizer_config.json"), "r") as f:
                tokenizer_config = json.load(f)
        else:
            tokenizer_config = {}
            
        self.chat_template_t = None
        if "chat_template" in tokenizer_config:
            logger.debug("Chat template found in tokenizer_config")
            self.chat_template_t = tokenizer_config["chat_template"]
        
        
        logger.info("Initializing vLLM deployment for model %s", self.model_config_t.id)
        
        self.engine_t = AsyncLLMEngine.from_engine_args(args, usage_context=UsageContext.OPENAI_API_SERVER)
        self.tokenizer_t = self.engine_t.engine.get_tokenizer()
        # tokenizer_tmp = self.engine.engine.tokenizer
        # self.special_token_ids = tokenizer_tmp.all_special_ids
        # self.space_tokens = [get_token_id(tokenizer_tmp, e) for e in ["\n", "\t", "\r", " \r"]]
        self.tokenizer_data_t = build_vllm_token_enforcer_tokenizer_data(self.engine_t.engine)
        
        self.vllm_model_config_t = self.engine_t.engine.get_model_config()
        
        self.base_model_paths_t = [
            BaseModelPath(name=model_config.id, model_path=model_config.system_path)
        ]
        
        self.openai_serving_models_t = OpenAIServingModels(
            engine_client=self.engine_t,
            model_config=self.vllm_model_config_t,
            base_model_paths=self.base_model_paths_t,
        )
        
        super().__init__(
            engine_client=self.engine_t,
            model_config=self.vllm_model_config_t,
            models=self.openai_serving_models_t,
            response_role="assistant",
            request_logger=None,
            # chat_template=None,
            chat_template=self.chat_template_t if not self.chat_template_t is None else "openai",
            chat_template_content_format="auto",
        )

        self._job_request_lock = asyncio.Lock()
        self._job_request_map: Dict[str, str] = {}
        self._publish_runtime_metadata()
    
    def count_tokens(self, input_string : str):
        return len(self.tokenizer_t(input_string)["input_ids"])
    
    def generate_prompt(self, 
                        request_dict : dict,
                        sources : List[dict] = [],
                        functions_available: List[Union[FunctionCallDefinition, dict]] = None,
                        get_multi_modal_history: bool = False):
        
        for i in range(len(sources)):
            if isinstance(sources[i], BaseModel):
                sources[i] = sources[i].model_dump()
        
        if "prompt" in request_dict:
            prompt = request_dict["prompt"]
            
            return {"formatted_prompt": prompt, "tokens": self.count_tokens(prompt)}
        elif "chat_history" in request_dict:
            chat_history = request_dict["chat_history"]
            chat_history = [
                e if not isinstance(e["content"], str) else {
                    **e, 
                    "content": [
                        {"type": "text", "text": e["content"]}
                    ]
                }
                for e in chat_history
            ]
            
            padding = 2 if (functions_available is None and len(sources) == 0) else (100 + 300 * len(sources) + 300 * (len(functions_available) if isinstance(functions_available, list) else 0))
            if len(sources) > 0:
                new_entry = {
                    "type": "text",
                    "text": ("SYSTEM MESSAGE - PROVIDED SOURCES\n<SOURCES>\n" +
                    '\n\n'.join(['[%d] Source %d\n\n%s' % (i+1, i+1, e['text']) for i, e in enumerate(sources)]) +
                    f"\n</SOURCES>\nEND SYSTEM MESSAGE\n")
                }
                chat_history[-1]["content"] = [new_entry] + chat_history[-1]["content"]
            if not functions_available is None:
                new_entry = {
                    "type": "text",
                    "text": f"SYSTEM MESSAGE - AVAILABLE FUNCTIONS\n<FUNCTIONS>{construct_functions_available_prompt(functions_available)}" + \
                    f"\n</FUNCTIONS>\nEND SYSTEM MESSAGE\n\n"
                }
                chat_history[-1]["content"] = [new_entry] + chat_history[-1]["content"]
            
            # TODO: Re-implement chopping/wrapping chat history since it no longer works.
            # TODO: Count tokens for multimodal inputs.
            multi_modal_input = any([e["type"] != "text" for c in chat_history for e in c["content"]])
            stripped_chat_history = [{**e, "content": [c for c in e["content"] if c["type"] == "text"]} for e in chat_history]
            # Join text messages into a single string if that's all there is.
            for i in range(len(stripped_chat_history)):
                if isinstance(stripped_chat_history[i]["content"], list) and all([e["type"] == "text" for e in stripped_chat_history[i]["content"]]):
                    stripped_chat_history[i]["content"] = " ".join([e["text"] for e in stripped_chat_history[i]["content"]])
            
            if multi_modal_input:
                prompt, chopped_chat_history = construct_chat_history(self.model_config_t, self.count_tokens, stripped_chat_history, request_dict["max_tokens"] - padding , return_chat_history=True)
                return {
                    "formatted_prompt": prompt,
                    "chat_history": chopped_chat_history, 
                    **({"chat_history_multi_modal": chat_history} if get_multi_modal_history else {}),
                    "tokens": 100
                }
            
            
            
            # Try normal chat formatting.
            try:
                prompt = self.tokenizer_t.apply_chat_template(
                    stripped_chat_history, 
                    add_generation_prompt=True, 
                    continue_final_message=False,
                    chat_template=self.chat_template_t
                )
                return {"formatted_prompt": prompt, "chat_history": chopped_chat_history, "tokens": self.count_tokens(prompt)}
            except:
                pass
            
            # Normal chat history without multi-modal
            prompt, chopped_chat_history = construct_chat_history(self.model_config_t, self.count_tokens, stripped_chat_history, request_dict["max_tokens"] - padding , return_chat_history=True)
            
            return {"formatted_prompt": prompt, "chat_history": chopped_chat_history, "tokens": self.count_tokens(prompt)}
            
            
        else:
            raise ValueError("Request dictionary must contain either 'prompt' or 'chat_history' key. Got: " + str(request_dict.keys()))
    
    
    async def get_result_loop(
        self, 
        request_dict : dict, 
        sources : List[dict] = [],
        functions_available: List[Union[FunctionCallDefinition, dict]] = None,
        lora_id: Optional[str] = None,
        job_id: Optional[str] = None,
    ):
        """
        This is the standard QL VLLM generator endpoint.
        It predates the OpenAI serving chat generator, and is used for
        some of QueryLake's internal code.
        """
        generate_prompt_dict = self.generate_prompt(
            request_dict, 
            sources, 
            functions_available, 
            get_multi_modal_history=True
        )
        prompt = generate_prompt_dict["formatted_prompt"]
        chopped_chat_history = generate_prompt_dict.get("chat_history") # Use .get for safety
        chat_history_multi_modal = generate_prompt_dict.get("chat_history_multi_modal")
        
        # Determine the final prompt format for the engine
        prompt_to_use: Union[str, TextPrompt, TokensPrompt]
        
        if chat_history_multi_modal:
            if not self.chat_template_t:
                # Log or raise? Multi-modal generally requires a template.
                logger.error("Multi-modal input received but no chat_template found")
                raise ValueError("Chat template required for multi-modal input.")
            logger.debug("Encoding multi-modal chat history using chat template")
            # This path was already handled correctly before, re-verified.
            prompt_to_use = encode_chat(
                self.engine_t, 
                self.tokenizer_t, 
                chat_history_multi_modal, 
                chat_template=self.chat_template_t
            )[0]
        elif self.chat_template_t and chopped_chat_history:
            # If a template exists and we have chopped history (not raw prompt input)
            # print("Encoding chopped chat history using chat template.")
            # prompt_to_use = encode_chat(
            #     self.engine_t, 
            #     self.tokenizer_t, 
            #     chopped_chat_history, 
            #     chat_template=self.chat_template_t
            # )[0]
            prompt_to_use = prompt
        else:
            # Fallback: Use the pre-formatted string from generate_prompt
            # This covers raw prompt input or cases where no chat template exists.
            logger.debug("Using pre-formatted prompt string (no template or raw input)")
            prompt_to_use = prompt
            
        # assert self.count_tokens(prompt) <= self.context_size, f"Prompt is too long."
        
        request_id = random_uuid()
        
        # --- LoRA Request Handling Start ---
        lora_request: Optional[LoRARequest] = None
        if lora_id:
            if lora_id in self.loras_map_t:
                lora_info = self.loras_map_t[lora_id]
                lora_request = LoRARequest(
                    lora_name=lora_id, # Use the provided string ID as the LoRA name
                    lora_int_id=lora_info["int_id"], # Internal integer ID
                    lora_path=lora_info["path"] # Path to adapter weights
                )
                logger.info("Using LoRA adapter %s at %s", lora_id, lora_info["path"])
            else:
                # Option: Raise error or log warning and proceed without LoRA
                logger.warning("Requested LoRA ID %s not found; continuing without LoRA", lora_id)
                # raise ValueError(f"Requested LoRA ID '{lora_id}' not found.") 
        # --- LoRA Request Handling End ---
        
        grammar_options = request_dict.pop("grammar", None)
        
        logits_processor_local = get_logits_processor_from_grammar_options(
            grammar_options,
            self.tokenizer_data_t, 
            # space_tokens=self.space_tokens, 
            # special_ids=self.special_token_ids,
        ) if not grammar_options is None else None
        
        keys = [
            'n', 
            'best_of', 
            'presence_penalty', 
            'frequency_penalty', 
            'repetition_penalty', 
            'temperature', 
            'top_p', 
            'top_k', 
            'min_p', 
            'use_beam_search', 
            'length_penalty', 
            'early_stopping', 
            'stop', 
            'stop_token_ids', 
            'include_stop_str_in_output', 
            'ignore_eos', 
            'max_tokens', 
            'logprobs', 
            'prompt_logprobs', 
            'skip_special_tokens', 
            'spaces_between_special_tokens', 
            'logits_processors'
        ]
        filtered_dict = {k: request_dict[k] for k in keys if k in request_dict}
        sampling_params = SamplingParams(**filtered_dict)
        
        if not logits_processor_local is None:
            sampling_params.logits_processors = [logits_processor_local]
        
        # The multi-modal case was handled above by encode_chat already.
        # if "chat_history_multi_modal" in generate_prompt_dict:
        #     pre_encode_prompt = generate_prompt_dict["chat_history_multi_modal"]
        #     prompt = encode_chat(
        #         self.engine_t, self.tokenizer_t, pre_encode_prompt,
        #         chat_template=self.chat_template_t
        #     )[0]
        
        results_generator = self.engine_t.generate(
            prompt_to_use, 
            sampling_params, 
            request_id, 
            lora_request=lora_request # Pass LoRA request to engine
        )
        
        return self._track_job_stream(job_id, request_id, results_generator)

    def _track_job_stream(
        self,
        job_id: Optional[str],
        request_id: str,
        generator: AsyncGenerator[Any, None],
    ) -> AsyncGenerator[Any, None]:
        if not job_id:
            return generator

        async def wrapped() -> AsyncGenerator[Any, None]:
            async with self._job_request_lock:
                self._job_request_map[job_id] = request_id
            try:
                async for item in generator:
                    yield item
            finally:
                async with self._job_request_lock:
                    existing = self._job_request_map.get(job_id)
                    if existing == request_id:
                        self._job_request_map.pop(job_id, None)

        return wrapped()

    async def cancel_job(self, job_id: str) -> bool:
        async with self._job_request_lock:
            request_id = self._job_request_map.get(job_id)
        if request_id is None:
            return False
        await self.engine_t.abort(request_id)
        async with self._job_request_lock:
            existing = self._job_request_map.get(job_id)
            if existing == request_id:
                self._job_request_map.pop(job_id, None)
        return True

    async def get_request_id(self, job_id: str) -> Optional[str]:
        async with self._job_request_lock:
            return self._job_request_map.get(job_id)
    
    
    async def create_chat_completion_original(
        self,
        request: ChatCompletionRequest,
        raw_request: Optional[Request] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Chat Completion API similar to OpenAI's API.

        See https://platform.openai.com/docs/api-reference/chat/create
        for the API specification. This API mimics the OpenAI
        Chat Completion API.
        
        
        Some modifications made, as calls to this Ray deployment must
        always recieve a generator. Hence, we wrap all responses in an async generator.
        """
        
        async def single_response_generator_protocol(
            content_message: str,
            prefix: str
        ):
            if not isinstance(content_message, str):
                if isinstance(content_message, BaseModel):
                    content_message = json.dumps(content_message.model_dump())
                else:
                    content_message = str(content_message)
            
            for _ in range(1):
                yield f"{prefix} | {content_message}"
            
        def error_generator_protocol(error_message):
            return single_response_generator_protocol(error_message, ">>>>>>>>>>>ERROR")
        
        def single_generator_protocol(error_message):
            return single_response_generator_protocol(error_message, ">>>>>>>>>>>STANDARD")
        
        async def stream_prepend_generator(
            generator: AsyncGenerator[str, None],
        ) -> AsyncGenerator[str, None]:
            yield ">>>>>>>>>>>STREAM"
            
            async for x in generator:
                yield x
                
                
        result = await self.create_chat_completion(
            request, 
            raw_request
        )
        
        if isinstance(result, AsyncGenerator):
            return stream_prepend_generator(result)
        elif isinstance(result, ChatCompletionResponse):
            return single_generator_protocol(result)
        else:
            return error_generator_protocol(result)
    
    
    async def create_chat_completion_new(
        self,
        request: ChatCompletionRequest,
        raw_request: Optional[Request] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Chat Completion API similar to OpenAI's API.

        See https://platform.openai.com/docs/api-reference/chat/create
        for the API specification. This API mimics the OpenAI
        Chat Completion API.
        
        
        Some modifications made, as calls to this Ray deployment must
        always recieve a generator. Hence, we wrap all responses in an async generator.
        """
        
        async def single_response_generator_protocol(
            content_message: str,
            prefix: str
        ):
            if not isinstance(content_message, str):
                if isinstance(content_message, BaseModel):
                    content_message = json.dumps(content_message.model_dump())
                else:
                    content_message = str(content_message)
            
            for _ in range(1):
                yield f"{prefix} | {content_message}"
            
        def error_generator_protocol(error_message):
            return single_response_generator_protocol(error_message, ">>>>>>>>>>>ERROR")
        
        def single_generator_protocol(error_message):
            return single_response_generator_protocol(error_message, ">>>>>>>>>>>STANDARD")
        
        async def stream_prepend_generator(
            generator: AsyncGenerator[str, None],
        ) -> AsyncGenerator[str, None]:
            yield ">>>>>>>>>>>STREAM"
            
            async for x in generator:
                yield x
        
        
        try:
            error_check_ret = await self._check_model(request)
            if error_check_ret is not None:
                
                # TODO: Error Generator Protocol
                # return error_check_ret
                return error_generator_protocol(error_check_ret)

            # If the engine is dead, raise the engine's DEAD_ERROR.
            # This is required for the streaming case, where we return a
            # success status before we actually start generating text :).
            if self.engine_client.errored:
                raise self.engine_client.dead_error

            try:
                (
                    lora_request,
                    prompt_adapter_request,
                ) = self._maybe_get_adapters(request)

                model_name = self.models.model_name(lora_request)

                # tokenizer = await self.engine_client.get_tokenizer(lora_request)

                tokenizer = self.tokenizer_t

                tool_parser = self.tool_parser

                # validation for OpenAI tools
                # tool_choice = "required" is not supported
                if request.tool_choice == "required":
                    
                    # TODO: Error Generator Protocol
                    return error_generator_protocol(self.create_error_response(
                        "tool_choice = \"required\" is not supported!"))

                if isinstance(tokenizer, MistralTokenizer):
                    # because of issues with pydantic we need to potentially
                    # re-serialize the tool_calls field of the request
                    # for more info: see comment in `maybe_serialize_tool_calls`
                    maybe_serialize_tool_calls(request)
                    truncate_tool_call_ids(request)

                if (request.tool_choice == "auto" and
                        not (self.enable_auto_tools and tool_parser is not None)
                        and not isinstance(tokenizer, MistralTokenizer)):
                    # for hf tokenizers, "auto" tools requires
                    # --enable-auto-tool-choice and --tool-call-parser
                    return error_generator_protocol(self.create_error_response(
                        "\"auto\" tool choice requires "
                        "--enable-auto-tool-choice and --tool-call-parser to be set"
                    ))

                tool_dicts = None if request.tools is None else [
                    tool.model_dump() for tool in request.tools
                ]
                
                (
                    conversation,
                    request_prompts,
                    engine_prompts,
                ) = await self._preprocess_chat(
                    request,
                    tokenizer,
                    request.messages,
                    chat_template=request.chat_template or self.chat_template_t,
                    chat_template_content_format=self.chat_template_content_format,
                    add_generation_prompt=request.add_generation_prompt,
                    continue_final_message=request.continue_final_message,
                    tool_dicts=tool_dicts,
                    documents=request.documents,
                    chat_template_kwargs=request.chat_template_kwargs,
                    tool_parser=tool_parser,
                    truncate_prompt_tokens=request.truncate_prompt_tokens,
                    add_special_tokens=request.add_special_tokens,
                )
            except ValueError as e:
                # TODO: Error Generator Protocol
                return error_generator_protocol(self.create_error_response(str(e)))
            
            request_id = "chatcmpl-" \
                        f"{self._base_request_id(raw_request, request.request_id)}"

            request_metadata = RequestResponseMetadata(request_id=request_id)
            if raw_request:
                raw_request.state.request_metadata = request_metadata

            # Schedule the request and get the result generator.
            generators: List[AsyncGenerator[RequestOutput, None]] = []
            try:
                
                max_comp_tokens = request.max_completion_tokens \
                    if not request.max_completion_tokens is None \
                    else request.max_tokens
                
                
                gen_prompt_args = {
                    "max_tokens": max_comp_tokens if not max_comp_tokens is None else 0,
                    "chat_history": request.model_dump(include=["messages"])["messages"]
                }
                generate_prompt_dict = self.generate_prompt(gen_prompt_args, get_multi_modal_history=True)
                prompt = generate_prompt_dict["formatted_prompt"]
                
                if "chat_history_multi_modal" in generate_prompt_dict:
                    pre_encode_prompt = generate_prompt_dict["chat_history_multi_modal"]
                    prompt = encode_chat(
                        self.engine_t, self.tokenizer_t, pre_encode_prompt, 
                        chat_template=self.chat_template_t
                    )[0]
                
                sampling_params: Union[SamplingParams, BeamSearchParams]
                
                if isinstance(prompt, str):
                    pre_prompt_size = len(self.tokenizer_t.encode(prompt))
                else: # Multimodal case
                    pre_prompt_size = len(self.tokenizer_t.encode(prompt["prompt"]))
                
                
                default_max_tokens = self.max_model_len - pre_prompt_size
                # Build default sampling params
                default_sampling_params = (
                    self.model_config.get_diff_sampling_param())
                if request.use_beam_search:
                    sampling_params = request.to_beam_search_params(
                        default_max_tokens, default_sampling_params)
                else:
                    sampling_params = request.to_sampling_params(
                        default_max_tokens,
                        self.model_config.logits_processor_pattern,
                        default_sampling_params)
                
                
                if not sampling_params.ignore_eos:
                    stop_token_id = self.tokenizer_t.eos_token_id
                    stop_token = self.tokenizer_t.decode([stop_token_id])
                    sampling_params.stop_token_ids.append(stop_token_id)
                    
                    
                    if stop_token == "<|end_of_text|>":
                        alt_stop_token = "<|eot_id|>"
                        alt_stop_token_id = self.tokenizer_t.convert_tokens_to_ids(alt_stop_token)
                        sampling_params.stop.append(alt_stop_token)
                
                # Official method from VLLM OpenAIServingChat
                trace_headers = (None if raw_request is None else await
                                     self._get_trace_headers(raw_request.headers))
                generator = self.engine_client.generate(
                    engine_prompts[0],
                    sampling_params,
                    request_id,
                    lora_request=lora_request,
                    trace_headers=trace_headers,
                    prompt_adapter_request=prompt_adapter_request,
                    priority=request.priority,
                )
                
                generators.append(generator)
                
            except ValueError as e:
                # TODO: Use a vllm-specific Validation Error
                return error_generator_protocol(self.create_error_response(str(e)))

            assert len(generators) == 1
            result_generator, = generators

            # Streaming response
            if request.stream:
                return stream_prepend_generator(self.chat_completion_stream_generator(
                    request, result_generator, request_id, model_name,
                    conversation, tokenizer, request_metadata))

            try:
                result = await self.chat_completion_full_generator(
                    request, result_generator, request_id, model_name,
                    conversation, tokenizer, request_metadata)
                return single_generator_protocol(result)
            
            except ValueError as e:
                return error_generator_protocol(self.create_error_response(str(e)))
            
        except Exception as e:
            stack_trace = traceback.format_exc()
            error_response = {
                "error": str(e),
                "stack_trace": stack_trace
            }
            return error_generator_protocol(self.create_error_response(json.dumps(error_response)))
