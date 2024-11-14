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
from typing import List, Union
from QueryLake.typing.function_calling import FunctionCallDefinition
from QueryLake.misc_functions.server_class_functions import construct_functions_available_prompt
from huggingface_hub import snapshot_download
import os

from typing import List, Union, Optional, Dict, Any
from vllm.entrypoints.llm import ChatCompletionMessageParam, cast, is_list_of, TokensPrompt, TextPrompt, parse_chat_messages, MistralTokenizer
from vllm.entrypoints.llm import apply_mistral_chat_template, apply_hf_chat_template
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.config import ModelConfig
from copy import deepcopy
from QueryLake.misc_functions.toolchain_state_management import safe_serialize
from pydantic import BaseModel

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
            msgs, model_config, tokenizer)

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
    
    print("FORMATTER GOT CHAT HISTORY:", json.dumps(chat_history, indent=4))
    
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
    
    print("RETURNING CHAT HISTORY:", json.dumps(chat_history, indent=4))
    
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
class VLLMDeploymentClass:
    def __init__(self,
                       model_config : Model,
                       **kwargs):
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
        self.model_config = model_config
        self.padding : Padding = model_config.padding
        self.default_model_args = self.model_config.default_parameters
        
        engine_args_make = {
            **kwargs,
            "gpu_memory_utilization": 0.4, # This should match the ray GPU allocation
            "enforce_eager": True,
            "disable_log_requests": True,  # Had to mute this thing because it was spamming the logs.
            **(self.model_config.engine_args if not self.model_config.engine_args is None else {})
        }
        
        args = AsyncEngineArgs( 
            **engine_args_make,
            model=self.model_config.system_path,
        )
        self.context_size = args.max_model_len
        
        print("INITIALIZING VLLM DEPLOYMENT")
        # self.engine = AsyncLLMEngine.from_engine_args(args)
        
        self.engine = AsyncLLMEngine.from_engine_args(args)
        self.tokenizer = self.engine.engine.get_tokenizer()
        # tokenizer_tmp = self.engine.engine.tokenizer
        # self.special_token_ids = tokenizer_tmp.all_special_ids
        # self.space_tokens = [get_token_id(tokenizer_tmp, e) for e in ["\n", "\t", "\r", " \r"]]
        self.tokenizer_data = build_vllm_token_enforcer_tokenizer_data(self.engine.engine)
        
        self.vllm_model_config = self.engine.engine.get_model_config()
        print("DONE INITIALIZING VLLM DEPLOYMENT")
    
    def count_tokens(self, input_string : str):
        return len(self.tokenizer(input_string)["input_ids"])
    
    def generate_prompt(self, 
                        request_dict : dict,
                        sources : List[dict] = [],
                        functions_available: List[Union[FunctionCallDefinition, dict]] = None,
                        get_multi_modal_history: bool = False):
        
        print("Generate prompt got sources:", sources)
        
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
                prompt, chopped_chat_history = construct_chat_history(self.model_config, self.count_tokens, stripped_chat_history, request_dict["max_tokens"] - padding , return_chat_history=True)
                return {
                    "formatted_prompt": prompt,
                    "chat_history": chopped_chat_history, 
                    **({"chat_history_multi_modal": chat_history} if get_multi_modal_history else {}),
                    "tokens": 100
                }
            
            
            
            # Try normal chat formatting.
            try:
                prompt = self.tokenizer.apply_chat_template(stripped_chat_history, add_generation_prompt=True, continue_final_message=False)
                return {"formatted_prompt": prompt, "chat_history": chopped_chat_history, "tokens": self.count_tokens(prompt)}
            except:
                pass
            
            # Normal chat history without multi-modal
            prompt, chopped_chat_history = construct_chat_history(self.model_config, self.count_tokens, stripped_chat_history, request_dict["max_tokens"] - padding , return_chat_history=True)
            return {"formatted_prompt": prompt, "chat_history": chopped_chat_history, "tokens": self.count_tokens(prompt)}
            
            # print("Chat history:", json.dumps(chat_history, indent=4))
            
            
            
        else:
            raise ValueError("Request dictionary must contain either 'prompt' or 'chat_history' key. Got: " + str(request_dict.keys()))
    
    def get_result_loop(self, 
                        request_dict : dict, 
                        sources : List[dict] = [],
                        functions_available: List[Union[FunctionCallDefinition, dict]] = None):
        
        generate_prompt_dict = self.generate_prompt(request_dict, sources, functions_available, get_multi_modal_history=True)
        prompt = generate_prompt_dict["formatted_prompt"]
        
        
        # assert self.count_tokens(prompt) <= self.context_size, f"Prompt is too long."
        
        request_id = random_uuid()
        
        grammar_options = request_dict.pop("grammar", None)
        
        # print("Got grammar options:", grammar_options)
        
        logits_processor_local = get_logits_processor_from_grammar_options(
            grammar_options,
            self.tokenizer_data, 
            # space_tokens=self.space_tokens, 
            # special_ids=self.special_token_ids,
        ) if not grammar_options is None else None
        
        # print("Got logits processor", logits_processor_local)
        
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
        
        if "chat_history_multi_modal" in generate_prompt_dict:
            pre_encode_prompt = generate_prompt_dict["chat_history_multi_modal"]
            prompt = encode_chat(self.engine, self.tokenizer, pre_encode_prompt)[0]
        
        results_generator = self.engine.generate(prompt, sampling_params, request_id)
        
        return results_generator