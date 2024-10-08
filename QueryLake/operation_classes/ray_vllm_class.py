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
        print("DONE INITIALIZING VLLM DEPLOYMENT")
    
    def count_tokens(self, input_string : str):
        return len(self.tokenizer(input_string)["input_ids"])
    
    def generate_prompt(self, 
                        request_dict : dict,
                        sources : List[dict] = [],
                        functions_available: List[Union[FunctionCallDefinition, dict]] = None):
        if "prompt" in request_dict:
            prompt = request_dict["prompt"]
            
            return {"formatted_prompt": prompt, "tokens": self.count_tokens(prompt)}
        elif "chat_history" in request_dict:
            chat_history = request_dict["chat_history"]
            padding = 2 if (functions_available is None and len(sources) == 0) else (100 + 300 * len(sources) + 300 * len(functions_available))
            if len(sources) > 0:
                chat_history[-1]["content"] = ("SYSTEM MESSAGE - PROVIDED SOURCES\n<SOURCES>\n" +
                    '\n\n'.join(['[%d] Source %d\n\n%s' % (i+1, i+1, e['text']) for i, e in enumerate(sources)]) +
                    f"\n</SOURCES>\nEND SYSTEM MESSAGE\n{chat_history[-1]['content']}")
            if not functions_available is None:
                chat_history[-1]["content"] = (
                    f"SYSTEM MESSAGE - AVAILABLE FUNCTIONS\n<FUNCTIONS>{construct_functions_available_prompt(functions_available)}" + \
                    f"\n</FUNCTIONS>\nEND SYSTEM MESSAGE\n\n{chat_history[-1]['content']}"
                )    
            prompt, chopped_chat_history = construct_chat_history(self.model_config, self.count_tokens, chat_history, request_dict["max_tokens"] - padding , return_chat_history=True)
            
            return {"formatted_prompt": prompt, "chat_history": chopped_chat_history, "tokens": self.count_tokens(prompt)}
        else:
            raise ValueError("Request dictionary must contain either 'prompt' or 'chat_history' key. Got: " + str(request_dict.keys()))
    
    def get_result_loop(self, 
                        request_dict : dict, 
                        sources : List[dict] = [],
                        functions_available: List[Union[FunctionCallDefinition, dict]] = None):
        
        prompt = self.generate_prompt(request_dict, sources, functions_available)["formatted_prompt"]
        
        assert self.count_tokens(prompt) <= self.context_size, f"Prompt is too long."
        
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
        
        # print(prompt)
        
        results_generator = self.engine.generate(prompt, sampling_params, request_id)
        
        return results_generator