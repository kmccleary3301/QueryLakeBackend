import time

import re, json

from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache, ExLlamaV2Tokenizer
from exllamav2.generator import ExLlamaV2DynamicGenerator, ExLlamaV2DynamicJob
from QueryLake.misc_functions.vllm_lmformating_modifed_banned_tokens import build_vllm_token_enforcer_tokenizer_data
from ray import serve

from QueryLake.misc_functions.grammar_sampling_functions import get_token_id, get_logits_processor_from_grammar_options

from QueryLake.typing.config import Padding, Model
from QueryLake.misc_functions.prompt_construction import construct_chat_history
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from typing import List, Union, AsyncIterator
from QueryLake.typing.function_calling import FunctionCallDefinition
from QueryLake.misc_functions.server_class_functions import construct_functions_available_prompt
import os


async def generator_wrapper(exl2_generator : ExLlamaV2DynamicJob) -> AsyncIterator:
     async for result in exl2_generator:
        print("RESULT:", result)
         
        # We'll only collect text here, but the result could contain other updates
        yield result


# TODO: This doesn't work. The async for loop stops after the first yield for some reason.
# Possibly to do with exllamav2 using asyncio and asyncio.queue for managing batches.
# The solution may be to use non-async dynamic generator from exllamav2.
# See here: https://github.com/turboderp/exllamav2/blob/master/examples/dynamic_gen.py
@serve.deployment(ray_actor_options={"num_gpus": 0.4}, max_replicas_per_node=1)
class ExllamaV2DeploymentClass:
    def __init__(self,
                 model_config : Model,
                 **kwargs):
        """
        Construct an ExLlamaV2 deployment.
        
        It's simply better than vLLM in every way. ¯\_(ツ)_/¯
        """
        self.model_config = model_config
        self.padding : Padding = model_config.padding
        self.default_model_args = self.model_config.default_parameters
        self.context_size = self.model_config.max_model_len
        print("INITIALIZING EXLLAMAV2 DEPLOYMENT")
        # os.environ["FORCE_CUDA"] = "1"
        self.config = ExLlamaV2Config(self.model_config.system_path)
        self.config.arch_compat_overrides()
        self.config.max_seq_len = self.context_size
        
        self.model = ExLlamaV2(self.config)
        self.cache = ExLlamaV2Cache(self.model, max_seq_len=self.context_size, lazy=True)
        self.model.load_autosplit(self.cache, progress = True)
        self.tokenizer = ExLlamaV2Tokenizer(self.config)
        
        self.generator = ExLlamaV2DynamicGenerator(
            model = self.model,
            cache = self.cache,
            tokenizer = self.tokenizer,
        )
        
        self.generator.warmup()
        
        # self.engine = AsyncLLMEngine.from_engine_args(args)
        
        # self.engine = AsyncLLMEngine.from_engine_args(args)
        # self.tokenizer = self.engine.engine.get_tokenizer()
        # tokenizer_tmp = self.engine.engine.tokenizer
        # self.special_token_ids = tokenizer_tmp.all_special_ids
        # self.space_tokens = [get_token_id(tokenizer_tmp, e) for e in ["\n", "\t", "\r", " \r"]]
        # self.tokenizer_data = build_vllm_token_enforcer_tokenizer_data(self.engine.engine)
        print("DONE INITIALIZING EXLLAMAV2 DEPLOYMENT")
    
    def count_tokens(self, input_string : str):
        return len(self.tokenizer.encode(input_string, add_bos = False))
    
    def generate_prompt(self, 
                        request_dict : dict,
                        sources : List[dict] = [],
                        functions_available: List[Union[FunctionCallDefinition, dict]] = None):
        if "prompt" in request_dict:
            prompt = request_dict.pop("prompt")
        else:
            chat_history = request_dict.pop("chat_history")
            if len(sources) > 0:
                chat_history[-1]["content"] = ("SYSTEM MESSAGE - PROVIDED SOURCES\n<SOURCES>\n" +
                    '\n\n'.join(['[%d] Source %d\n\n%s' % (i+1, i+1, e['text']) for i, e in enumerate(sources)]) +
                    f"\n</SOURCES>\n END SYSTEM MESSAGE\n{chat_history[-1]['content']}")
            if not functions_available is None:
                chat_history[-1]["content"] = (
                    f"SYSTEM MESSAGE - PROVIDED SOURCES\n{construct_functions_available_prompt(functions_available)}" + \
                    f"\nEND SYSTEM MESSAGE\n\n{chat_history[-1]['content']}"
                )
            
            
            prompt = construct_chat_history(self.model_config, self.count_tokens, chat_history, request_dict["max_new_tokens"] + 2)
        return prompt
    
    def create_generator(self, 
                         request_dict : dict, 
                         sources : List[dict] = [],
                         functions_available: List[Union[FunctionCallDefinition, dict]] = None):
        
        prompt = self.generate_prompt(request_dict, sources, functions_available)
        
        assert self.count_tokens(prompt) <= self.context_size, f"Prompt is too long."
        
        grammar_options = request_dict.pop("grammar", None)
        
        # print("Got grammar options:", grammar_options)
        
        # logits_processor_local = get_logits_processor_from_grammar_options(
        #     grammar_options,
        #     self.tokenizer_data, 
        #     # space_tokens=self.space_tokens, 
        #     # special_ids=self.special_token_ids,
        # ) if not grammar_options is None else None
        
        # print("Got logits processor", logits_processor_local)
        
        keys = [
            "token_repetition_penalty",
            "token_repetition_range",
            "token_repetition_decay",
            "token_frequency_penalty",
            "token_presence_penalty",
            "temperature",
            "smoothing_factor",
            "min_temp",
            "max_temp",
            "temp_exponent",
            "top_k",
            "top_p",
            "top_a",
            "min_p",
            "tfs",
            "typical",
            "skew",
            "max_new_tokens"
        ]
        filtered_dict = {k: request_dict[k] for k in keys if k in request_dict}
        # sampling_params = SamplingParams(**filtered_dict)
        
        # if not logits_processor_local is None:
        #     sampling_params.logits_processors = [logits_processor_local]
        
        # results_generator = self.engine.generate(prompt, sampling_params, request_id)
        
        
        
        # results_generator = ExLlamaV2DynamicJobAsync(
        #     generator,
        #     input_ids = self.tokenizer.encode(prompt, add_bos = False),
        #     **filtered_dict
        # )
        
        
        job = ExLlamaV2DynamicJob(
            input_ids = self.tokenizer.encode(prompt, add_bos = False),
            **filtered_dict
        )
        
        
        # async for r in results_generator:
        #     print(r)
        #     yield r
        self.generator.enqueue(job)
        
        # return results_generator
        return job
    
    async def get_result_loop(self, 
                              request_dict : dict, 
                              sources : List[dict] = [],
                              functions_available: List[Union[FunctionCallDefinition, dict]] = None):
        generator = self.create_generator(request_dict, sources, functions_available)
        print("Inside get_result_loop recieved Generator")
        results = []
        async for result in generator_wrapper(generator):
            yield result