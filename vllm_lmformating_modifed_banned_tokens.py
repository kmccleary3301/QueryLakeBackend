"""
Originally copied from PR:
https://github.com/noamgat/lm-format-enforcer/commit/1bd4f416cbd1f195982eeca5de9b0fcc6da0ff97#diff-6810482b4f46405e7cbc1268dbfc7c9acd35eb8055b955f7399108e46a7de330
at https://github.com/noamgat/lm-format-enforcer/blob/main/lmformatenforcer/integrations/vllm.py

With personal modifications to allow for banned tokens.
"""

try:
    import torch
    import vllm
    from transformers import PreTrainedTokenizerBase
except ImportError:
    raise ImportError('vllm is not installed. Please install it with "pip install vllm"')
from lmformatenforcer import CharacterLevelParser, TokenEnforcer, FormatEnforcerAnalyzer, TokenEnforcerTokenizerData
from lmformatenforcer.integrations.transformers import build_token_enforcer_tokenizer_data
from typing import List, Optional, Union
import math


class VLLMLogitsProcessor:
    def __init__(self, token_enforcer: TokenEnforcer, analyze, banned_tokens : List[int] = None):
        self.token_enforcer = token_enforcer
        self.analyzer = FormatEnforcerAnalyzer(token_enforcer) if analyze else None
        self.mask: Optional[torch.Tensor] = None
        self.banned_tokens = torch.tensor(banned_tokens) if not banned_tokens is None else None

    def __call__(self, input_ids: List[int], scores: torch.Tensor) -> torch.Tensor:
        token_sequence = input_ids
        if self.analyzer:
            self.analyzer.report_raw_logits(token_sequence, scores.tolist())
        allowed_tokens = self.token_enforcer.get_allowed_tokens(token_sequence)
        if self.mask is not None:
            self.mask.fill_(-math.inf)
        else:
            # We create it here because full_like() also copies the device and dtype
            self.mask = torch.full_like(scores, -math.inf)
        self.mask[allowed_tokens] = 0
        scores = scores + self.mask
        
        if not self.banned_tokens is None:
            scores[self.banned_tokens] = -math.inf
        
        return scores


def build_vllm_token_enforcer_tokenizer_data(llm: Union[vllm.LLM, PreTrainedTokenizerBase]) -> TokenEnforcerTokenizerData:
    tokenizer = llm.get_tokenizer() if isinstance(llm, vllm.LLM) else llm
    return build_token_enforcer_tokenizer_data(tokenizer)


def build_vllm_logits_processor(llm: Union[vllm.LLM, PreTrainedTokenizerBase, TokenEnforcerTokenizerData], 
                                character_level_parser: CharacterLevelParser, 
                                analyze: bool=False, banned_tokens : List[int] = None) -> VLLMLogitsProcessor:
    """Build the logits processor function that llama.cpp will use to filter the tokens generated by the model. The result
    can be passed in the logits_processor list that is sent to the call or generate() method of llama.cpp models."""
    if not isinstance(llm, TokenEnforcerTokenizerData):
        llm = build_vllm_token_enforcer_tokenizer_data(llm)
    token_enforcer = TokenEnforcer(llm, character_level_parser)
    return VLLMLogitsProcessor(token_enforcer, analyze, banned_tokens=banned_tokens)


__all__ = ['build_vllm_logits_processor', 'build_vllm_token_enforcer_tokenizer_data']
