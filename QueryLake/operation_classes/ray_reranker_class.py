import json
from fastapi import Request
import time
from fastapi.responses import Response
from typing import Tuple, Union, List
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from math import exp
from ray import serve
import itertools
from asyncio import gather
from ..typing.config import LocalModel


def modified_sigmoid(x : Union[torch.Tensor, float]):
    if type(x) == float or type(x) == int:
        return 100 / (1 + exp(-8 * ((x / 100) - 0.5)))
    if type(x) == list:
        x = torch.tensor(x)
    return 100 * torch.sigmoid(-8 * ((x / 100) - 0.5))

def S(x):
    return torch.sigmoid(4*(x - 0.5)) if isinstance(x, torch.Tensor) else 1 / (1 + exp(-4 * (x - 0.5)))

def H(x):
    return 100 * S(x / 100) if isinstance(x, torch.Tensor) else 100 * S(x / 100)

def F(x):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor([x], dtype=torch.float32)

    h_zero = H(0)
    delta_h = H(x) - h_zero
    result = delta_h * (100 / (100 - h_zero))

    # If input was a single value, convert back to scalar output
    result = torch.minimum(result, torch.full(result.shape, 100.0))
    
    if len(result.shape) == 0:
        result = float(result)

    return result


class RerankerDeployment:
    def __init__(self, model_card : LocalModel):
        print("INITIALIZING RERANKER DEPLOYMENT")
        
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_card.system_path)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # self.device = "cpu"
        self.model = AutoModelForSequenceClassification.from_pretrained(model_card.system_path).to(self.device)
        print("DONE INITIALIZING RERANKER DEPLOYMENT")
        # self.model = FlagReranker(model_key, use_fp16=True)
    
    @serve.batch(max_batch_size=32, batch_wait_timeout_s=0.05)
    async def handle_batch(self, inputs: List[Tuple[str, str]], normalize = List[bool]) -> List[float]:
        
        with torch.no_grad():
            start_time = time.time()
            tokenized_inputs = self.tokenizer(inputs, padding=True, truncation=True, return_tensors='pt', max_length=512).to(self.device)
            
            pad_id = self.tokenizer.pad_token_id
            
            token_ids = tokenized_inputs["input_ids"].tolist()
            token_counts = [sum([1 for x in y if x != pad_id]) for y in token_ids]
            
        
            
            mid_time = time.time()
            scores = self.model(**tokenized_inputs, return_dict=True).logits.view(-1, ).float().to("cpu") # This takes 8 seconds!!! WTF?!?!
            time_taken_model, time_taken_tokens = time.time() - mid_time, mid_time - start_time
            print("Time taken for rerank inference: %.2f s + %.2f s = %.2f s" % (time_taken_model, time_taken_tokens, time_taken_model + time_taken_tokens))
        
        scores = torch.exp(scores.clone().detach())
        scores_normed = F(scores)
        
        scores = list(map(lambda x: float(scores_normed[x]) if normalize[x] else float(scores[x]), list(range(len(scores)))))
        
        return [{"score": scores[i], "token_count": token_counts[i]} for i in range(len(scores))]

    async def run(self, input: Tuple[str, str], normalize : bool = True) -> List[List[float]]:
        # Fire them all off at once and get the coroutines, but await them as a list.
        return await self.handle_batch(input, normalize)