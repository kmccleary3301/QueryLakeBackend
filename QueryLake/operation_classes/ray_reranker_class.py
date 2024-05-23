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


def modified_sigmoid(x : Union[torch.Tensor, float]):
    if type(x) == float or type(x) == int:
        return 100 / (1 + exp(-8 * ((x / 100) - 0.5)))
    if type(x) == list:
        x = torch.tensor(x)
    return 100 * torch.sigmoid(-8 * ((x / 100) - 0.5))

def S(x):
    # return 1/(1 + torch.exp(-4 * (x - 0.5))) if isinstance(x, torch.Tensor) else 1 / (1 + math.exp(-4 * (x - 0.5)))
    return torch.sigmoid(4*(x - 0.5)) if isinstance(x, torch.Tensor) else 1 / (1 + exp(-4 * (x - 0.5)))

def H(x):
    return 100 * S(x / 100) if isinstance(x, torch.Tensor) else 100 * S(x / 100)

def F(x):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor([x], dtype=torch.float32)

    h_zero = H(0)
    # h_one = H(100)
    delta_h = H(x) - h_zero
    result = delta_h * (100 / (100 - h_zero))

    # If input was a single value, convert back to scalar output
    result = torch.minimum(result, torch.full(result.shape, 100.0))
    
    if len(result.shape) == 0:
        result = float(result)

    return result

@serve.deployment(ray_actor_options={"num_gpus": 0, "num_cpus": 2}, max_replicas_per_node=1)
class RerankerDeployment:
    def __init__(self, model_key: str):
        print("INITIALIZING RERANKER DEPLOYMENT")
        self.tokenizer = AutoTokenizer.from_pretrained(model_key)
        # self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = "cpu"
        self.model = AutoModelForSequenceClassification.from_pretrained(model_key).to(self.device)
        print("DONE INITIALIZING RERANKER DEPLOYMENT")
        # self.model = FlagReranker(model_key, use_fp16=True)
    
    @serve.batch(max_batch_size=32, batch_wait_timeout_s=0.1)
    async def handle_batch(self, inputs: List[Tuple[str, str]]) -> List[float]:
        
        with torch.no_grad():
            start_time = time.time()
            tokenized_inputs = self.tokenizer(inputs, padding=True, truncation=True, return_tensors='pt', max_length=512).to(self.device)
            # mid_time = time.time()
            scores = self.model(**tokenized_inputs, return_dict=True).logits.view(-1, ).float().to("cpu") # This takes 8 seconds!!! WTF?!?!
            time_taken = time.time() - start_time
            print("Time taken for rerank inference:", time_taken)
        
        scores = torch.exp(torch.tensor(scores.clone().detach()))
        scores = F(scores)
        
        return scores.tolist()

    async def run(self, request : Union[dict, List[Tuple[str, str]]]) -> List[List[float]]:
        
        if isinstance(request, dict):
            inputs = request["text"]
        else:
            inputs = request
        
        # Fire them all off at once and get the coroutines, but await them as a list.
        return await gather(*[self.handle_batch(e) for e in inputs])