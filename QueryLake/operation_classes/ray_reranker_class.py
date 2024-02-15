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

@serve.deployment(ray_actor_options={"num_gpus": 0.4}, max_replicas_per_node=1)
class RerankerDeployment:
    def __init__(self, model_key: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_key)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForSequenceClassification.from_pretrained(model_key).to(self.device)
        
        # self.model = FlagReranker(model_key, use_fp16=True)
    
    @serve.batch()
    async def handle_batch(self, inputs: List[List[Tuple[str, str]]]) -> List[List[float]]:
        batch_size = len(inputs)
        flat_inputs = [(item, i_1, i_2) for i_1, sublist in enumerate(inputs) for i_2, item in enumerate(sublist)]
        
        flat_inputs_indices = [item[1:] for item in flat_inputs]
        flat_inputs_text = [item[0] for item in flat_inputs]
        
        with torch.no_grad():
            start_time = time.time()
            tokenized_inputs = self.tokenizer(flat_inputs_text, padding=True, truncation=True, return_tensors='pt', max_length=512).to(self.device)
            mid_time = time.time()
            scores = self.model(**tokenized_inputs, return_dict=True).logits.view(-1, ).float().to("cpu") # This takes 8 seconds!!! WTF?!?!
            
        scores = torch.exp(torch.tensor(scores.clone()))
        scores = F(scores)
        
        outputs = [[] for _ in range(batch_size)]
        
        for i, output in enumerate(scores.tolist()):
            request_index, _ = flat_inputs_indices[i]
            outputs[request_index].append(output)
        
        return outputs

    async def __call__(self, request : Request) -> Response:
        request_dict = await request.json()
        # print("Got request:", request_dict)
        return_tmp = await self.handle_batch(request_dict["text"])
        return Response(content=json.dumps({"output": return_tmp}))

    async def run(self, request_dict : dict) -> List[List[float]]:
        start_time = time.time()
        values = await self.handle_batch(request_dict["text"])
        end_time = time.time()
        print("RERANKER INTERNAL TIME %7.3fs" % (end_time - start_time))
        return values