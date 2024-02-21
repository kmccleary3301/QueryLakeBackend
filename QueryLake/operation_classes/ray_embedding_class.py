
import json
from fastapi import Request
from fastapi.responses import Response
from typing import List, Union
from ray import serve
from transformers import AutoTokenizer, AutoModel
import torch
from asyncio import gather

@serve.deployment(ray_actor_options={"num_gpus": 0, "num_cpus": 2}, max_replicas_per_node=1)
class EmbeddingDeployment:
    def __init__(self, model_key: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_key)
        # self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = "cpu"
        self.model = AutoModel.from_pretrained(model_key).to(self.device)

    @serve.batch(max_batch_size=16, batch_wait_timeout_s=0.5)
    async def handle_batch(self, inputs: List[str]) -> List[List[float]]:
        print("Running handle_batch with input length:", len(inputs))
        
        encoded_input = self.tokenizer(inputs, padding=True, truncation=True, return_tensors='pt').to(self.device)

        with torch.no_grad():
            model_output = self.model(**encoded_input)
            sentence_embeddings = model_output[0][:, 0]
        
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        
        return sentence_embeddings.tolist()
    
    async def run(self, request_dict : Union[dict, List[str]]) -> List[List[float]]:
        
        if isinstance(request_dict, dict):
            inputs = request_dict["text"]
        else:
            inputs = request_dict
        
        print("Calling embedding within handle with input length:", len(inputs))
        # return await [self.handle_batch(e) for e in inputs] 
        
        # Fire them all off at once, but wait for them all to finish before returning
        return await gather(*[self.handle_batch(e) for e in inputs])
