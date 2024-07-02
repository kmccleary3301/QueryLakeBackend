
import json
from fastapi import Request
from fastapi.responses import Response
from typing import List, Union
from ray import serve
from transformers import AutoTokenizer, AutoModel
import torch
from asyncio import gather
from FlagEmbedding import BGEM3FlagModel
import time

@serve.deployment(ray_actor_options={"num_gpus": 0.1, "num_cpus": 2}, max_replicas_per_node=1, max_ongoing_requests=128)
class EmbeddingDeployment:
    def __init__(self, model_key: str):
        print("INITIALIZING EMBEDDING DEPLOYMENT")
        self.tokenizer = AutoTokenizer.from_pretrained(model_key)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # self.device = "cpu"
        # self.model = AutoModel.from_pretrained(model_key).to(self.device)
        self.model = BGEM3FlagModel(model_key, use_fp16=True, device=self.device) 
        print("DONE INITIALIZING EMBEDDING DEPLOYMENT")

    @serve.batch(max_batch_size=128, batch_wait_timeout_s=1)
    async def handle_batch(self, inputs: List[str]) -> List[List[float]]:
        
        m_1 = time.time()
        print("Handling batch of size", len(inputs))
        sentence_embeddings = self.model.encode(
            inputs,
            # max_length=8192,
            max_length=4096,
        )['dense_vecs']
        embed_list = sentence_embeddings.tolist()
        print("Done handling batch of size", len(inputs))
        m_2 = time.time()
        print("Time taken for batch:", m_2 - m_1)
        
        return embed_list
    
    async def run(self, request_dict : Union[dict, List[str]]) -> List[List[float]]:
        
        if isinstance(request_dict, dict):
            inputs = request_dict["text"]
        else:
            inputs = request_dict
        
        # Fire them all off at once, but wait for them all to finish before returning
        result = await gather(*[self.handle_batch(e) for e in inputs])
        return result
