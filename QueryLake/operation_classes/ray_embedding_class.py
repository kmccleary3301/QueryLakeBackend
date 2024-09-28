
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

# @serve.deployment(
#     ray_actor_options={"num_gpus": 0.04, "num_cpus": 2}, 
#     # max_replicas_per_node=1, 
#     max_ongoing_requests=128,
#     autoscaling_config={
#         "min_replicas": 0,
#         "max_replicas": 3,
#         "downscale_delay_s": 5,
#         "target_num_ongoing_requests_per_replica": 128,
#     }
# )
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
            batch_size=4,
            # max_length=8192,
            return_sparse=True,
            max_length=1024
        )
        
        inputs_tokenized = self.model.tokenizer(
            inputs,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=1024,
        )["input_ids"].tolist()
        
        pad_id = self.model.tokenizer.pad_token_id
        token_counts = [sum([1 for x in y if x != pad_id]) for y in inputs_tokenized]
        # sparse_vecs = sentence_embeddings['lexical_weights']
        
        # print("Sparse Vector:", sparse_vecs)
        
        embed_list = sentence_embeddings['dense_vecs'].tolist()
        print("Done handling batch of size", len(inputs))
        m_2 = time.time()
        print("Time taken for batch:", m_2 - m_1)
        
        return [{"embedding": embed_list[i], "token_count": token_counts[i]} for i in range(len(inputs))]
    
    async def run(self, request_dict : Union[dict, List[str]]) -> List[List[float]]:
        
        if isinstance(request_dict, dict):
            inputs = request_dict["text"]
        else:
            inputs = request_dict
        
        # Fire them all off at once, but wait for them all to finish before returning
        result = await gather(*[self.handle_batch(e) for e in inputs])
        return result
