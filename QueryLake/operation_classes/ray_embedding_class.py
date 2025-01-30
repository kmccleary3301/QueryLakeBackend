
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
from ..typing.config import LocalModel

class EmbeddingDeployment:
    def __init__(self, model_card : LocalModel):
        print("INITIALIZING EMBEDDING DEPLOYMENT")
        self.tokenizer = AutoTokenizer.from_pretrained(model_card.system_path)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = BGEM3FlagModel(model_card.system_path, use_fp16=True, device=self.device) 
        print("DONE INITIALIZING EMBEDDING DEPLOYMENT")

    @serve.batch(max_batch_size=128, batch_wait_timeout_s=0.05)
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
