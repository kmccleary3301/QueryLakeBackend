
import json
from fastapi import Request
from fastapi.responses import Response
from typing import  List
from ray import serve
from transformers import AutoTokenizer, AutoModel
import torch

@serve.deployment
class EmbeddingDeployment:
    def __init__(self, model_key: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_key)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = AutoModel.from_pretrained(model_key).to(self.device)

    @serve.batch()
    async def handle_batch(self, inputs: List[List[str]]) -> List[List[List[float]]]:
        print("Running Embedding 3.1")
        # print("Our input array has length:", len(inputs))

        # print("Input array:", inputs)
        
        batch_size = len(inputs)
        # We need to flatten the inputs into a single list of string, then recombine the outputs according to the original structure.
        flat_inputs = [(item, i_1, i_2) for i_1, sublist in enumerate(inputs) for i_2, item in enumerate(sublist)]
        
        flat_inputs_indices = [item[1:] for item in flat_inputs]
        flat_inputs_text = [item[0] for item in flat_inputs]
        encoded_input = self.tokenizer(flat_inputs_text, padding=True, truncation=True, return_tensors='pt').to(self.device)

        with torch.no_grad():
            model_output = self.model(**encoded_input)
            sentence_embeddings = model_output[0][:, 0]
        
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        
        outputs = [[] for _ in range(batch_size)]
        
        for i, output in enumerate(sentence_embeddings.tolist()):
            request_index, input_index = flat_inputs_indices[i]
            outputs[request_index].append(output)
            # outputs[item[1]][item[2]] = sentence_embeddings[i].tolist()
        
        return outputs

    async def __call__(self, request : Request) -> Response:
        print("Running Embedding 2.1")
        request_dict = await request.json()
        print("Running Embedding 2.2")
        # return await self.handle_batch(request_dict["text"])
        return_tmp = await self.handle_batch(request_dict["text"])
        print("Running Embedding 2.3")
        return Response(content=json.dumps({"output": return_tmp}))
    
    async def run(self, request_dict : dict) -> List[List[float]]:
        return await self.handle_batch(request_dict["text"])
