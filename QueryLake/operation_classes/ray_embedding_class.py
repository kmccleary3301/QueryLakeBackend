import json
import os
import ray
from fastapi import Request
from fastapi.responses import Response
from typing import List, Union
from ray import serve
from transformers import AutoTokenizer, AutoModel
import torch
from asyncio import gather
from FlagEmbedding import BGEM3FlagModel
import time
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
from ..typing.config import LocalModel

def get_physical_gpu_info():
    """Get the actual physical GPU index and information."""
    if not PYNVML_AVAILABLE:
        return "pynvml not available"
    
    try:
        # Initialize NVML
        pynvml.nvmlInit()
        
        # Get current torch device
        if torch.cuda.is_available():
            current_device = torch.cuda.current_device()
            
            # If CUDA_VISIBLE_DEVICES is set, we need to map back to physical GPU
            cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
            if cuda_visible:
                visible_devices = cuda_visible.split(",")
                if current_device < len(visible_devices):
                    # Physical GPU index is the value in CUDA_VISIBLE_DEVICES
                    physical_gpu_id = int(visible_devices[current_device])
                else:
                    physical_gpu_id = current_device
            else:
                physical_gpu_id = current_device
            
            # Get GPU handle and information
            handle = pynvml.nvmlDeviceGetHandleByIndex(physical_gpu_id)
            gpu_name_raw = pynvml.nvmlDeviceGetName(handle)
            
            # Handle both string and bytes return values
            if isinstance(gpu_name_raw, bytes):
                gpu_name = gpu_name_raw.decode('utf-8')
            else:
                gpu_name = str(gpu_name_raw)
            
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            memory_used_mb = memory_info.used // (1024**2)
            memory_total_mb = memory_info.total // (1024**2)
            
            pynvml.nvmlShutdown()
            
            return {
                "physical_gpu_id": physical_gpu_id,
                "logical_gpu_id": current_device,
                "gpu_name": gpu_name,
                "memory_used_mb": memory_used_mb,
                "memory_total_mb": memory_total_mb,
                "memory_utilization_percent": round((memory_used_mb / memory_total_mb) * 100, 2)
            }
        else:
            return "CUDA not available"
            
    except Exception as e:
        if 'pynvml' in locals():
            try:
                pynvml.nvmlShutdown()
            except:
                pass
        return f"Error getting GPU info: {str(e)}"

class EmbeddingDeployment:
    def __init__(self, model_card : LocalModel):
        print("INITIALIZING EMBEDDING DEPLOYMENT")
        self.tokenizer = AutoTokenizer.from_pretrained(model_card.system_path)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = BGEM3FlagModel(model_card.system_path, use_fp16=True, device=self.device)
        
        # Get GPU assignment information for placement verification
        gpu_ids = ray.get_gpu_ids()
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "None")
        node_id = ray.get_runtime_context().get_node_id()
        physical_gpu_info = get_physical_gpu_info()
        
        print(f"🤖 Embedding model [{model_card.id}] GPU assignment:")
        print(f"   Node ID: {node_id}")
        print(f"   Ray GPU IDs: {gpu_ids}")
        print(f"   CUDA_VISIBLE_DEVICES: {cuda_visible}")
        
        if isinstance(physical_gpu_info, dict):
            print(f"   Physical GPU ID: {physical_gpu_info['physical_gpu_id']}")
            print(f"   Logical GPU ID: {physical_gpu_info['logical_gpu_id']}")
            print(f"   GPU Name: {physical_gpu_info['gpu_name']}")
            print(f"   Memory: {physical_gpu_info['memory_used_mb']}/{physical_gpu_info['memory_total_mb']} MB ({physical_gpu_info['memory_utilization_percent']}%)")
        else:
            print(f"   Physical GPU Info: {physical_gpu_info}")
        
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
