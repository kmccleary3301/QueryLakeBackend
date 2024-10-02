import argparse
import json
import os
import subprocess
import huggingface_hub.errors
from pynvml import nvmlInit, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
from colorama import init, Fore, Style
import time
import huggingface_hub

# Initialize colorama
init(autoreset=True)

def setup_config(config_path):
    default_config = {
        "default_toolchain": "test_chat_session_normal_streaming",
        "default_models": {
            "llm": "llama-3.1-8b-instruct",
            "embedding": "bge-m3",
            "rerank": "bge-reranker-v2-m3"
        },
        "enabled_model_classes": {
            "llm": True,
            "embedding": True,
            "rerank": True
        },
        "models": [],
        "external_model_providers": {
            "openai": []
        },
        "providers": [
            "OpenAI",
            "Anthropic",
            "Serper.dev"
        ],
        "other_local_models": {
            "rerank_models": [],
            "embedding_models": []
        }
    }

    with open(config_path, 'w') as f:
        json.dump(default_config, f, indent=4)
    print(f"{Fore.GREEN}Config file created at {config_path}")

def download_model(model_id, save_path):
    if not os.path.exists(save_path):
        print(f"{Fore.YELLOW}Downloading model {model_id} using huggingface-cli...")
        subprocess.run(["huggingface-cli", "download", model_id, save_path], check=True)
        print(f"{Fore.GREEN}Model saved to {save_path}")
    else:
        print(f"{Fore.CYAN}Model already exists at {save_path}")

def check_gpus():
    nvmlInit()
    device_count = nvmlDeviceGetCount()
    gpus = []
    for i in range(device_count):
        handle = nvmlDeviceGetHandleByIndex(i)
        memory_info = nvmlDeviceGetMemoryInfo(handle)
        gpus.append({
            "index": i,
            "total_vram": memory_info.total // (1024 ** 2)  # Convert bytes to MB
        })
    return gpus

def main():
    parser = argparse.ArgumentParser(description="Setup config file and download models.")
    parser.add_argument('--config', default=None, type=str, help="Path to the config file")
    parser.add_argument('--model_id', default=None, type=str, help="ID of the model to download from HuggingFace")
    parser.add_argument('--save_path', default=None, type=str, help="Path to save the downloaded model")
    args = parser.parse_args()
    
    gpus = check_gpus()
    print(f"{Fore.MAGENTA}GPUs and their VRAM totals:")
    for gpu in gpus:
        print(f"{Fore.MAGENTA}GPU {gpu['index']}: {gpu['total_vram']} MB")
    
    if all([v is None for v in vars(args).values()]):
        print(f"{Fore.BLUE}Running QueryLake setup")

    
        enable_llms = input(f"Would you like to enable LLMs? {Fore.YELLOW}[Y/n] -> ").strip().lower()[:1]
        # if enable_llms == 'y' or enable_llms == '':
        enable_llms = (enable_llms == 'y')
        print(f"{Fore.GREEN if enable_llms else Fore.RED}LLMs will{'' if enable_llms else ' not'} be enabled.")
        
        if enable_llms:
            print(f"\nLets begin adding models to the config file.")
            print(f"You must enter models via their huggingface id (i.e. {Fore.BLUE}meta-llama/Llama-2-70b-chat-hf{Fore.RESET})\n")
        while enable_llms:
            
            
            new_model = input(f"\nPlease enter a model address -> {Fore.BLUE}").strip()
            # print("Got new model:", new_model)
            
            
            if new_model == '':
                print(f"{Fore.RED}Done adding models.")
                break
            
            
            # huggingface_hub.hf_hub_download(new_model)
            # huggingface_hub.cached_download(f"https://huggingface.co/{new_model}")
            
            while True:
                try:
                    huggingface_hub.snapshot_download(new_model, cache_dir="/home/kmccleary/projects/QueryLake/ai_models/test_tmp")
                    print(f"{Fore.GREEN}Added model {Fore.BLUE}{new_model.split('/')[-1]} {Fore.GREEN}to config file.")
                    break
                except huggingface_hub.errors.RepositoryNotFoundError:
                    print(f"{Fore.RED}Error: Model not found. Please try again.")
                except Exception as e:
                    print(f"{Fore.RED}Error downloading model. Please try again.")
                    print(f"{Fore.RED}Error message: {e}")
                new_model = input(f"Please enter a model address -> {Fore.BLUE}").strip()
                if new_model == '':
                    print(f"{Fore.RED}Done adding models.")
                    break
            # huggingface_hub.snapshot_download(new_model, cache_dir="/home/kmccleary/projects/QueryLake/ai_models/test_tmp")
            
            # print("\n", end="")
            
            context_size_set = input(f"Would you like to set a custom context window? You would only do this to preserve memory. {Fore.YELLOW}[Y/n] -> ").strip().lower()[:1]
            context_size_set = (context_size_set == 'y')
            if context_size_set:
                while True:
                    context_size = input(f"Enter the context window size -> {Fore.BLUE}").strip()
                    try:
                        context_size = int(context_size)
                        print(f"{Fore.RESET}Setting context window size to {Fore.BLUE}{context_size}")
                        break
                    except ValueError:
                        print(f"{Fore.RED}Please enter a valid integer.")
            
        # else:
        #     print(f"{Fore.RED}LLMs will not be enabled.")
    
    # setup_config(args.config)

    # if args.model_id and args.save_path:
    #     download_model(args.model_id, args.save_path)

    

if __name__ == "__main__":
    main()