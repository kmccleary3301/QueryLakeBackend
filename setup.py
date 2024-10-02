import argparse
import json
import os
import subprocess
import huggingface_hub.errors
from pynvml import nvmlInit, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
from colorama import init, Fore, Style
import time
import huggingface_hub
from copy import deepcopy
from QueryLake.typing.config import Config, Model, LocalModel
from typing import List, Union

# Initialize colorama
init(autoreset=True)

# def setup_config(config_path):
DEFAULT_CONFIG = Config(**{
    "default_toolchain": "test_chat_session_normal_streaming",
    "default_models": {
        "llm": "llama-3.1-8b-instruct",
        "embedding": "bge-m3",
        "rerank": "bge-reranker-v2-m3"
    },
    "enabled_model_classes": {
        "llm": False,
        "embedding": False,
        "rerank": False
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
})

    # with open(config_path, 'w') as f:
    #     json.dump(default_config, f, indent=4)
    # print(f"{Fore.GREEN}Config file created at {config_path}")

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

def custom_setup(config_current : Config, default_models : Config, gpus : list):
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
                    
                    
def premade_models_setup(config_current : Config, default_models : Config, gpus : list):
    enable_llms = input(f"Would you like to enable LLMs? {Fore.YELLOW}[Y/n] -> ").strip().lower()[:1]
    # if enable_llms == 'y' or enable_llms == '':
    enable_llms = (enable_llms == 'y')
    print(f"{Fore.GREEN if enable_llms else Fore.RED}LLMs will{'' if enable_llms else ' not'} be enabled.")
    
    models_wanted = []
    if enable_llms:
        print(f"\nPlease select models from the following list:")
        max_numbers = len(default_models.models)
        max_digits = len(str(max_numbers))
        
        for i, model in enumerate(default_models.models):
            digit = str(i + 1).rjust(max_digits, ' ')
            print(f"[{digit}] {Fore.BLUE}{model.name}")
        models_wanted = input(
            "\nPlease enter the models you want to enable, separated by commas " + \
            f"(i.e. '{Fore.BLUE}1 2 3 4{Fore.RESET}'). -> "
        )
        models_wanted = [int(e) for e in models_wanted.split(" ") if e.strip() != ""]
        print("Models wanted:", models_wanted)
        
        models_wanted : List[Model] = [default_models.models[i - 1] for i in models_wanted]
        print("Models wanted:", [e.name for e in models_wanted])
        
        
    config_current.enabled_model_classes.llm = enable_llms
    config_current.models = models_wanted
    
    enable_embeddings = input(f"Would you like to enable embeddings? {Fore.YELLOW}[Y/n] -> ").strip().lower()[:1]
    # if enable_llms == 'y' or enable_llms == '':
    enable_embeddings = (enable_embeddings == 'y')
    print(f"{Fore.GREEN if enable_embeddings else Fore.RED}Embeddings will{'' if enable_embeddings else ' not'} be enabled.")
    
    models_wanted_e = []
    if enable_embeddings:
        print(f"\nPlease select models from the following list:")
        max_numbers = len(default_models.other_local_models.embedding_models)
        max_digits = len(str(max_numbers))
        
        for i, model in enumerate(default_models.other_local_models.embedding_models):
            digit = str(i + 1).rjust(max_digits, ' ')
            print(f"[{digit}] {Fore.BLUE}{model.name}")
        models_wanted_e : str = input(
            "\nPlease enter the models you want to enable, separated by commas " + \
            f"(i.e. '{Fore.BLUE}1 2 3 4{Fore.RESET}'). -> "
        )
        models_wanted_e = [int(e) for e in models_wanted_e.split(" ") if e.strip() != ""]
        models_wanted_e : List[LocalModel] = [default_models.other_local_models.embedding_models[i - 1] for i in models_wanted_e]
    
    config_current.enabled_model_classes.embedding = enable_embeddings
    config_current.other_local_models.embedding_models = models_wanted_e
    
    
    enable_rerank = input(f"Would you like to enable rerank models? {Fore.YELLOW}[Y/n] -> ").strip().lower()[:1]
    enable_rerank = (enable_rerank == 'y')
    print(f"{Fore.GREEN if enable_rerank else Fore.RED}Rerank models will{'' if enable_rerank else ' not'} be enabled.")
    
    models_wanted_r = []
    if enable_rerank:
        print(f"\nPlease select models from the following list:")
        max_numbers = len(default_models.other_local_models.rerank_models)
        max_digits = len(str(max_numbers))
        
        for i, model in enumerate(default_models.other_local_models.rerank_models):
            digit = str(i + 1).rjust(max_digits, ' ')
            print(f"[{digit}] {Fore.BLUE}{model.name}")
        models_wanted_r : str = input(
            "\nPlease enter the models you want to enable, separated by commas " + \
            f"(i.e. '{Fore.BLUE}1 2 3 4{Fore.RESET}'). -> "
        )
        models_wanted_r = [int(e) for e in models_wanted_r.split(" ") if e.strip() != ""]
        models_wanted_r : List[LocalModel] = [default_models.other_local_models.rerank_models[i - 1] for i in models_wanted_r]
    
    config_current.enabled_model_classes.rerank = enable_rerank
    config_current.other_local_models.rerank_models = models_wanted_r
    
    return config_current

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
    
    SERVER_DIR = os.path.dirname(os.path.abspath(__file__))
    os.chdir(SERVER_DIR)
    
    with open ("QueryLake/other/default_config.json") as f:
        DEFAULT_CONFIG : Config = Config(**json.load(f))
        f.close()
    
    if os.path.exists("config.json"):
        with open("config.json") as f:
            CURRENT_CONFIG : Config = Config(**json.load(f))
            f.close()
    else:
        CURRENT_CONFIG = deepcopy(DEFAULT_CONFIG)
    
    if not os.path.exists("models"):
        os.mkdir("models")
    
    
    if all([v is None for v in vars(args).values()]):
        print(f"{Fore.BLUE}Running QueryLake setup")
        basic_setup_type = input(
            f"Would you like to select premade models " + \
            f"({Fore.YELLOW}P{Fore.RESET}) or load custom models ({Fore.YELLOW}C{Fore.RESET})? " + \
            f"{Fore.YELLOW}[P/C] -> "
        ).strip().lower()[:1]
        if basic_setup_type == 'c':
            CURRENT_CONFIG = custom_setup(CURRENT_CONFIG, DEFAULT_CONFIG, gpus)
        else:
            CURRENT_CONFIG = premade_models_setup(CURRENT_CONFIG, DEFAULT_CONFIG, gpus)
        
        all_model_ids = \
            [model.system_path for model in CURRENT_CONFIG.models] + \
            [model.source for model in CURRENT_CONFIG.other_local_models.embedding_models] + \
            [model.source for model in CURRENT_CONFIG.other_local_models.rerank_models]
        # all_model_ids = list(set(all_model_ids))
        print("All model ids:", all_model_ids)
        
        print(f"\n{Fore.GREEN}Downloading models...")
        
        models_downloaded = set()
        
        model_snaps : List[Union[Model, LocalModel]] = \
            CURRENT_CONFIG.models + \
            CURRENT_CONFIG.other_local_models.embedding_models + \
            CURRENT_CONFIG.other_local_models.rerank_models
        
        for i, model_snap in enumerate(model_snaps):
            if model_snap.source in models_downloaded:
                continue
            models_downloaded.add(model_snap.source)
            print(f"\n{Fore.GREEN}Downloading model [{Fore.RED}{i+1}/{len(model_snaps)}{Fore.GREEN}] {Fore.BLUE}{model_snap.source}")
            folder = huggingface_hub.snapshot_download(model_snap.source, cache_dir="models")
            folder = os.path.join(SERVER_DIR, folder)
            model_snap.system_path = folder
            # print("Model saved to", folder)
        
        with open("config.json", 'w') as f:
            json.dump(CURRENT_CONFIG.model_dump(), f, indent=4)
            f.close()
            


    

if __name__ == "__main__":
    main()