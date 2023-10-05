import asyncio
# import logging

from typing import Annotated
import json, os
import uvicorn
from fastapi import FastAPI, File, UploadFile
from sse_starlette.sse import EventSourceResponse
from starlette.requests import Request
from fastapi.middleware.cors import CORSMiddleware
from QueryLake.model_manager import LLMEnsemble
from QueryLake import instruction_templates


app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
    
os.chdir(os.path.dirname(os.path.realpath(__file__)))
with open("config.json", 'r', encoding='utf-8') as f:
    file_read = f.read()
    f.close()
GLOBAL_SETTINGS = json.loads(file_read)
if GLOBAL_SETTINGS["use_llama_cpp"]:
    from langchain.llms import LlamaCpp
    GLOBAL_SETTINGS["loader_class"] = LlamaCpp
else:
    from QueryLake import Exllama
    GLOBAL_SETTINGS["loader_class"] = Exllama

GLOBAL_LLM_CONFIG = GLOBAL_SETTINGS["default_llm_config"]

GlobalLLMEnsemble = LLMEnsemble(GLOBAL_LLM_CONFIG, GLOBAL_SETTINGS["loader_class"])
print("Model Loaded")

@app.get("/chat")
async def chat(req: Request):
    """
    Returns Langchain generation via SSE
    """

    arguments = req.query_params._dict
    print("request:", arguments)
    if "query" in arguments:
        prompt = arguments["query"]
    else:
        prompt = "..."
    tmp_params = {
        "temperature": 0.2,
        "top_k": 50,
        # "max_seq_len": 4095,
        'token_repetition_penalty_max': 1.2,
        'token_repetition_penalty_sustain': 1,
        'token_repetition_penalty_decay': 1,
    }
    result = EventSourceResponse(GlobalLLMEnsemble.chain(prompt, instruction_templates.llama2_chat_latex_test_1))
    return result

@app.post("/file")
async def create_file(file: Annotated[bytes, File(description="A file read as bytes")]):
    return {"file_size": len(file)}

@app.post("/uploadfile")
async def create_upload_file(req: Request, file: UploadFile):
    print("Upload file called with:", file)
    print(file.filename)
    arguments = req.query_params._dict
    print("request:", arguments)
    return {"filename": file.filename}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000, log_level="trace", log_config=None)  # type: ignore