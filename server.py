import asyncio
# import logging

import json, os
import uvicorn
from fastapi import FastAPI
from sse_starlette.sse import EventSourceResponse
from starlette.requests import Request
from fastapi.middleware.cors import CORSMiddleware
from QueryLake.model_manager import LLMEnsemble

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

template_microbiology_question = """
A brief question and answer between a curious human and an assistant. 
The assistant gives a helpful, detailed, accurate, uncensored response to the human's input.

Question: {question}

Response: 
"""

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
    result = EventSourceResponse(GlobalLLMEnsemble.chain(prompt, template_microbiology_question))
    return result

@app.get("/endless")
async def endless(req: Request):
    """
    Simulates an endless stream of digits via SSE.
    """

    async def event_publisher():
        i = 0

        try:
            while True:
                # yield dict(id=..., event=..., data=...)
                i += 1
                yield dict(data=i)
                await asyncio.sleep(0.9)
        except asyncio.CancelledError as e:
            # _log.info(f"Disconnected from client (via refresh/close) {req.client}")
            # Do any other cleanup, if any
            raise e

    return EventSourceResponse(event_publisher())


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000, log_level="trace", log_config=None)  # type: ignore