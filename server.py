import asyncio
import logging

import uvicorn
from fastapi import FastAPI
from sse_starlette.sse import EventSourceResponse
from starlette.requests import Request
from fastapi.middleware.cors import CORSMiddleware


from typing import Any, Dict, Iterator, List, Optional
import threading
import queue
import sys
from exllama.langchain import Exllama
# from langchain.llms import OpenAI
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from typing import Any, Dict, List, Union
from langchain.schema import AgentAction, AgentFinish, LLMResult
from langchain.callbacks.base import BaseCallbackHandler
from langchain import PromptTemplate, LLMChain

print("Imports Finished")

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GLOBAL_LLM_CONFIG = {
    "streaming": True,
    "model_path": "/home/user/python_projects/langchain/Llama-2-7b-Chat-GPTQ", 
    "lora_path": None,
    "temperature": 0.3,
    "typical": .7,
    "verbose": True,
    "max_seq_len": 2095,
    "fused_attn": False,
    # "beams": 1, 
    # "beam_length": 40, 
    # "alpha_value": 1.0, #For use with any models
    # "compress_pos_emb": 4.0, #For use with superhot
    # "set_auto_map": "3, 2" #Gpu split, this will split 3gigs/2gigs
    # "stop_sequences": ["### Input", "### Response", "### Instruction", "Human:", "Assistant", "User:", "AI:"],
}

template_microbiology_question = """
You are a world class professor in microbiology.
Give a long and insightful response to the following question.

Question: {question}

Response: 
"""

class ThreadedGenerator:
    def __init__(self):
        self.queue = queue.Queue()

    def __iter__(self):
        return self

    def __next__(self):
        item = self.queue.get()
        if item is StopIteration: raise item
        return item

    def send(self, data):
        self.queue.put(data)

    def close(self):
        self.queue.put(StopIteration)

class CustomStreamHandler(BaseCallbackHandler):
    """Callback handler for streaming. Only works with LLMs that support streaming."""

    def __init__(self, gen) -> None:
        super().__init__()
        self.gen = gen

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Run when LLM starts running."""

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        if not self.gen is None:
            self.gen.send(token)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when LLM ends running."""
        # print("LLM End")
        if not self.gen is None:
            self.gen.send("-DONE-")
        # self.gen.close()

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Run when LLM errors."""

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Run when chain starts running."""

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Run when chain ends running."""
        # print("Chain End")

    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Run when chain errors."""

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> None:
        """Run when tool starts running."""

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Run on agent action."""
        pass

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Run when tool ends running."""
        print("Tool End")
        # self.gen.send("---Tool End---")

    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Run when tool errors."""
        # print("Agent Finish")
        # self.gen.send("---END AGENT---")

    def on_text(self, text: str, **kwargs: Any) -> None:
        """Run on arbitrary text."""

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
        """Run on agent end."""
        # print("Agent Finish")
        # self.gen.send("---END AGENT---")

class LLMEnsemble:
    def __init__(self) -> None:
        self.max_instances = 1
        self.llm_instances = []
        self.make_new_instance(GLOBAL_LLM_CONFIG)

    def make_new_instance(self, parameters):
        new_model = {
            "lock": False,
            "handler": CustomStreamHandler(None),
            }
        parameters["callback_manager"] = CallbackManager([new_model["handler"]])
        new_model["model"] = Exllama(**parameters)
        self.llm_instances.append(new_model)

    def choose_llm_for_request(self):
        """
        This class is structured to cycle multiple instances of LLMs
        to handle heavy server load. This method is designed to select the most available
        llm instance from the ensemble.
        """
        return 0

    def chain(self, prompt, parameters : dict = None):
        print("Chain has been called")
        model_index = self.choose_llm_for_request()
        previous_attr = {}
        # if not parameters is None:
        #     for k, v in parameters.items():
        #         previous_attr[k] = getattr(self.llm_instances[0], k)
        #         setattr(self.llm_instances[0], k, v)
        g = ThreadedGenerator()
        self.llm_instances[model_index]["handler"].gen = g
        threading.Thread(target=self.llm_thread, args=(g, prompt, model_index, previous_attr)).start()
        # print()
        return g

    def llm_thread(self, g, prompt, model_index, reset_values):
        try:
            while self.llm_instances[model_index]["lock"] == True:
                pass
            
            self.llm_instances[model_index]["lock"] = True
            prompt = PromptTemplate(input_variables=["question"], template=template_microbiology_question)
            
            llm_chain = LLMChain(prompt=prompt, llm=self.llm_instances[model_index]["model"])

            llm_chain.run(prompt)
            # self.llm_instances[model_index]["model"](prompt)
        finally:
            print("Response finished")
            self.llm_instances[model_index]["model"].callback_manager = None
            g.close()
            self.llm_instances[model_index]["handler"].gen = None
            self.llm_instances[model_index]["lock"] = False

GlobalLLMEnsemble = LLMEnsemble()

@app.get("/chat")
async def chat(req: Request):
    """
    Returns Langchain generation via SSE
    """

    arguments = req.query_params._dict
    print("chat called")
    # print("Parameters:", parameters)
    print("request:", arguments)
    prompt = "What is a telomere?"
    # result = EventSourceResponse(chain(prompt))
    result = EventSourceResponse(GlobalLLMEnsemble.chain(prompt))
    print("Returning", type(result))
    return result

@app.get("/endless")
async def endless(req: Request, **kwargs):
    """Simulates and endless stream

    In case of server shutdown the running task has to be stopped via signal handler in order
    to enable proper server shutdown. Otherwise, there will be dangling tasks preventing proper shutdown.
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