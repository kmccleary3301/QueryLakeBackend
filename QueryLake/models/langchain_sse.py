import queue
import urllib.parse
from typing import Any, Dict, List, Union
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish, LLMResult
import time
from threading import Thread

class ThreadedGenerator:
    def __init__(self, encode_hex : bool = True):
        self.encode_hex = encode_hex
        self.done = False
        self.sent_values = []
        self.report_callbacks = []
        self.queue = queue.Queue()

    def __iter__(self):
        return self

    def __next__(self):
        item = self.queue.get()
        if item is StopIteration: raise item
        return item

    def send(self, data):
        assert type(data) is str, "Non-string passed to threaded generator"
        if not data is None:
            if self.encode_hex:
                self.queue.put(data.encode("utf-8").hex())
            else:
                self.queue.put(data)
            for callback in self.report_callbacks:
                callback(data)
            self.sent_values.append(data)

    def close(self):
        print("Closing Generator")
        self.queue.put("<<CLOSE>>")
        self.done = True
        self.queue.put(StopIteration)

def generatorMessageTarget(g : ThreadedGenerator, message, timeout : float = 0.010):
    time.sleep(timeout)
    if type(message) is list:
        for msg in message:
            g.send(msg)
    elif type(message) is str:
        g.send(message)

def raiseErrorInGenerator(g : ThreadedGenerator, message, timeout : float = 0.010):
    Thread(target=generatorMessageTarget, args=(g, message), kwargs={"timeout": timeout}).start()

def ErrorAsGenerator(message):
    g = ThreadedGenerator
    Thread(target=generatorMessageTarget, args=(g, message), kwargs={"timeout": 0.010}).start()
    return g

class CustomStreamHandler(BaseCallbackHandler):
    """Callback handler for streaming. Only works with LLMs that support streaming."""

    def __init__(self, gen : ThreadedGenerator) -> None:
        super().__init__()
        self.gen = gen
        self.tokens_generated = 0

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Run when LLM starts running."""

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        if not self.gen is None:
            # self.gen.send(urllib.parse.quote(token, safe='_!"#$%&\'()*+,/:;=?@[]'))
            self.gen.send(token)
        self.tokens_generated += 1

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when LLM ends running."""
        if not self.gen is None:
            self.gen.close()

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

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Run when tool ends running."""

    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Run when tool errors."""

    def on_text(self, text: str, **kwargs: Any) -> None:
        """Run on arbitrary text."""

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
        """Run on agent end."""

