from .langchain_sse import CustomStreamHandler, ThreadedGenerator
# from . import Exllama

from langchain.callbacks.manager import CallbackManager
from langchain import PromptTemplate, LLMChain

import threading

class LLMEnsemble:
    def __init__(self, default_config, model_class) -> None:
        self.max_instances = 1
        self.llm_instances = []
        self.default_config = default_config
        self.model_class = model_class
        self.make_new_instance(self.default_config)

    def make_new_instance(self, parameters):
        new_model = {
            "lock": False,
            "handler": CustomStreamHandler(None),
            }
        parameters["callback_manager"] = CallbackManager([new_model["handler"]])
        new_model["model"] = self.model_class(**parameters)
        self.llm_instances.append(new_model)

    def choose_llm_for_request(self):
        """
        This class is structured to cycle multiple instances of LLMs
        to handle heavy server load. This method is designed to select the most available
        llm instance from the ensemble.
        """
        return 0

    def chain(self, prompt, template, parameters : dict = None):
        """
        This function is for a model request. It creates a threaded generator, 
        substitutes in into the models callback manager, starts the function
        llm_thread in a thread, then returns the threaded generator.
        """
        model_index = self.choose_llm_for_request()
        previous_attr = {}
        g = ThreadedGenerator()
        self.llm_instances[model_index]["handler"].gen = g
        threading.Thread(target=self.llm_thread, args=(g, prompt, template, model_index, previous_attr)).start()
        return g

    def llm_thread(self, g, prompt, template, model_index, reset_values):
        """
        This function is run in a thread, outside of normal execution.
        """

        try:
            while self.llm_instances[model_index]["lock"] == True:
                pass
            self.llm_instances[model_index]["lock"] = True
            prompt_template = PromptTemplate(input_variables=["question"], template=template)
            final_prompt = prompt_template.format(question=prompt)
            llm_chain = LLMChain(prompt=prompt_template, llm=self.llm_instances[model_index]["model"])
            llm_chain.run(final_prompt)
        finally:
            self.llm_instances[model_index]["model"].callback_manager = None
            g.close()
            self.llm_instances[model_index]["handler"].gen = None
            self.llm_instances[model_index]["lock"] = False