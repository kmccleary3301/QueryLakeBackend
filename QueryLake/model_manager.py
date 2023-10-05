from .langchain_sse import CustomStreamHandler, ThreadedGenerator
# from . import Exllama

from langchain.callbacks.manager import CallbackManager
from langchain import PromptTemplate, LLMChain

import threading
import copy

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

    def delete_model(self, model_index : int) -> None:
        """
        Properly deletes a model and clears the memory.
        """
        if str(type(self.llm_instances[model_index]["model"].client)) == "ExLlama":
             self.llm_instances[model_index]["model"].client.free_unmanaged()
             del self.llm_instances[model_index]["model"]
             del self.llm_instances[model_index]

    def choose_llm_for_request(self):
        """
        This class is structured to cycle multiple instances of LLMs
        to handle heavy server load. This method is designed to select the most available
        llm instance from the ensemble.
        """
        return 0
    
    def validate_parameters(self, parameters : dict) -> bool:
        """
        Should return true or false if new model parameters are valid.
        NTF
        """
        return True

    def chain(self, prompt, template, parameters : dict = None):
        """
        This function is for a model request. It creates a threaded generator, 
        substitutes in into the models callback manager, starts the function
        llm_thread in a thread, then returns the threaded generator.
        """

        if not (parameters is None or parameters == {}):
            if not self.validate_parameters(parameters):
                return None
        
        model_index = self.choose_llm_for_request()
        g = ThreadedGenerator()
        self.llm_instances[model_index]["handler"].gen = g
        threading.Thread(target=self.llm_thread, args=(g, prompt, template, model_index, parameters)).start()
        return g


    def llm_thread(self, g, prompt, template, model_index, parameters):
        """
        This function is run in a thread, outside of normal execution.
        """
        try:
            previous_values = {}
            if not (parameters is None or parameters == {}):
                for key, _ in parameters.items():
                    previous_values[key] = self.llm_instances[model_index]["model"].__dict__[key]
                self.llm_instances[model_index]["model"].refresh_params(parameters)
            
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
            if previous_values != {}: # Reset the model parameters to normal if they were changed.
                # self.llm_instances[model_index]["model"].__dict__.update(previous_values)
                self.llm_instances[model_index]["model"].refresh_params(previous_values)
            del previous_values