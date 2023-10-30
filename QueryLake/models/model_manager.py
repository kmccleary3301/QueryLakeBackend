from .langchain_sse import CustomStreamHandler, ThreadedGenerator
# from . import Exllama

from langchain.callbacks.manager import CallbackManager
from langchain import PromptTemplate, LLMChain
from datetime import datetime

import threading
import copy

from sqlmodel import Session, select
from ..database.sql_db_tables import model_query_raw, access_token, chat_session_new, model, chat_entry_model_response, chat_entry_user_question, model
import time
import json
from langchain.llms import LlamaCpp
from .exllama_langchain import Exllama
from .exllamav2_langchain import ExllamaV2

class LLMEnsemble:
    def __init__(self, database: Session, default_llm : str) -> None:
        self.max_instances = 1
        self.llm_instances = []
        self.default_llm = default_llm
        self.make_new_instance(database, default_llm)

    def make_new_instance(self, database, model_name):
        new_model = {
            "lock": False,
            "handler": CustomStreamHandler(None),
        }

        model_db_entry = database.exec(select(model).where(model.name == model_name)).first()

        parameters = json.loads(model_db_entry.default_settings)

        parameters["callback_manager"] = CallbackManager([new_model["handler"]])

        if (model_db_entry.necessary_loader == "exllama"):
            new_model["model"] = Exllama(**parameters)
        elif (model_db_entry.necessary_loader == "exllama_v2"):
            new_model["model"] = ExllamaV2(**parameters)
        elif (model_db_entry.necessary_loader == "llama_cpp"):
            new_model["model"] = LlamaCpp(**parameters)
        
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

    def chain(self, 
              user_name,
              database: Session,
              session_hash,
              question,
              parameters : dict = None,
              model_choice : str = "Llama-2-7b-Chat-GPTQ",
              context=None):
        """
        This function is for a model request. It creates a threaded generator, 
        substitutes in into the models callback manager, starts the function
        llm_thread in a thread, then returns the threaded generator.
        """

        print("Calling Model")

        if not (parameters is None or parameters == {}):
            if not self.validate_parameters(parameters):
                return None
        
        model_index = self.choose_llm_for_request()
        g = ThreadedGenerator()
        self.llm_instances[model_index]["handler"].gen = g
        threading.Thread(target=self.llm_thread, args=(g, 
                                                       user_name,
                                                       database, 
                                                       session_hash,
                                                       question,
                                                       model_index, 
                                                       parameters, context)).start()
        return g


    def llm_thread(self, 
                   g, 
                   user_name,
                   database: Session,
                   session_hash,
                   question,
                   model_index, 
                   parameters,
                   context):
        """
        This function is run in a thread, outside of normal execution.
        """
        first_of_session = False
        try:
            previous_values = {}
            # if not (parameters is None or parameters == {}):
            #     for key, _ in parameters.items():
            #         previous_values[key] = self.llm_instances[model_index]["model"].__dict__[key]
            #     self.llm_instances[model_index]["model"].refresh_params(parameters)
            
            while self.llm_instances[model_index]["lock"] == True:
                pass


            start_time = time.time()
            self.llm_instances[model_index]["lock"] = True
            session = database.exec(select(chat_session_new).where(chat_session_new.hash_id == session_hash)).first()
            
            model_entry_db = database.exec(select(model).where(model.name == session.model)).first()
            
            system_instruction_prompt = model_entry_db.system_instruction_wrapper.replace("{system_instruction}", model_entry_db.default_system_instruction)
            final_prompt = ""
            if not context is None:
                final_prompt += model_entry_db.context_wrapper.replace("{context}", context)

            bot_responses_previous = database.exec(select(chat_entry_model_response).where(chat_entry_model_response.chat_session_id == session.id)).all()

            bot_responses_previous = sorted(bot_responses_previous, key=lambda x: x.timestamp)

            if len(bot_responses_previous) == 0:
                session.title = question.split(r"[.|?|!|\n|\t]")[-1]

            system_instruction_prompt_token_count = self.llm_instances[model_index]["model"].get_num_tokens(system_instruction_prompt)

            cutoff_space = self.llm_instances[model_index]["model"].config.max_seq_len - 1000
            
            chat_history, sum_tokens = [], system_instruction_prompt_token_count

            for bot_response in bot_responses_previous[::-1]:
                question_previous = database.exec(select(chat_entry_user_question).where(chat_entry_user_question.id == bot_response.chat_entry_response_to)).first()
                chat_entry = ""
                
                chat_entry += model_entry_db.user_question_wrapper.replace("{question}", question_previous.content)
                chat_entry += model_entry_db.bot_response_wrapper.replace("{response}", bot_response.content)

                chat_entry_token_count = self.llm_instances[model_index]["model"].get_num_tokens(chat_entry)
                sum_tokens += chat_entry_token_count
                if sum_tokens > cutoff_space:
                    break
                chat_history.append(chat_entry)
            
            for entry in chat_history[::-1]:
                final_prompt += entry

            final_prompt += model_entry_db.user_question_wrapper.replace("{question}", question)

            prompt_template = PromptTemplate(input_variables=["question"], template=system_instruction_prompt+"{question}")
            prompt_medium = prompt_template.format(question=final_prompt)
            # final_prompt = prompt_template.format(question=prompt)
            llm_chain = LLMChain(prompt=prompt_template, llm=self.llm_instances[model_index]["model"])
            # llm_chain.run({"question": prompt, "system_instruction": system_instruction})

            print("final_prompt")
            print(prompt_medium)
            
            response = llm_chain.run(prompt_medium)
            end_time = time.time()

            print(response)

            tokens_add = self.llm_instances[model_index]["handler"].tokens_generated
            first_key = database.exec(select(access_token).where(access_token.author_user_name == user_name)).first()

            first_key.tokens_used += tokens_add
            database.commit()
            request_data = {
                "prompt": final_prompt,
                "response" : response,
                "response_size_tokens": self.llm_instances[model_index]["handler"].tokens_generated,
                "prompt_size_tokens": self.llm_instances[model_index]["model"].get_num_tokens(final_prompt),
                "model": self.llm_instances[model_index]["model"].model_path.split("/")[-1],
                "timestamp": time.time(),
                "time_taken": end_time-start_time,
                "model_settings": str(self.llm_instances[model_index]["model"].__dict__),
                # "author": user_name,
                "access_token_id": first_key.id
            }
            new_request = model_query_raw(**request_data)
            database.add(new_request)
            database.commit()

            
            # new_user = sql_db.User(**user_data)
            question_entry = chat_entry_user_question(
                chat_session_id=session.id,
                timestamp=time.time(),
                content=question,
            )
            database.add(question_entry)
            database.commit()
            database.flush()

            model_response = chat_entry_model_response(
                chat_session_id=session.id,
                timestamp=time.time(),
                content=response,
                chat_entry_response_to=question_entry.id,
                #model query raw id.
            )
            database.add(model_response)
            database.commit()
        finally:
            self.llm_instances[model_index]["handler"].tokens_generated = 0
            self.llm_instances[model_index]["model"].callback_manager = None
            g.close()
            self.llm_instances[model_index]["handler"].gen = None
            self.llm_instances[model_index]["lock"] = False
            # if previous_values != {}: # Reset the model parameters to normal if they were changed.
            #     # self.llm_instances[model_index]["model"].__dict__.update(previous_values)
            #     self.llm_instances[model_index]["model"].refresh_params(previous_values)
            del previous_values