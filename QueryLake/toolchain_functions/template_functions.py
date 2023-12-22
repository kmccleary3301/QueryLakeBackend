import queue
import threading
import time
import asyncio
from ..models.langchain_sse import ThreadedGenerator

def create_document_summary(file_input):
    return {
        "document_summary": "Document is about war and peace"
    }
    


def answer_user_question(user_question : str, document_summary : str, user_provided_chat_history : list, provided_generator : ThreadedGenerator):
    print("Got inputs", document_summary, user_question)

    
    # model_reponse = "\'"+user_question+"\'"+ " is a dumb question."
    # test_generator = ThreadedGenerator()
    new_chat = "DOCUMENT_SUMMARY: " + document_summary + "\n\n"
    for entry in user_provided_chat_history:
        if entry["role"] == "assistant":
            new_chat += "ASSISTANT: "+entry["content"] + "\n\n"
        else:
            new_chat += "USER: "+entry["content"] + "\n\n"

    new_chat += "USER: " + user_question
    print("NEW CHAT")
    print(new_chat)

    # threading.Thread(target=test_thread_target).start()
    thread_make = threading.Thread(target=yield_numbers_thread_target, kwargs={"generator": provided_generator, "appendage" : "\'"+user_question+"\'"+ " is a dumb question."})
    thread_make.start()
    return {
        "model_response": provided_generator
    }

def test_thread_target():
    print("Thread target called")

def yield_numbers_thread_target(generator : ThreadedGenerator, appendage : str):
    print("Calling yield numbers thread target")
    for i in range(10):
        generator.send(str(i))
        time.sleep(2)
    generator.send("\n")
    generator.send(appendage)
    print("Closing generator previous")
    generator.close()

def generate_split_output(input):
    return {
        "split_output": [i for i in input]
    }

def split_processing_single(single_input):
    return {
        "token": single_input
    }

def organize_split_input(split_input):
    return {
        "recombined_input": "".join(split_input)
    }