import asyncio
# import logging

from typing import Annotated
import json, os
import uvicorn
from fastapi import FastAPI, File, UploadFile
from sse_starlette.sse import EventSourceResponse
from starlette.requests import Request
from fastapi.middleware.cors import CORSMiddleware
from QueryLake.model_manager_old import LLMEnsemble
from QueryLake import instruction_templates, authentication
# from QueryLake.sql_db import User, File, 
# from sqlmodel import Field, Session, SQLModel, create_engine, select
# from sqlmodel import Field, Session, SQLModel, create_engine, select
# from typing import Optional
from sqlmodel import Field, Session, SQLModel, create_engine, select
from sqlalchemy.orm import scoped_session
from sqlalchemy.orm import sessionmaker



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
    # from QueryLake import ExllamaV2
    GLOBAL_SETTINGS["loader_class"] = Exllama

GLOBAL_LLM_CONFIG = GLOBAL_SETTINGS["default_llm_config"]

GlobalLLMEnsemble = LLMEnsemble(GLOBAL_LLM_CONFIG, GLOBAL_SETTINGS["loader_class"])
# engine = create_engine("sqlite:///database.db")
engine = create_engine("sqlite:///user_data.db", connect_args={"check_same_thread" : False})
SQLModel.metadata.create_all(engine)
# session_factory = sessionmaker(bind=engine)
# Session = scoped_session(session_factory)
database = Session(engine)
token_tracker = authentication.TokenTracker(database)

# SQLALCHEMY_DATABASE_URL = 'sqlite+pysqlite:///.db.sqlite3:' 
# engine_2 = create_engine(SQLALCHEMY_DATABASE_URL,
#                            connect_args={"check_same_thread": False},
#                            echo=True,
#                            future=True
#                            ) 
# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine_2) 

# session_make = scoped_session(sessionmaker(bind=engine_2), scopefunc=get_current_request)

# Base = declarative_base()


# SQLModel.metadata.create_all(engine)
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

    get_user_id = authentication.get_user_id(database, arguments["username"], arguments["password_prehash"])
    if get_user_id < 0:
        return EventSourceResponse("Invalid Authentication")

    result = EventSourceResponse(GlobalLLMEnsemble.chain(
                                    user_name=arguments["username"],
                                    token_tracker=token_tracker,
                                    database=database,
                                    prompt=prompt,
                                    system_instruction=instruction_templates.latex_basic_system_instruction,
                                    template=instruction_templates.llama2_chat_latex_test_1
                                ))
    return result

@app.post("/file")
async def create_file(file: Annotated[bytes, File(description="A file read as bytes")]):
    return {"file_size": len(file)}

@app.post("/uploadfile")
async def create_upload_file(req: Request, file: UploadFile):
    if len(await file.read()) >= 83886080:
        return {"file_upload_success": False, "note": "Your file is more than 80MB"}
    print("Upload file called with:", file)
    print(file.filename)
    arguments = req.query_params._dict
    print("request:", arguments)
    # return {"filename": file.filename}
    # try:
    result = authentication.file_save(database=database, 
                                      name=arguments["name"], 
                                      password_prehash=arguments["password_prehashed"], 
                                      file=file)
    print(result)
    return result
    # except:
    #     return {"file_upload_success": False, "note": "Server Error"}

@app.post("/create_account")
def create_account(req: Request):
    arguments = req.query_params._dict
    result = authentication.add_user(database=database, name=arguments["name"], password=arguments["password"])
    # Add some code for session rotations.
    return result

@app.post("/login")
def login(req: Request):
    arguments = req.query_params._dict
    result = authentication.auth_user(database=database, username=arguments["name"], password=arguments["password"])
    # Add some code for session rotations.
    return result

@app.post("/auth")
def authenticate(req: Request):
    arguments = req.query_params._dict
    print(arguments)
    return {"result": authentication.hash_function(arguments["input"])}


if __name__ == "__main__":
    # with Session(engine) as session:
    #     statement = select(sql_db.Hero).where(sql_db.Hero.name == "Spider-Boy")
    #     hero = session.exec(statement).first()
    #     print(hero)
    uvicorn.run(app, host="0.0.0.0", port=5000, log_level="trace", log_config=None)  # type: ignore