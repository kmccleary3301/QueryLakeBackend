from .chat_sessions import *
from .collections import *
from .document import *
from .hashing import *
from .organizations import *
from .user_auth import *
from .web_search import *
from .llm_model_calls import *

server_dir = "/".join(os.path.dirname(os.path.realpath(__file__)).split("/")[:-2])
print(server_dir)
if not os.path.exists(server_dir+"/user_db"):
    os.mkdir(server_dir+"/user_db")
if not os.path.exists(server_dir+"/user_db/files"):
    os.mkdir(server_dir+"/user_db/files")
user_db_path = server_dir+"/user_db/files/"

system_arguments = ["database", "vector_database", "llm_ensemble", "public_key", "server_private_key"]

excluded_member_function_descriptions = [
    "and_", 
    "is_", 
    "not_", 
    "select", 
    "create_embeddings_in_database", 
    "query_database",
    "prune_empty_chat_sessions",
    "get_user",
    "get_user_id",
    "upload_document_to_collection",
    "get_user_private_key",
    "get_organization_private_key"
]

async_member_functions = [
    "llm_call_chat_session",
    "fetch_document"
]

        