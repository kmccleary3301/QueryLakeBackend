from .chat_sessions import *
from .collections import *
from .document import *
from .hashing import *
from .organizations import *
from .user_auth import *
from .web_search import *
from .llm_model_calls import *
from .toolchains import *
from .patho_report_stager import *

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
    "get_organization_private_key",
    "load_toolchain_into_carousel",
    "save_toolchain_session",
    "prune_inactive_toolchain_sessions",
    "retrieve_files_for_session"
    "session_notification",
    "delete_file_on_delay",
    "aes_decrypt_zip_file",
    "aes_encrypt_zip_file",
    "deepcopy"
]

async_member_functions = [
    "llm_call_chat_session",
    "fetch_document",
    "get_session_global_generator",
    "get_generator_by_id",
    "toolchain_entry_call",
    "toolchain_event_call",
    "toolchain_stream_node_propagation_call",
    "get_toolchain_output_file_response"
]

remaining_independent_api_functions = """
add_user
craft_document_access_token
create_document_collection
create_organization
delete_document
fetch_all_collections
fetch_collection
fetch_document
fetch_document_collections_belonging_to
fetch_memberships
fetch_memberships_of_organization
get_available_models
get_available_toolchains
get_document_secure
get_openai_api_key
get_serp_key
invite_user_to_organization
login
modify_document_collection
resolve_organization_invitation
set_organization_openai_id
set_organization_serp_key
set_user_openai_api_key
set_user_serp_key
upload_document
""".split("\n")

remaining_independent_api_functions = [x.strip() for x in remaining_independent_api_functions if x.strip() != ""]



