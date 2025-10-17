from .collections import *
from .document import *
from .hashing import *
from .organizations import *
from .user_auth import *
from .web_search import *
# from .llm_model_calls import *
from .toolchains import *
from .custom_model_functions.standalone_question import llm_isolate_question
from .custom_model_functions.create_conversation_title import llm_make_conversation_title
from .custom_model_functions.multi_search import llm_multistep_search
from .custom_model_functions.self_guided_search import self_guided_search
from .search import *
from .usage_tracking import *
from .custom_model_functions.surya import process_pdf_with_surya, process_pdf_with_surya_2
from ..operation_classes.ray_vllm_class import format_chat_history
# from .patho_report_stager import *

server_dir = "/".join(os.path.dirname(os.path.realpath(__file__)).split("/")[:-2])

system_arguments = ["database", "vector_database", "llm_ensemble", "public_key", "server_private_key"]


def health():
    """Lightweight health probe for /api/health."""
    return {"success": True, "note": "QueryLake API responsive"}


def healthz():
    """Alias so /api/healthz mirrors /healthz."""
    return {"success": True}

excluded_member_function_descriptions = [
    # Janitorial
    "prune_empty_chat_sessions",
    "prune_inactive_toolchain_sessions",
    
    # Imported Builtins
    "get",
    "copy",
    "deepcopy",
    "gather",
    
    # SQLModel functions
    "and_", 
    "is_", 
    "not_", 
    "select", 

    # User Auth
    "get_user",
    "get_user_id",
    "get_user_private_key",
    "get_organization_private_key",
    
    # Vector Database
    "create_embeddings_in_database", 
    # "query_database",
    
    # Documents
    "upload_document_to_collection",
    "assert_collections_viewable",
    
    # Toolchains
    "save_toolchain_session",
    "retrieve_files_for_session",
    "session_notification",
    "delete_file_on_delay",
    
    # Cryptography
    "aes_decrypt_zip_file",
    "aes_encrypt_zip_file",
    "aes_delete_file_from_zip_blob",
    
    # Usage Tracking
    "increment_merge_dictionaries",
    "increment_usage_tally",
    
    # Other
    "Field",
    "delete",
    "text",
    "find_overlap"
]

exposed_api = [
    "add_user",
    "assess_categories_multiple",
    "assess_category_single",
    "convert_dict_list_to_markdown",
    "craft_document_access_token",
    "craft_google_query_from_history",
    "create_api_key",
    "create_chat_session",
    "create_document_collection",
    "create_organization",
    "create_toolchain_session",
    "create_website_embeddings",
    "deepcopy",
    "delete_api_key",
    "delete_document",
    "download_dict_list_as_csv",
    "embed_urls",
    "fetch_all_collections",
    "fetch_chat_sessions",
    "fetch_collection",
    "fetch_document",
    "fetch_document_collections_belonging_to",
    "fetch_memberships",
    "fetch_memberships_of_organization",
    "fetch_session",
    "fetch_toolchain_session",
    "fetch_toolchain_sessions",
    "generate_measurement",
    "get_available_models",
    "get_available_toolchains",
    "get_document_secure",
    "get_file_bytes",
    "get_openai_api_key",
    "get_serp_key",
    "get_session_state",
    "get_toolchain_output_file_response",
    "hash_function",
    "hide_chat_session",
    "invite_user_to_organization",
    "llm_call_chat_session",
    "llm_call_chat_session_direct",
    "llm_call_model_synchronous",
    "llm_isolate_question",
    "llm_make_conversation_title",
    "llm_multistep_search",
    "login",
    "modify_document_collection",
    "modify_user_external_providers",
    "ocr_pdf_file",
    "parse_PDFs",
    "parse_url",
    "parse_urls",
    "perform_search_query",
    "query_database",
    "random_hash",
    "resolve_organization_invitation",
    "retrieve_toolchain_from_db",
    "run_function_safe",
    "safe_serialize",
    "search_google",
    "set_organization_openai_id",
    "set_organization_serp_key",
    "set_user_openai_api_key",
    "set_user_serp_key",
    "shuffle_seed",
    "stage_breast_cancer_report",
    "toolchain_entry_call",
    "toolchain_event_call",
    "toolchain_file_upload_event_call",
    "toolchain_session_notification",
    "toolchain_stream_node_propagation_call",
    "upload_document",
    "web_search"
]

all_function_names = [
	"add_user",
	"aes_decrypt_zip_file",
	"aes_encrypt_zip_file",
	"and_",
	"assess_categories_multiple",
	"assess_category_single",
	"convert_dict_list_to_markdown",
	"craft_document_access_token",
	"craft_google_query_from_history",
	"create_chat_session",
	"create_document_collection",
	"create_embeddings_in_database",
	"create_organization",
	"create_toolchain_session",
	"create_website_embeddings",
	"deepcopy",
	"delete_document",
	"delete_file_on_delay",
	"download_dict_list_as_csv",
	"embed_urls",
	"fetch_all_collections",
	"fetch_chat_sessions",
	"fetch_collection",
	"fetch_document",
	"fetch_document_collections_belonging_to",
	"fetch_memberships",
	"fetch_memberships_of_organization",
	"fetch_session",
	"fetch_toolchain_session",
	"fetch_toolchain_sessions",
	"generate_measurement",
	"get",
	"get_available_models",
	"get_available_toolchains",
	"get_document_secure",
	"get_file_bytes",
	"get_openai_api_key",
	"get_organization_private_key",
	"get_serp_key",
	"get_session_state",
	"get_toolchain_output_file_response",
	"get_user",
	"get_user_id",
	"get_user_private_key",
	"hash_function",
	"hide_chat_session",
	"invite_user_to_organization",
	"is_",
	"llm_call_chat_session",
	"llm_call_chat_session_direct",
	"llm_call_model_synchronous",
	"login",
	"modify_document_collection",
	"not_",
	"ocr_pdf_file",
	"parse_PDFs",
	"parse_url",
	"parse_urls",
	"perform_search_query",
	"prune_empty_chat_sessions",
	"query_database",
	"query_vector_db",
	"random_hash",
	"resolve_organization_invitation",
	"retrieve_files_for_session",
	"retrieve_toolchain_from_db",
	"run_function_safe",
	"safe_serialize",
	"save_toolchain_session",
	"search_google",
	"select",
	"set_organization_openai_id",
	"set_organization_serp_key",
	"set_user_openai_api_key",
	"set_user_serp_key",
	"shuffle_seed",
	"stage_breast_cancer_report",
	"toolchain_entry_call",
	"toolchain_event_call",
	"toolchain_file_upload_event_call",
	"toolchain_session_notification",
	"toolchain_stream_node_propagation_call",
	"upload_document"
]


# included_member_function_descriptions = 


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
query_vector_db
resolve_organization_invitation
set_organization_openai_id
set_organization_serp_key
set_user_openai_api_key
set_user_serp_key
upload_document
""".split("\n")

remaining_independent_api_functions = [x.strip() for x in remaining_independent_api_functions if x.strip() != ""]

from ..typing.toolchains import ToolChainV2
from ..database import sql_db_tables
from sqlmodel import select
import json as _json

def register_toolchain_v2(
    database: Session,
    global_config: Config,
    auth: AuthInputType,
    toolchain: dict,
) -> dict:
    """
    Admin helper to upsert a native v2 toolchain definition directly into the DB.
    """
    # TODO: Add proper admin/role checks
    tc = ToolChainV2.model_validate(toolchain)
    existing = database.exec(
        select(sql_db_tables.toolchain).where(sql_db_tables.toolchain.toolchain_id == tc.id)
    ).first()
    if existing:
        existing.title = tc.name
        existing.category = tc.category
        existing.content = _json.dumps(tc.model_dump(exclude_none=True), indent=4)
    else:
        row = sql_db_tables.toolchain(
            toolchain_id=tc.id,
            title=tc.name,
            category=tc.category,
            content=_json.dumps(tc.model_dump(exclude_none=True), indent=4),
        )
        database.add(row)
    database.commit()
    return {"toolchain_id": tc.id}


# -------------------------
# Files helper API (v2)
# -------------------------

async def files_process_version(
    database: Session,
    auth: AuthInputType,
    umbrella,
    file_id: str,
    version_id: str,
):
    """Process a specific file version through the Files runtime (Phase 2).

    Returns {"job_id": ..., "status": ...}.
    """
    _ = database  # present for parity; not required here
    return await umbrella.files_runtime.process_version(file_id, version_id, auth=auth)
