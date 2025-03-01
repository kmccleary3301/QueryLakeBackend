from fastapi import UploadFile, Request
from fastapi.responses import StreamingResponse, FileResponse, Response
import asyncio
import json
import inspect
import traceback
from ..api import api

async def handle_document(
    self, # Umbrella class, can't type hint because of circular imports
    clean_function_arguments_for_api,
    req : Request, 
    file : UploadFile
):
    endpoint = req.scope['path']
    try:
        
        # We have to take the arguments in the header, because the body is the file.
        arguments = json.loads(req.query_params._dict["parameters"])
        file_name = file.filename
        file_ext = file_name.split(".")[-1]
        if endpoint.strip() == "/update_documents":
            target_func, target_func_str = api.update_documents, "update_documents"
        elif file_ext in ["zip", "7z", "rar", "tar"]:
            target_func, target_func_str = api.upload_archive, "upload_archive"
        else:
            target_func, target_func_str = api.upload_document, "upload_document"
        
        true_arguments = clean_function_arguments_for_api({
            **self.default_function_arguments,
            "file": file,
        }, arguments, target_func_str)
        return {"success": True, "result": await target_func(**true_arguments)}

    except Exception as e:
        # if isinstance(e, InFailedSqlTransaction):
        self.database.rollback()
        self.database.flush()
        error_message = str(e)
        stack_trace = traceback.format_exc()
        return_msg = {"success": False, "note": error_message, "trace": stack_trace}
        # print(return_msg)
        print(error_message[:2000])
        return return_msg