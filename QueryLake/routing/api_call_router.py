from fastapi import UploadFile, Request
from fastapi.responses import StreamingResponse, FileResponse, Response
import asyncio
import json
import inspect
import traceback

async def api_general_call(
    self, # Umbrella class, can't type hint because of circular imports
    clean_function_arguments_for_api,
    API_FUNCTION_HELP_DICTIONARY,
    API_FUNCTION_HELP_GUIDE, 
    req: Request, 
    rest_of_path: str, 
    file: UploadFile = None
):
    """
    This is a wrapper around every api function that is allowed. 
    It will call the function with the arguments provided, after filtering them for security.
    """
    
    try:
        print("Calling:", rest_of_path)
        
        if not file is None:
            print("File:", file.filename)
        
        if "parameters" in req.query_params._dict:
            arguments = json.loads(req.query_params._dict["parameters"])
        else:
            # We use ujson because normal `await req.json()` completely stalls on large inputs.
            # print("Awaiting JSON")
            
            arguments = await asyncio.wait_for(req.json(), timeout=10)
        
        
        
        # print("arguments:", arguments)
        route = req.scope['path']
        route_split = route.split("/")
        print("/".join(route_split[:3]))
        if rest_of_path == "help":
            if len(route_split) > 3:
                function_name = route_split[3]
                return {"success": True, "note": API_FUNCTION_HELP_DICTIONARY[function_name]}
            else:
                print(API_FUNCTION_HELP_GUIDE)
                return {"success": True, "note": API_FUNCTION_HELP_GUIDE}
        else:
            function_actual = self.api_function_getter(rest_of_path.split("/")[0])
            true_args = clean_function_arguments_for_api(
                self.default_function_arguments, 
                arguments, 
                function_object=function_actual
            )
            
            if inspect.iscoroutinefunction(function_actual):
                args_get = await function_actual(**true_args)
            else:
                args_get = function_actual(**true_args)
            
            # print("Type of args_get:", type(args_get))
            
            if type(args_get) is StreamingResponse:
                return args_get
            elif type(args_get) is FileResponse:
                return args_get
            elif type(args_get) is Response:
                return args_get
            elif args_get is True:
                return {"success": True}
            return {"success": True, "result": args_get}
    except Exception as e:
        
        self.database.rollback()
        self.database.flush()
        
        error_message = str(e)
        stack_trace = traceback.format_exc()
        return_dict = {"success": False, "error": error_message, "trace": stack_trace}
        print("RETURNING:", json.dumps(return_dict, indent=4))
        return return_dict