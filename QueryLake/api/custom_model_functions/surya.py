from typing import List, Callable, Awaitable, Dict, Any, Union, Literal, Tuple, Optional

import ray.cloudpickle
import ray.serve
from sqlmodel import Session
from io import BytesIO
from ..user_auth import *
from ...database.encryption import aes_decrypt_zip_file
from ..document import get_document_secure
from ..hashing import random_hash
# from chromadb.api import ClientAPI
from ...typing.config import AuthType
from PIL import Image
import ray
from pypdfium2 import PdfDocument
from ray.serve.handle import DeploymentHandle

# from marker.ocr.lang import replace_langs_with_codes, validate_langs
# from marker.cleaners.headers import filter_common_titles
# from marker.cleaners.bullets import replace_bullets
# from marker.postprocessors.markdown import merge_spans, merge_lines, get_full_text
# from marker.cleaners.text import cleanup_text
# from marker.images.save import images_to_dict
# from marker.pdf.images import render_image
# from marker.settings import settings
# from marker.pdf.extract_text import get_text_blocks
# from marker.schema.bbox import rescale_bbox
# from marker.schema.page import Page
# from marker.schema.block import Block, Line, Span

# from marker.ocr.recognition import get_batch_size as get_ocr_batch_size
# from marker.ocr.detection import get_batch_size as get_detector_batch_size
# from marker.tables.table import (
#     get_table_boxes,
#     get_cells,
#     get_batch_size,
#     assign_rows_columns,
#     formatter
# )




from tabled.inference.recognition import (
    run_recognition,
    batch_table_recognition,
    TableResult
)

import base64
import time




async def process_pdf_with_surya(
    database: Session,
    auth: AuthType,
    server_surya_handles: Dict[str, DeploymentHandle],
    file: BytesIO = None,
    document_id: str = None,
    max_pages: int = None,
    start_page: int = None,
    metadata: Optional[Dict] = None,
    langs: Optional[List[str]] = None,
    batch_multiplier: int = 1,
    ocr_all_pages: bool = False
) -> Tuple[str, Dict[str, Image.Image], Dict]:
    """
    Process a single PDF file using Ray Surya deployments.

    Args:
        database: Database session.
        server_surya_handles: Dictionary of Ray Surya deployment handles.
        file: BytesIO object of the PDF file.
        max_pages: Maximum number of pages to process.
        start_page: Page number to start processing from.
        metadata: Metadata dictionary.
        langs: List of languages for OCR.
        batch_multiplier: Batch multiplier for model processing.
        ocr_all_pages: Whether to OCR all pages.

    Returns:
        Tuple containing the full text, dictionary of images, and output metadata.
    """
    _, _ = get_user(database, auth)
    
    assert any([file, document_id]), "Must provide either file or document_id"
    
    # Ensure file is a BytesIO object
    if not file is None:
        assert isinstance(file, BytesIO), "File must be a BytesIO object"
    else:
        fetch_parameters = get_document_secure(
            database=database,
            auth=auth,
            hash_id=document_id,
        )
        
        password = fetch_parameters["password"]

        file = aes_decrypt_zip_file(
            database, 
            password,
            fetch_parameters["hash_id"]
        )
    
    file.seek(0)
    file_bytes = file.getvalue()
    file_name = getattr(file, 'name', 'document.pdf')

    doc_ref = ray.put(file_bytes)
    
    print("Server Surya Handles:", list(server_surya_handles.keys()))
    marker_handle = server_surya_handles['Marker (V1)']
    
    
    request_id, request_queued, request_response = random_hash(), False, "REQUEST_IN_PROGRESS"
    
    # for i in range(30):
    #     print("Sleeping for %4d seconds   " % (30-i), end="\r")
    #     time.sleep(1)
    
    # Get the reference from the async call
    
    while isinstance(request_response, str) and \
        request_response in ["QUEUE_NOT_FOUND", "REQUEST_IN_PROGRESS"]:
        
        try:
            marker_handle_new = marker_handle.handle_options()
            
            marker_results = marker_handle.remote(
                doc=doc_ref,
                request_id=request_id,
                already_made=request_queued,
            )
            
            print("Marker results type:", type(marker_results))
            
            _, pending = ray.wait([marker_results], timeout=5)
            
            print("Pending:", pending)
            
            # Process the result
            marker_results = ray.cloudpickle.load(BytesIO(base64.b64decode(marker_results)))
        except Exception as e:
            print("Error:", e)
            print("Error type:", type(e))
            raise e
        
        request_response = marker_results
        
        if not request_queued:
            request_queued = True
        
        
    
    print("Marker results type:", type(marker_results))
    
    
    
    return marker_results