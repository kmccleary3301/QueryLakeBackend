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
    
    # time.sleep(40)
    
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
    
    # Get the reference from the async call
    
    marker_results = await marker_handle.remote(
        doc=doc_ref
    )
    
    print("Marker results type:", type(marker_results))
    
    # Decode the result
    marker_results = ray.cloudpickle.load(BytesIO(base64.b64decode(marker_results)))
    
    
    print("Marker results type:", type(marker_results))
    
    return marker_results
    
    return True

from surya.layout import batch_layout_detection
from surya.ordering import batch_ordering
from .surya_dependency_3 import replace_equations as replace_equations_override
from .surya_dependency_2 import format_tables as format_tables_override
from .surya_dependency_1 import run_ocr as run_ocr_override

from marker.utils import flush_cuda_memory
from marker.tables.table import format_tables
from marker.debug.data import dump_bbox_debug_data, draw_page_debug_images
from marker.layout.layout import surya_layout, annotate_block_types
from marker.layout.order import surya_order, sort_blocks_in_reading_order
from marker.ocr.lang import replace_langs_with_codes, validate_langs
from marker.ocr.recognition import run_ocr
from marker.pdf.extract_text import get_text_blocks
from marker.cleaners.headers import filter_header_footer, filter_common_titles
from marker.equations.equations import replace_equations
from marker.pdf.utils import find_filetype
from marker.cleaners.code import identify_code_blocks, indent_blocks
from marker.cleaners.bullets import replace_bullets
from marker.cleaners.headings import split_heading_blocks, infer_heading_levels
from marker.cleaners.fontstyle import find_bold_italic
from marker.postprocessors.markdown import merge_spans, merge_lines, get_full_text
from marker.cleaners.text import cleanup_text
from marker.images.extract import extract_images
from marker.images.save import images_to_dict
from marker.cleaners.toc import compute_toc
from marker.pdf.images import render_image
from marker.ocr.detection import batch_text_detection
from surya.schema import TextDetectionResult
from asyncio import gather

async def process_pdf_with_surya_2(
    database: Session,
    auth: AuthType,
    server_surya_handles: Dict[str, DeploymentHandle],
    file: BytesIO = None,
    document_id: str = None,
	# model_lst: List,
	max_pages: int = None,
	start_page: int = None,
	metadata: Optional[Dict] = None,
	langs: Optional[List[str]] = None,
	batch_multiplier: int = 1,
	ocr_all_pages: bool = False
) -> Tuple[str, Dict[str, Image.Image], Dict]:
    _, _ = get_user(database, auth)
    
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
        file.name = fetch_parameters["file_name"]
    
    file.seek(0)
    file_bytes = file.getvalue()
    file_name = getattr(file, 'name', 'document.pdf')
    file.name = file_name
    
    ocr_all_pages = ocr_all_pages or False

    if metadata:
        langs = metadata.get("languages", langs)

    langs = replace_langs_with_codes(langs)
    validate_langs(langs)

    # Find the filetype
    filetype = file.name.split(".")[-1].lower()
    if filetype != "pdf":
        filetype = "other"
    
    file_bytes = file.getvalue()
    
    # Setup output metadata
    out_meta = {
        "languages": langs,
        "filetype": filetype,
    }

    if filetype == "other": # We can't process this file
        return "", {}, out_meta

    # Get initial text blocks from the pdf
    doc = PdfDocument(file_bytes)
    pages, toc = get_text_blocks(
        doc,
        # fname,
        file_bytes,
        max_pages=max_pages,
        start_page=start_page
    )
    out_meta.update({
        "pdf_toc": toc,
        "pages": len(pages),
    })

    # Trim pages from doc to align with start page
    if start_page:
        for page_idx in range(start_page):
            doc.del_page(0)

    max_len = min(len(pages), len(doc))
    lowres_images = [render_image(doc[pnum], dpi=96) for pnum in range(max_len)]

    # Unpack models from list
    (texify_model, layout_model, order_model, detection_model, det_processor, ocr_model, table_rec_model) = \
        (None, None, None, None, None, None, None)

    # Identify text lines, layout, reading order
    # TODO: model call batch endpoint detection model
    # TODO: DEPENDENCY INJECTION ADDED
    
    
    detection_handle = server_surya_handles['Surya Detection']
    recognition_handle = server_surya_handles['Surya Recognition']
    layout_handle = server_surya_handles['Surya Layout']
    texify_handle = server_surya_handles['Surya Texify']
    order_handle = server_surya_handles['Surya Ordering']
    
    async def batch_order_new(
        images: List[Image.Image],
        bboxes: List[List[List[int]]],
        *args, **kwargs
    ) -> List[List[TextDetectionResult]]:
        result = await gather(*[order_handle.run.remote(
            image=image,
            bboxes=bbox,
        ) for image, bbox in zip(images, bboxes)])
        return [ray.cloudpickle.load(BytesIO(base64.b64decode(e))) for e in result]
    
    async def batch_detection_new(
        images: List[Image.Image],
        *args, **kwargs
    ) -> List[List[TextDetectionResult]]:
        result = await gather(*[detection_handle.run.remote(
            image=image
        ) for image in images])
        return [ray.cloudpickle.load(BytesIO(base64.b64decode(e))) for e in result]
    
    async def batch_recognition_new(
        images: List[Image.Image],
        langs: List[str],
        *args, **kwargs
    ) -> List[List[TextDetectionResult]]:
        assert len(images) == len(langs), f"Mismatched lengths: {len(images)} != {len(langs)}"
        result = await gather(*[recognition_handle.run.remote(
            image=image,
            languages=langs
        ) for image in images])
        return [ray.cloudpickle.load(BytesIO(base64.b64decode(e))) for e in result]
    
    async def batch_texify_token_count(
        texts: List[str],
        *args, **kwargs
    ) -> List[List[TextDetectionResult]]:
        result = await gather(*[texify_handle.get_total_texify_tokens.remote(
            text=text
        ) for text in texts])
        return ray.get([ray.cloudpickle.load(BytesIO(base64.b64decode(e))) for e in result])
    
    
    def generate_new_batching(
        handle_name: str,
    ):
        local_handle = server_surya_handles[handle_name]
        
        async def new_batch_layout_endpoint(
            images: List[Image.Image],
            *args, **kwargs
        ) -> List[List[TextDetectionResult]]:
            result = await gather(*[local_handle.run.remote(
                image=image
            ) for image in images])
            return ray.get([ray.cloudpickle.load(BytesIO(base64.b64decode(e))) for e in result])
        
        return new_batch_layout_endpoint
    
    # TODO: Add internal override
    predictions = await batch_detection_new(lowres_images, detection_model, det_processor, batch_size=8)
    
    # print("Got predictions:", predictions)
    for (page, pred) in zip(pages, predictions):
        page.text_lines = pred
        
    # return True
    

    # OCR pages as needed
    # TODO: model call batch endpoint OCR model
    # TODO: DEPENDENCY INJECTION ADDED
    pages, ocr_stats = await run_ocr_override(
        doc, pages, langs, ocr_model, batch_multiplier=batch_multiplier, ocr_all_pages=ocr_all_pages,
        batch_recognition_override=batch_recognition_new, 	# TODO: Add internal override
    )
    flush_cuda_memory()

    out_meta["ocr_stats"] = ocr_stats
    if len([b for p in pages for b in p.blocks]) == 0:
        print(f"Could not extract any text blocks for {file.name}")
        return "", {}, out_meta

	# TODO: model call batch endpoint layout model
	# TODO: DEPENDENCY INJECTION ADDED
	
    text_detection_results = [p.text_lines for p in pages]
    # TODO: Add internal override
    layout_batcher_new = generate_new_batching("Surya Layout")
    
    layout_results = await layout_batcher_new(
        lowres_images, None, None, detection_results=text_detection_results, 
        batch_size=8
    )
    for page, layout_result in zip(pages, layout_results):
        page.layout = layout_result

    # Find headers and footers
    bad_span_ids = filter_header_footer(pages)
    out_meta["block_stats"] = {"header_footer": len(bad_span_ids)}

    # Add block types from layout
    annotate_block_types(pages)

    # Sort from reading order
    # TODO: model call batch endpoint order model
    # TODO: DEPENDENCY INJECTION ADDED
    bboxes = []
    for page in pages:
        bbox = [b.bbox for b in page.layout.bboxes][:255]
        bboxes.append(bbox)

 
	# TODO: Add Internal Override for batch ordering
    order_results = await batch_order_new(lowres_images, bboxes, None, None, batch_size=8)
    for page, order_result in zip(pages, order_results):
        page.order = order_result
    
    sort_blocks_in_reading_order(pages)

    # Dump debug data if flags are set
    # draw_page_debug_images(fname, pages) # TODO: Use bytes instead of path
    # dump_bbox_debug_data(fname, pages) # TODO: Use bytes instead of path
    draw_page_debug_images(file_bytes, pages)
    dump_bbox_debug_data(file_bytes, pages)

    # Fix code blocks
    code_block_count = identify_code_blocks(pages)
    out_meta["block_stats"]["code"] = code_block_count
    indent_blocks(pages)

    # Fix table blocks
    # table_count = format_tables(pages, doc, fname, detection_model, table_rec_model, ocr_model) # TODO: Use bytes instead of path
    # TODO: model call batch endpoint detection+table rec+ocr model
    # TODO: DEPENDENCY INJECTION ADDED
    table_count = await format_tables_override(
        pages, doc, file_bytes, detection_model, table_rec_model, ocr_model,
        run_table_batch_recognition=generate_new_batching("Surya Table Recognition"), 		# TODO: Add internal override
        run_ocr_batch_recognition=batch_recognition_new,			# TODO: Add internal override
        run_detection_batch_recognition=generate_new_batching("Surya Detection"),	# TODO: Add internal override
    )
    out_meta["block_stats"]["table"] = table_count

    for page in pages:
        for block in page.blocks:
            block.filter_spans(bad_span_ids)
            block.filter_bad_span_types()

    # TODO: model call batch endpoint texify model
    # TODO: DEPENDENCY INJECTION ADDED
    
    filtered, eq_stats = await replace_equations_override(
        doc,
        pages,
        texify_model,
        batch_multiplier=batch_multiplier,
        run_texify_batch_recognition=generate_new_batching("Surya Texify"), 	# TODO: Add internal override
        get_total_texify_tokens=batch_texify_token_count, 		# TODO: Add internal override
    )
    flush_cuda_memory()
    out_meta["block_stats"]["equations"] = eq_stats
	
    # Extract images and figures
    if True:
        extract_images(doc, pages)

    # Split out headers
    split_heading_blocks(pages)
    infer_heading_levels(pages)
    find_bold_italic(pages)

    # Use headers to compute a table of contents
    out_meta["computed_toc"] = compute_toc(pages)
	
    # Copy to avoid changing original data
    merged_lines = merge_spans(filtered)
    text_blocks = merge_lines(merged_lines)
    text_blocks = filter_common_titles(text_blocks)
    full_text = get_full_text(text_blocks)

    # Handle empty blocks being joined
    full_text = cleanup_text(full_text)

    # Replace bullet characters with a -
    full_text = replace_bullets(full_text)

    doc_images = images_to_dict(pages)

    return full_text, doc_images, out_meta, pages, text_blocks