# from ray import serve
# from ..typing.config import LocalModel
from QueryLake.typing.config import LocalModel
# from surya.detection import batch_text_detection
# from surya.input.load import load_from_folder, load_from_file
# from surya.layout import batch_layout_detection
# from surya.model.detection.model import load_model, load_processor
# from surya.postprocessing.heatmap import draw_polys_on_image
from typing import List, Union, Dict, Optional, Tuple
# from asyncio import gather
# from marker.convert import convert_single_pdf


from surya.model.detection.model import load_model as load_detection_model, \
                                        load_processor as load_detection_processor
from texify.model.model import load_model as load_texify_model
from texify.model.processor import load_processor as load_texify_processor
from surya.model.recognition.model import load_model as load_recognition_model
from surya.model.recognition.processor import load_processor as load_recognition_processor
from surya.model.ordering.model import load_model as load_order_model
from surya.model.ordering.processor import load_processor as load_order_processor
from surya.model.table_rec.model import load_model as load_table_model
from surya.model.table_rec.processor import load_processor as load_table_processor

from pypdfium2 import PdfDocument

from PIL import Image

from marker.utils import flush_cuda_memory
from marker.tables.table import format_tables
from marker.debug.data import dump_bbox_debug_data, draw_page_debug_images
from marker.layout.layout import surya_layout, annotate_block_types
from marker.layout.order import surya_order, sort_blocks_in_reading_order
from marker.ocr.lang import replace_langs_with_codes, validate_langs
from marker.ocr.detection import surya_detection
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
from marker.settings import settings
from io import BytesIO
import sys

def convert_single_pdf(
        file: BytesIO,
        model_lst: List,
        max_pages: int = None,
        start_page: int = None,
        metadata: Optional[Dict] = None,
        langs: Optional[List[str]] = None,
        batch_multiplier: int = 1,
        ocr_all_pages: bool = False
) -> Tuple[str, Dict[str, Image.Image], Dict]:
    
    ocr_all_pages = ocr_all_pages or settings.OCR_ALL_PAGES

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
    lowres_images = [render_image(doc[pnum], dpi=settings.SURYA_DETECTOR_DPI) for pnum in range(max_len)]

    # Unpack models from list
    (texify_model, layout_model, order_model, detection_model, ocr_model, table_rec_model) = model_lst

    # Identify text lines, layout, reading order
    surya_detection(lowres_images, pages, detection_model, batch_multiplier=batch_multiplier)

    # OCR pages as needed
    pages, ocr_stats = run_ocr(doc, pages, langs, ocr_model, batch_multiplier=batch_multiplier, ocr_all_pages=ocr_all_pages)
    flush_cuda_memory()

    out_meta["ocr_stats"] = ocr_stats
    if len([b for p in pages for b in p.blocks]) == 0:
        print(f"Could not extract any text blocks for {file.name}")
        return "", {}, out_meta

    surya_layout(lowres_images, pages, layout_model, batch_multiplier=batch_multiplier)

    # Find headers and footers
    bad_span_ids = filter_header_footer(pages)
    out_meta["block_stats"] = {"header_footer": len(bad_span_ids)}

    # Add block types from layout
    annotate_block_types(pages)

    # Sort from reading order
    surya_order(lowres_images, pages, order_model, batch_multiplier=batch_multiplier)
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
    table_count = format_tables(pages, doc, file_bytes, detection_model, table_rec_model, ocr_model)
    out_meta["block_stats"]["table"] = table_count

    for page in pages:
        for block in page.blocks:
            block.filter_spans(bad_span_ids)
            block.filter_bad_span_types()

    filtered, eq_stats = replace_equations(
        doc,
        pages,
        texify_model,
        batch_multiplier=batch_multiplier
    )
    flush_cuda_memory()
    out_meta["block_stats"]["equations"] = eq_stats

    # Extract images and figures
    if settings.EXTRACT_IMAGES:
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

    return full_text, doc_images, out_meta


class SuryaMarkerDeployment:
    """
    Ray deployment class for Surya Marker document scanning.
    """
    
    def __init__(self, model_card : List[LocalModel]):
        print("INITIALIZING SURYA DEPLOYMENT")
        # self.model = load_model(checkpoint=model_card.system_path)
        # self.processor = load_processor(checkpoint=model_card.system_path)
        model_list = {m.name : m for m in model_card}
        
        # Since marker HARDCODES the huggingface model, we need to change the model path to the one we have
        # Very annoying, but it is what it is
        texify_settings_module = sys.modules['texify.settings']
        texify_settings_module.settings.MODEL_CHECKPOINT = model_list["Surya Texify"]["system_path"]

        surya_settings_module = sys.modules['surya.settings']
        surya_settings_module.settings.DETECTOR_MODEL_CHECKPOINT = model_list["Surya Detection"]["system_path"]
        surya_settings_module.settings.RECOGNITION_MODEL_CHECKPOINT = model_list["Surya Recognition"]["system_path"]
        surya_settings_module.settings.LAYOUT_MODEL_CHECKPOINT = model_list["Surya Layout"]["system_path"]
        surya_settings_module.settings.ORDER_MODEL_CHECKPOINT = model_list["Surya Ordering"]["system_path"]
        surya_settings_module.settings.TABLE_REC_MODEL_CHECKPOINT = model_list["Surya Table Recognition"]["system_path"]
        
        self.texify_model = load_texify_model(checkpoint=model_list["Surya Texify"].system_path)
        self.texify_processor = load_texify_processor()
        
        self.detection_model = load_detection_model(checkpoint=model_list["Surya Detection"].system_path)
        self.detection_processor = load_detection_processor(checkpoint=model_list["Surya Detection"].system_path)
        
        self.recognition_model = load_recognition_model(checkpoint=model_list["Surya Recognition"].system_path)
        self.recognition_processor = load_recognition_processor()
        
        self.order_model = load_order_model(checkpoint=model_list["Surya Ordering"].system_path)
        self.order_processor = load_order_processor(checkpoint=model_list["Surya Ordering"].system_path)
        
        self.table_rec_model = load_table_model(checkpoint=model_list["Surya Table Recognition"].system_path)
        self.table_rec_processor = load_table_processor()
        
        self.layout_model = load_detection_model(checkpoint=model_list["Surya Layout"].system_path)
        self.layout_processor = load_detection_processor(checkpoint=model_list["Surya Layout"].system_path)
        
        # This fixes some of the code in marker that assumes the processor is attached to the model
        # by manually attaching the processor to the model
        setattr(self.texify_model, "processor", self.texify_processor)
        setattr(self.detection_model, "processor", self.detection_processor)
        setattr(self.layout_model, "processor", self.layout_processor)
        setattr(self.recognition_model, "processor", self.recognition_processor)
        setattr(self.order_model, "processor", self.order_processor)
        setattr(self.table_rec_model, "processor", self.table_rec_processor)
        
        print("DONE INITIALIZING SURYA DEPLOYMENT")

    # @serve.batch(max_batch_size=36, batch_wait_timeout_s=1)
    # async def handle_batch(self, images: List) -> List[dict]:
    #     full_text, doc_images, out_meta = convert_single_pdf(
    #         images,
    #         model_lst,
    #         ocr_all_pages=True
    #     )
    #     batch_layout_detection(images, model, processor, line_predictions)
        
    #     sentence_embeddings = self.model.encode(
    #         inputs,
    #         batch_size=4,
    #         # max_length=8192,
    #         return_sparse=True,
    #         max_length=1024
    #     )
        
    #     inputs_tokenized = self.model.tokenizer(
    #         inputs,
    #         padding=True,
    #         truncation=True,
    #         return_tensors='pt',
    #         max_length=1024,
    #     )["input_ids"].tolist()
        
    #     pad_id = self.model.tokenizer.pad_token_id
    #     token_counts = [sum([1 for x in y if x != pad_id]) for y in inputs_tokenized]
    #     # sparse_vecs = sentence_embeddings['lexical_weights']
        
    #     # print("Sparse Vector:", sparse_vecs)
        
    #     embed_list = sentence_embeddings['dense_vecs'].tolist()
    #     print("Done handling batch of size", len(inputs))
    #     m_2 = time.time()
    #     print("Time taken for batch:", m_2 - m_1)
        
    #     return [{"embedding": embed_list[i], "token_count": token_counts[i]} for i in range(len(inputs))]
    
    async def run(self, 
                  fname: str,
                  max_pages: int = None,
                  start_page: int = None,
                  metadata: Optional[Dict] = None,
                  langs: Optional[List[str]] = None,
                  batch_multiplier: int = 1,
                  ocr_all_pages: bool = False) -> List[List[float]]:
        
        model_lst = (
            self.texify_model, 
            self.layout_model, 
            self.order_model, 
            self.detection_model, 
            self.recognition_model, 
            self.table_rec_model
        )
        # if isinstance(request_dict, dict):
        #     inputs = request_dict["text"]
        # else:
        #     inputs = request_dict
        
        # # Fire them all off at once, but wait for them all to finish before returning
        # result = await gather(*[self.handle_batch(e) for e in inputs])
        # return result
