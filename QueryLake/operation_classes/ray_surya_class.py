from typing import List, Dict, Tuple, Set, Optional
from QueryLake.typing.config import LocalModel
from ray import serve
from PIL import Image
import torch
import sys
import pickle
import ray
from io import BytesIO
from ray import cloudpickle




# Current marker code.
# Using marker-pdf==0.3.9 and surya-ocr==0.6.11

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
import json
import base64
import time

from marker.ocr.detection import batch_text_detection, Page, get_batch_size

def surya_detection(images: list, pages: List[Page], det_model, det_processor, batch_multiplier=1):

    predictions = batch_text_detection(images, det_model, det_processor, batch_size=int(get_batch_size() * batch_multiplier))
    for (page, pred) in zip(pages, predictions):
        page.text_lines = pred

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
    (texify_model, layout_model, order_model, detection_model, det_processor, ocr_model, table_rec_model) = model_lst

    # Identify text lines, layout, reading order
    surya_detection(lowres_images, pages, detection_model, det_processor, batch_multiplier=batch_multiplier)

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

    return full_text, doc_images, out_meta, pages, text_blocks

class MarkerDeployment:
    def __init__(self, model_card: LocalModel):
        print("INITIALIZING MARKER DEPLOYMENT")
        texify_settings_module = sys.modules['texify.settings']
        texify_settings_module.settings.MODEL_CHECKPOINT = model_card.system_path
        
        assert isinstance(model_card.system_path, dict), "Model card system path must be a dictionary for Marker Deployment"
        assert all([
            (key in model_card.system_path) for key in [
                "layout_model",
                "texify_model",
                "recognition_model",
                "table_rec_model",
                "detection_model",
                "ordering_model"
            ]
        ])
        
        marker_module = sys.modules['marker.settings']
        marker_module.settings.TEXIFY_MODEL_NAME = model_card.system_path["texify_model"]
        
        texify_module = sys.modules['texify.settings']
        texify_module.settings.MODEL_CHECKPOINT = model_card.system_path["texify_model"]
        
        surya_settings_module = sys.modules['surya.settings']
        surya_settings_module.settings.DETECTOR_MODEL_CHECKPOINT = model_card.system_path["detection_model"]
        surya_settings_module.settings.RECOGNITION_MODEL_CHECKPOINT = model_card.system_path["recognition_model"]
        surya_settings_module.settings.LAYOUT_MODEL_CHECKPOINT = model_card.system_path["layout_model"]
        surya_settings_module.settings.TABLE_REC_MODEL_CHECKPOINT = model_card.system_path["table_rec_model"]

        model_list = model_card.system_path
        
        texify_settings_module = sys.modules['texify.settings']
        texify_settings_module.settings.MODEL_CHECKPOINT = model_list["texify_model"]

        surya_settings_module = sys.modules['surya.settings']
        surya_settings_module.settings.DETECTOR_MODEL_CHECKPOINT = model_list["detection_model"]
        surya_settings_module.settings.RECOGNITION_MODEL_CHECKPOINT = model_list["recognition_model"]
        surya_settings_module.settings.LAYOUT_MODEL_CHECKPOINT = model_list["layout_model"]
        surya_settings_module.settings.ORDER_MODEL_CHECKPOINT = model_list["ordering_model"]
        surya_settings_module.settings.TABLE_REC_MODEL_CHECKPOINT = model_list["table_rec_model"]
                
        
        self.texify_model = load_texify_model(checkpoint=model_list["texify_model"])
        self.texify_processor = load_texify_processor()
        self.detection_model = load_detection_model(checkpoint=model_list["detection_model"])
        self.detection_processor = load_detection_processor(checkpoint=model_list["detection_model"])

        self.recognition_model = load_recognition_model(checkpoint=model_list["recognition_model"])
        self.recognition_processor = load_recognition_processor()
        self.order_model = load_order_model(checkpoint=model_list["ordering_model"])
        self.order_processor = load_order_processor(checkpoint=model_list["ordering_model"])
        self.table_rec_model = load_table_model(checkpoint=model_list["table_rec_model"])
        self.table_rec_processor = load_table_processor()
        self.layout_model = load_detection_model(checkpoint=model_list["layout_model"])
        self.layout_processor = load_detection_processor(checkpoint=model_list["layout_model"])
        
        setattr(self.texify_model, "processor", self.texify_processor)
        setattr(self.detection_model, "processor", self.detection_processor)
        setattr(self.layout_model, "processor", self.layout_processor)
        setattr(self.recognition_model, "processor", self.recognition_processor)
        setattr(self.order_model, "processor", self.order_processor)
        setattr(self.table_rec_model, "processor", self.table_rec_processor)
        
        self.model_list = (
            self.texify_model, 
            self.layout_model, 
            self.order_model, 
            self.detection_model,
            self.detection_processor,
            self.recognition_model, 
            self.table_rec_model
        )
        
        self.queued_outputs = {}

        print("DONE INITIALIZING MARKER DEPLOYMENT")

    @serve.batch(max_batch_size=8, batch_wait_timeout_s=0.1)
    async def handle_batch(
        self,
        doc: List[bytes]
        # request_id: List[str]
        # pages: List[List[Page]]
        # inputs: List[bytes]
    ):
        results = []
        for i in range(len(doc)):
            doc_local = doc[i]
            # request_id_local = request_id[i]
            
            # self.queued_outputs[request_id_local] = False
            
            doc_bytes = BytesIO(doc_local)
            doc_bytes.name = "test.pdf"
            full_text, doc_images, out_meta, pages, text_blocks = convert_single_pdf(
                doc_bytes,
                self.model_list,
                ocr_all_pages=True,
                batch_multiplier=4,
                langs=["en"]
            )
            results.append((full_text, doc_images, out_meta, pages, text_blocks))
        
        return results

    async def __call__(self, doc: bytes):
        # doc_wrapped = PdfDocument(doc)
        # result = await self.handle_batch(doc_wrapped, pages)
        
        
        # This request queueing is necessary to bypass
        # ray's hardcoded 30s request timeout.
        # Instead, we'll keep bugging it until it's done.
        
        
        # Forcefully wait for 30 seconds to test the timeout conditions
        # for i in range(30):
        #     print("Sleeping for %4d seconds   " % (30-i), end="\r")
        #     time.sleep(1)
        
        # if request_id in self.queued_outputs:
        #     if self.queued_outputs[request_id] == False:
        #         result = "REQUEST_IN_PROGRESS"
        #     else:
        #         print("Finished result ")
        #         result = self.queued_outputs.pop(request_id)
        # elif already_made:
        #     result =  "QUEUE_NOT_FOUND"
        # else:
        #     result = await self.handle_batch(doc)
        
        result = await self.handle_batch(doc)
        
        
        result_buf = BytesIO()
        cloudpickle.dump(ray.put(result), result_buf)
        result_ref_encoded = base64.b64encode(result_buf.getvalue()).decode('ascii')
        
        print("Done with results, returning")
        
        return result_ref_encoded





# Upcoming code for new marker.
# TODO: Waiting for fixes in the markdown package.

# from surya.model.detection.model import load_model as load_detection_model, load_processor as load_detection_processor
# from surya.model.layout.model import load_model as load_layout_model
# from surya.model.layout.processor import load_processor as load_layout_processor
# from texify.model.model import load_model as load_texify_model
# from texify.model.processor import load_processor as load_texify_processor
# from marker.settings import settings
# from surya.model.recognition.model import load_model as load_recognition_model
# from surya.model.recognition.processor import load_processor as load_recognition_processor
# from surya.model.table_rec.model import load_model as load_table_model
# from surya.model.table_rec.processor import load_processor as load_table_processor

# from texify.model.model import GenerateVisionEncoderDecoderModel
# from surya.model.layout.encoderdecoder import SuryaLayoutModel
# from surya.model.detection.model import EfficientViTForSemanticSegmentation
# from surya.model.recognition.encoderdecoder import OCREncoderDecoderModel
# from surya.model.table_rec.encoderdecoder import TableRecEncoderDecoderModel
# from marker.config.parser import ConfigParser
# from marker.config.printer import CustomClickPrinter
# from marker.converters.pdf import PdfConverter, PdfProvider

# from texify.settings import settings

# from pypdfium2 import PdfDocument
# import base64


# import pypdfium2 as pdfium
# from ftfy import fix_text
# from pdftext.extraction import dictionary_output
# from PIL import Image

# from marker.providers.utils import alphanum_ratio
# from marker.providers import BaseProvider, ProviderOutput, ProviderPageLines
# from marker.schema.polygon import PolygonBox
# from marker.schema import BlockTypes
# from marker.schema.registry import get_block_class
# from marker.schema.text.line import Line
# from marker.schema.text.span import Span

# from marker.builders.document import DocumentBuilder
# from marker.builders.layout import LayoutBuilder
# from marker.builders.ocr import OcrBuilder
# from marker.builders.structure import StructureBuilder

# import atexit
# import re


# class MarkerDeployment:
#     def __init__(self, model_card: LocalModel):
#         print("INITIALIZING MARKER DEPLOYMENT")
#         texify_settings_module = sys.modules['texify.settings']
#         texify_settings_module.settings.MODEL_CHECKPOINT = model_card.system_path
        
#         assert isinstance(model_card.system_path, dict), "Model card system path must be a dictionary for Marker Deployment"
#         assert all([
#             (key in model_card.system_path) for key in [
#                 "layout_model",
#                 "texify_model",
#                 "recognition_model",
#                 "table_rec_model",
#                 "detection_model"
#             ]
#         ])
        
#         marker_module = sys.modules['marker.settings']
#         marker_module.settings.TEXIFY_MODEL_NAME = model_card.system_path["texify_model"]
        
#         texify_module = sys.modules['texify.settings']
#         texify_module.settings.MODEL_CHECKPOINT = model_card.system_path["texify_model"]
        
#         surya_settings_module = sys.modules['surya.settings']
#         surya_settings_module.settings.DETECTOR_MODEL_CHECKPOINT = model_card.system_path["detection_model"]
#         surya_settings_module.settings.RECOGNITION_MODEL_CHECKPOINT = model_card.system_path["recognition_model"]
#         surya_settings_module.settings.LAYOUT_MODEL_CHECKPOINT = model_card.system_path["layout_model"]
#         surya_settings_module.settings.TABLE_REC_MODEL_CHECKPOINT = model_card.system_path["table_rec_model"]

        
#         model_layout = load_layout_model(checkpoint=model_card.system_path["layout_model"])
#         model_layout.processor = load_layout_processor()
        
#         model_texify = load_texify_model(checkpoint=model_card.system_path["texify_model"])
#         model_texify.processor = load_texify_processor()
        
#         model_recognition = load_recognition_model(checkpoint=model_card.system_path["recognition_model"])
#         model_recognition.processor = load_recognition_processor()
        
#         model_table_rec = load_table_model(checkpoint=model_card.system_path["table_rec_model"])
#         model_table_rec.processor = load_table_processor()
        
#         model_detection = load_detection_model(checkpoint=model_card.system_path["detection_model"])
#         model_detection.processor = load_detection_processor()
        
#         self.models = {
#             "layout_model": model_layout,
#             "texify_model": model_texify,
#             "recognition_model": model_recognition,
#             "table_rec_model": model_table_rec,
#             "detection_model": model_detection
#         }
        
        
#         self.pdf_converter = PdfConverter(
#             artifact_dict=self.models,
#         )
#         print("DONE INITIALIZING MARKER DEPLOYMENT")

#     @serve.batch(max_batch_size=8, batch_wait_timeout_s=0.1)
#     async def handle_batch(
#         self,
#         doc: List[bytes], 
#         # pages: List[List[Page]]
#         # inputs: List[bytes]
#     ):
#         results = []
#         for i in range(len(doc)):
#             doc_local = doc[i]
            
#             pdf_provider = PdfProvider(doc_local, self.pdf_converter.config)
#             layout_builder = self.pdf_converter.resolve_dependencies(LayoutBuilder)
#             ocr_builder = self.pdf_converter.resolve_dependencies(OcrBuilder)
            
#             # pdf_provider.filepath="/aabbaabb"
            
#             document = DocumentBuilder(self.pdf_converter.config)(pdf_provider, layout_builder, ocr_builder)
#             StructureBuilder(self.pdf_converter.config)(document)

#             for i, processor_cls in enumerate(self.pdf_converter.processor_list):
#                 print("Running processer (%d/%d)" % (i+1, len(self.pdf_converter.processor_list)))
#                 processor = self.pdf_converter.resolve_dependencies(processor_cls)
#                 processor(document)

#             renderer = self.pdf_converter.resolve_dependencies(self.pdf_converter.renderer)
#             result = renderer(document)
            
#             results.append(result)
        
#     #         pages_local = pages[i]
#     #         with torch.no_grad():
#     #             result = replace_equations(doc_local, pages_local, self.model, batch_multiplier=1)
#     #         results.append(result)
#     #     return results
#         return results

#     async def run(self, doc: bytes):
#         # doc_wrapped = PdfDocument(doc)
#         # result = await self.handle_batch(doc_wrapped, pages)
        
#         print("RUNNING MARKER WITH doc=", type(doc))
        
        
        
        
#         result = await self.handle_batch(doc)
#         # result = self.pdf_converter(doc)
        
#         result_buf = BytesIO()
#         cloudpickle.dump(ray.put(result), result_buf)
#         result_ref_encoded = base64.b64encode(result_buf.getvalue()).decode('ascii')
        
#         print("Done with results, returning")
        
#         return result_ref_encoded

from surya.ordering import batch_ordering

class SuryaOrderDeployment:
    def __init__(self, model_card: LocalModel):
        print("INITIALIZING SURYA ORDER DEPLOYMENT")

        surya_settings_module = sys.modules['surya.settings']
        surya_settings_module.settings.ORDER_MODEL_CHECKPOINT = model_card.system_path

        self.model = load_order_model(checkpoint=model_card.system_path)
        self.processor = load_order_processor(checkpoint=model_card.system_path)
        setattr(self.model, "processor", self.processor)
        print("DONE INITIALIZING SURYA ORDER DEPLOYMENT")

    @serve.batch(max_batch_size=8, batch_wait_timeout_s=0.1)
    async def handle_batch(
        self,
        images: List[Image.Image],
        bboxes: List[List[List[float]]]
    ):
        return batch_ordering(images, bboxes, self.model, self.processor, batch_size=8)

    async def run(
        self,
        image: Image.Image,
        bboxes: List[List[float]]
    ) -> Tuple[List[Dict], Dict]:
        result = await self.handle_batch(image, bboxes)
        
        result_buf = BytesIO()
        cloudpickle.dump(ray.put(result), result_buf)
        result_ref_encoded = base64.b64encode(result_buf.getvalue()).decode('ascii')
        
        return result_ref_encoded


from surya.tables import batch_table_recognition

class SuryaTableDeployment:
    def __init__(self, model_card: LocalModel):
        print("INITIALIZING SURYA TABLE DEPLOYMENT")

        surya_settings_module = sys.modules['surya.settings']
        surya_settings_module.settings.TABLE_REC_MODEL_CHECKPOINT = model_card.system_path

        self.model = load_table_model(checkpoint=model_card.system_path)
        self.processor = load_table_processor()
        setattr(self.model, "processor", self.processor)
        print("DONE INITIALIZING SURYA TABLE DEPLOYMENT")

    @serve.batch(max_batch_size=8, batch_wait_timeout_s=0.1)
    async def handle_batch(
        self,
        images: List[Image.Image],
        table_cells: List[List[Dict]]
    ):
        return batch_table_recognition(images, table_cells=table_cells, model=self.model, processor=self.processor, batch_size=8)

    async def run(
        self,
        image: Image.Image,
        table_cells: List[Dict]
    ) -> Tuple[List[Dict], Dict]:
        result = await self.handle_batch(image, table_cells)
        
        result_buf = BytesIO()
        cloudpickle.dump(ray.put(result), result_buf)
        result_ref_encoded = base64.b64encode(result_buf.getvalue()).decode('ascii')
        
        return result_ref_encoded

from surya.layout import batch_layout_detection, TextDetectionResult

class SuryaLayoutDeployment:
    def __init__(self, model_card: LocalModel):
        print("INITIALIZING SURYA LAYOUT DEPLOYMENT")

        surya_settings_module = sys.modules['surya.settings']
        surya_settings_module.settings.LAYOUT_MODEL_CHECKPOINT = model_card.system_path

        self.model = load_detection_model(checkpoint=model_card.system_path)
        self.processor = load_detection_processor(checkpoint=model_card.system_path)
        setattr(self.model, "processor", self.processor)
        print("DONE INITIALIZING SURYA LAYOUT DEPLOYMENT")

    @serve.batch(max_batch_size=8, batch_wait_timeout_s=0.1)
    async def handle_batch(
        self,
        images: List[Image.Image],
        # text_detection_results: List[TextDetectionResult | None]
    ):
        return batch_layout_detection(images, self.model, self.processor, batch_size=8)

    async def run(
        self,
        image: Image.Image,
    ) -> Tuple[List[Dict], Dict]:
        result = await self.handle_batch(image)
        
        result_buf = BytesIO()
        cloudpickle.dump(ray.put(result), result_buf)
        result_ref_encoded = base64.b64encode(result_buf.getvalue()).decode('ascii')
        
        return result_ref_encoded


from marker.ocr.detection import batch_text_detection as batch_text_detection_original

class SuryaDetectionDeployment:
    def __init__(self, model_card: LocalModel):
        print("INITIALIZING SURYA DETECTION DEPLOYMENT")

        surya_settings_module = sys.modules['surya.settings']
        surya_settings_module.settings.DETECTOR_MODEL_CHECKPOINT = model_card.system_path

        self.model = load_detection_model(checkpoint=model_card.system_path)
        self.processor = load_detection_processor(checkpoint=model_card.system_path)
        setattr(self.model, "processor", self.processor)
        print("DONE INITIALIZING SURYA DETECTION DEPLOYMENT")

    @serve.batch(max_batch_size=8, batch_wait_timeout_s=0.1)
    async def handle_batch(
        self,
        images: List[Image.Image],
    ) -> List[TextDetectionResult]:
        return batch_text_detection_original(images, self.model, self.processor, batch_size=8)

    async def run(
        self,
        image: Image.Image,
    ) -> Tuple[List[Dict], Dict]:
        result = await self.handle_batch(image)
        
        result_buf = BytesIO()
        cloudpickle.dump(result, result_buf)
        result_ref_encoded = base64.b64encode(result_buf.getvalue()).decode('ascii')
        
        return result_ref_encoded

from surya.recognition import batch_recognition
class SuryaOCRDeployment:
    def __init__(self, model_card: LocalModel):
        print("INITIALIZING SURYA RECOGNITION DEPLOYMENT")

        surya_settings_module = sys.modules['surya.settings']
        surya_settings_module.settings.RECOGNITION_MODEL_CHECKPOINT = model_card.system_path

        self.model = load_recognition_model(checkpoint=model_card.system_path)
        self.processor = load_recognition_processor()
        setattr(self.model, "processor", self.processor)
        self.langs = replace_langs_with_codes(None)
        print("DONE INITIALIZING SURYA RECOGNITION DEPLOYMENT")

    @serve.batch(max_batch_size=8, batch_wait_timeout_s=0.1)
    async def handle_batch(
        self,
        images: List[Image.Image],
        languages: List[str]
    ):
        # TODO: Maybe handle langs differently here
        return batch_recognition(images, languages, self.model, self.processor, batch_size=8)

    async def run(
        self,
        image: Image.Image,
        languages: str
    ) -> Tuple[List[Dict], Dict]:
        result = await self.handle_batch(image, languages)
        
        result_buf = BytesIO()
        cloudpickle.dump(ray.put(result), result_buf)
        result_ref_encoded = base64.b64encode(result_buf.getvalue()).decode('ascii')
        
        return result_ref_encoded




# Old code
from marker.equations.inference import get_total_texify_tokens, get_latex_batched

class SuryaTexifyDeployment:
    def __init__(self, model_card: LocalModel):
        print("INITIALIZING SURYA TEXIFY DEPLOYMENT")
        texify_settings_module = sys.modules['texify.settings']
        texify_settings_module.settings.MODEL_CHECKPOINT = model_card.system_path
        
        self.model = load_texify_model(checkpoint=model_card.system_path)
        self.processor = load_texify_processor()
        setattr(self.model, "processor", self.processor)
        print("DONE INITIALIZING SURYA TEXIFY DEPLOYMENT")

    @serve.batch(max_batch_size=8, batch_wait_timeout_s=0.1)
    async def handle_batch(
        self,
        images: List[Image.Image],
        token_counts: List[int]
    ):
        results = []
        results = get_latex_batched(images, token_counts, self.model, batch_multiplier=8)
        return results

    async def run(self, image: Image.Image, token_count: int) -> Tuple[List[Dict], Dict]:
        result = await self.handle_batch(image, token_count)
        
        result_buf = BytesIO()
        cloudpickle.dump(ray.put(result), result_buf)
        result_ref_encoded = base64.b64encode(result_buf.getvalue()).decode('ascii')
        
        return result_ref_encoded
    
    async def get_total_texify_tokens(
        self,
        text
    ):
        return get_total_texify_tokens(text, self.processor)