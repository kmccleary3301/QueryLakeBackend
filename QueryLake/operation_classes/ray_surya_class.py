from typing import List, Dict, Tuple
from QueryLake.typing.config import LocalModel
from ray import serve
from PIL import Image
import torch
import sys
import pickle
import ray
from io import BytesIO
from ray import cloudpickle
from texify.model.model import load_model as load_texify_model
from texify.model.processor import load_processor as load_texify_processor

from surya.model.detection.model import load_model as load_detection_model, \
                                        load_processor as load_detection_processor

from surya.model.recognition.model import load_model as load_recognition_model
from surya.model.recognition.processor import load_processor as load_recognition_processor

from surya.model.ordering.model import load_model as load_order_model
from surya.model.ordering.processor import load_processor as load_order_processor

from surya.model.table_rec.model import load_model as load_table_model
from surya.model.table_rec.processor import load_processor as load_table_processor

from marker.ocr.detection import batch_text_detection, Page, get_batch_size
from marker.layout.layout import surya_layout, annotate_block_types
from marker.layout.order import surya_order, sort_blocks_in_reading_order
from marker.equations.equations import replace_equations
from marker.ocr.recognition import run_ocr
from marker.tables.table import format_tables
from pypdfium2 import PdfDocument
import base64

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
        doc: List[PdfDocument], 
        pages: List[List[Page]]
    ):
        results = []
        for i in range(len(doc)):
            doc_local = doc[i]
            pages_local = pages[i]
            with torch.no_grad():
                result = replace_equations(doc_local, pages_local, self.model, batch_multiplier=1)
            results.append(result)
        return results

    async def run(self, doc: bytes, pages: List[Page]) -> Tuple[List[Dict], Dict]:
        doc_wrapped = PdfDocument(doc)
        result = await self.handle_batch(doc_wrapped, pages)
        
        result_buf = BytesIO()
        cloudpickle.dump(ray.put(result), result_buf)
        result_ref_encoded = base64.b64encode(result_buf.getvalue()).decode('ascii')
        
        return result_ref_encoded

class SuryaLayoutDeployment:
    def __init__(self, model_card: LocalModel):
        print("INITIALIZING SURYA LAYOUT DEPLOYMENT")
        surya_settings_module = sys.modules['surya.settings']
        surya_settings_module.settings.LAYOUT_MODEL_CHECKPOINT = model_card.system_path
        
        self.model = load_detection_model(checkpoint=model_card.system_path)
        self.processor = load_detection_processor(checkpoint=model_card.system_path)
        setattr(self.model, "processor", self.processor)
        print("DONE INITIALIZING SURYA LAYOUT DEPLOYMENT")

    @serve.batch(max_batch_size=4, batch_wait_timeout_s=0.1)
    async def handle_batch(
        self,
        images: List[List[Image.Image]], 
        pages: List[List[Page]]
    ):
        results = []
        for i in range(len(images)):
            images_local = images[i]
            images_local = [Image.open(BytesIO(image)) for image in images_local]
            pages_local = pages[i]
            with torch.no_grad():
                surya_layout(images_local, pages_local, self.model, batch_multiplier=1)
                annotate_block_types(pages_local)
                results.append({
                    "pages": pages_local,
                    "images": images_local
                })
        return results

    async def run(self, images: List[Image.Image], pages: List[Page]):
        result = await self.handle_batch(images, pages)
        
        result_buf = BytesIO()
        cloudpickle.dump(ray.put(result), result_buf)
        result_ref_encoded = base64.b64encode(result_buf.getvalue()).decode('ascii')
        
        return result_ref_encoded

class SuryaDetectionDeployment:
    def __init__(self, model_card: LocalModel):
        print("INITIALIZING SURYA DETECTION DEPLOYMENT")
        surya_settings_module = sys.modules['surya.settings']
        surya_settings_module.settings.DETECTOR_MODEL_CHECKPOINT = model_card.system_path
        
        self.model = load_detection_model(checkpoint=model_card.system_path)
        self.processor = load_detection_processor(checkpoint=model_card.system_path)
        setattr(self.model, "processor", self.processor)
        print("DONE INITIALIZING SURYA DETECTION DEPLOYMENT")

    @serve.batch(max_batch_size=4, batch_wait_timeout_s=0.1)
    async def handle_batch(self, images : List[List[bytes]], pages : List[List[Page]]):
        
        
        results = []
        
        for i in range(0, len(images)):
            # images_ref, pages_ref, batch_multiplier = args
            batch_multiplier = 1
            # images = ray.get(images[i])
            # pages = ray.get(pages[i])
            images_local = images[i]
            images_local = [Image.open(BytesIO(image)) for image in images_local]
            pages_local = pages[i]
            with torch.no_grad():
                predictions = batch_text_detection(
                    images_local,
                    self.model, 
                    self.processor, 
                    batch_size=int(get_batch_size() * batch_multiplier)
                )
                for page, pred in zip(pages_local, predictions):
                    page.text_lines = pred
                
                
                local_result = {
                    "pages": pages_local,
                    "images": images_local
                }
                
                results.append(local_result)
                
        return results

    async def run(self, images: List[Image.Image], pages: List[Page]):
        print("RUNNING SURYA DETECTION WITH")
        print(f"Images: {len(images)}", type(images[0]))
        print(f"Pages: {len(images)}", type(pages[0]))
        
        result = await self.handle_batch(images, pages)
        
        result_buf = BytesIO()
        cloudpickle.dump(ray.put(result), result_buf)
        result_ref_encoded = base64.b64encode(result_buf.getvalue()).decode('ascii')
        
        return result_ref_encoded

class SuryaRecognitionDeployment:
    def __init__(self, model_card: LocalModel):
        print("INITIALIZING SURYA RECOGNITION DEPLOYMENT")
        surya_settings_module = sys.modules['surya.settings']
        surya_settings_module.settings.RECOGNITION_MODEL_CHECKPOINT = model_card.system_path
        
        self.model = load_recognition_model(checkpoint=model_card.system_path)
        self.processor = load_recognition_processor()
        setattr(self.model, "processor", self.processor)
        print("DONE INITIALIZING SURYA RECOGNITION DEPLOYMENT")

    @serve.batch(max_batch_size=32, batch_wait_timeout_s=0.1)
    async def handle_batch(
        self, 
        doc: List[PdfDocument], 
        pages: List[List[Page]], 
        langs: List[List[str]],
        ocr_all_pages: List[bool]
    ):
        results = []
        for i in range(len(doc)):
            doc_local = doc[i]
            pages_local = pages[i]
            langs_local = langs[i]
            ocr_all_pages_local = ocr_all_pages[i]
            with torch.no_grad():
                result = run_ocr(
                    doc_local, 
                    pages_local, 
                    langs_local, 
                    self.model, 
                    batch_multiplier=1, 
                    ocr_all_pages=ocr_all_pages_local
                )
            results.append(result)
        return results

    async def run(self, doc: bytes, pages: List[Page], langs: List[str], ocr_all_pages: bool = False):
        doc_wrapped = PdfDocument(doc)
        result = await self.handle_batch(doc_wrapped, pages, langs, ocr_all_pages)
        
        result_buf = BytesIO()
        cloudpickle.dump(ray.put(result), result_buf)
        result_ref_encoded = base64.b64encode(result_buf.getvalue()).decode('ascii')
        
        return result_ref_encoded
        

class SuryaOrderingDeployment:
    def __init__(self, model_card: LocalModel):
        print("INITIALIZING SURYA ORDERING DEPLOYMENT")
        surya_settings_module = sys.modules['surya.settings']
        surya_settings_module.settings.ORDER_MODEL_CHECKPOINT = model_card.system_path
        
        self.model = load_order_model(checkpoint=model_card.system_path)
        self.processor = load_order_processor(checkpoint=model_card.system_path)
        setattr(self.model, "processor", self.processor)
        print("DONE INITIALIZING SURYA ORDERING DEPLOYMENT")

    @serve.batch(max_batch_size=4, batch_wait_timeout_s=0.1)
    async def handle_batch(
        self,
        images: List[List[Image.Image]], 
        pages: List[List[Page]]
    ):
        results = []
        for i in range(len(images)):
            images_local = images[i]
            images_local = [Image.open(BytesIO(image)) for image in images_local]
            pages_local = pages[i]
            with torch.no_grad():
                surya_order(images_local, pages_local, self.model, batch_multiplier=1)
                sort_blocks_in_reading_order(pages_local)
            results.append(pages_local)
        return results
        
    async def run(self, images: List[Image.Image], pages: List[Page]):
        result = await self.handle_batch(images, pages)
        
        result_buf = BytesIO()
        cloudpickle.dump(ray.put(result), result_buf)
        result_ref_encoded = base64.b64encode(result_buf.getvalue()).decode('ascii')
        
        return result_ref_encoded

class SuryaTableRecognitionDeployment:
    
    # TODO: this won't work due to the `format_tables` function requiring multiple models
    
    def __init__(self, model_card: LocalModel):
        print("INITIALIZING SURYA TABLE RECOGNITION DEPLOYMENT")
        surya_settings_module = sys.modules['surya.settings']
        surya_settings_module.settings.TABLE_REC_MODEL_CHECKPOINT = model_card.system_path
        
        self.model = load_table_model(checkpoint=model_card.system_path)
        self.processor = load_table_processor()
        setattr(self.model, "processor", self.processor)
        print("DONE INITIALIZING SURYA TABLE RECOGNITION DEPLOYMENT")

    @serve.batch(max_batch_size=4, batch_wait_timeout_s=0.1)
    async def handle_batch(self, pickled_args_list):
        # Unpickle input arguments using ray.cloudpickle
        args_list = [cloudpickle.loads(args) for args in pickled_args_list]
        results = []
        for args in args_list:
            pages_ref, doc_ref, file_bytes_ref, detection_model_ref, recognition_model_ref = args
            pages = ray.get(pages_ref)
            doc = ray.get(doc_ref)
            file_bytes = ray.get(file_bytes_ref)
            detection_model = ray.get(detection_model_ref)
            recognition_model = ray.get(recognition_model_ref)
            with torch.no_grad():
                result = format_tables(pages, doc, file_bytes, detection_model, self.model, recognition_model)
            results.append(result)
        return results

    async def run(self, pages: List[Page], doc: PdfDocument, file_bytes: bytes, detection_model, recognition_model):
        return await self.handle_batch(pages, doc, file_bytes, detection_model, recognition_model)
