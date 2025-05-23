import tempfile
from copy import deepcopy
from itertools import repeat
from typing import List, Optional, Dict, Tuple

import pypdfium2 as pdfium
import io
from concurrent.futures import ThreadPoolExecutor

from surya.ocr import (
    Image, 
    convert_if_not_rgb, 
    slice_polys_from_image, 
    slice_bboxes_from_image, 
    TextLine, 
    OCRResult, 
    batch_recognition
)

from marker.models import setup_recognition_model
from marker.ocr.heuristics import should_ocr_page, no_text_found, detect_bad_ocr
from marker.ocr.lang import langs_to_ids
from marker.pdf.images import render_image
from marker.schema.bbox import rescale_bbox
from marker.schema.page import Page
from marker.schema.block import Block, Line, Span
from marker.settings import settings
from marker.pdf.extract_text import get_text_blocks
import inspect


def get_batch_size():
    if settings.RECOGNITION_BATCH_SIZE is not None:
        return settings.RECOGNITION_BATCH_SIZE
    elif settings.TORCH_DEVICE_MODEL == "cuda":
        return 32
    elif settings.TORCH_DEVICE_MODEL == "mps":
        return 32
    return 32


async def run_recognition(
    images: List[Image.Image], 
    langs: List[List[str] | None], 
    rec_model, 
    rec_processor, 
    bboxes: List[List[List[int]]] = None, 
    polygons: List[List[List[List[int]]]] = None, 
    batch_size=None,
    batch_recognition_override=None,
) -> List[OCRResult]:
    # Polygons need to be in corner format - [[x1, y1], [x2, y2], [x3, y3], [x4, y4]], bboxes in [x1, y1, x2, y2] format
    assert bboxes is not None or polygons is not None
    assert len(images) == len(langs), "You need to pass in one list of languages for each image"

    images = convert_if_not_rgb(images)

    slice_map = []
    all_slices = []
    all_langs = []
    for idx, (image, lang) in enumerate(zip(images, langs)):
        if polygons is not None:
            slices = slice_polys_from_image(image, polygons[idx])
        else:
            slices = slice_bboxes_from_image(image, bboxes[idx])
        slice_map.append(len(slices))
        all_slices.extend(slices)
        all_langs.extend([deepcopy(lang)] * len(slices))

    
    if not batch_recognition_override is None:
        if inspect.iscoroutinefunction(batch_recognition_override):
            rec_predictions, _ = await batch_recognition_override(
                all_slices, all_langs, rec_model, rec_processor, batch_size=batch_size
            )
        else:
            rec_predictions, _ = batch_recognition_override(
                all_slices, all_langs, rec_model, rec_processor, batch_size=batch_size
            )
    else:
        rec_predictions, _ = batch_recognition(all_slices, all_langs, rec_model, rec_processor, batch_size=batch_size)

    predictions_by_image = []
    slice_start = 0
    for idx, (image, lang) in enumerate(zip(images, langs)):
        slice_end = slice_start + slice_map[idx]
        image_lines = rec_predictions[slice_start:slice_end]
        slice_start = slice_end

        text_lines = []
        for i in range(len(image_lines)):
            if polygons is not None:
                poly = polygons[idx][i]
            else:
                bbox = bboxes[idx][i]
                poly = [[bbox[0], bbox[1]], [bbox[2], bbox[1]], [bbox[2], bbox[3]], [bbox[0], bbox[3]]]

            text_lines.append(TextLine(
                text=image_lines[i],
                polygon=poly
            ))

        pred = OCRResult(
            text_lines=text_lines,
            languages=lang,
            image_bbox=[0, 0, image.size[0], image.size[1]]
        )
        predictions_by_image.append(pred)

    return predictions_by_image


async def run_ocr(
    doc, 
    pages: List[Page], 
    langs: List[str], 
    rec_model, 
    batch_multiplier=1, 
    ocr_all_pages=False,
    batch_recognition_override=None,
) -> Tuple[List[Page], Dict]:
    ocr_pages = 0
    ocr_success = 0
    ocr_failed = 0
    no_text = no_text_found(pages)
    ocr_idxs = []
    for pnum, page in enumerate(pages):
        ocr_needed = should_ocr_page(page, no_text, ocr_all_pages=ocr_all_pages)
        if ocr_needed:
            ocr_idxs.append(pnum)
            ocr_pages += 1

    # No pages need OCR
    if ocr_pages == 0:
        return pages, {"ocr_pages": 0, "ocr_failed": 0, "ocr_success": 0, "ocr_engine": "none"}

    ocr_method = settings.OCR_ENGINE
    if ocr_method is None or ocr_method == "None":
        return pages, {"ocr_pages": 0, "ocr_failed": 0, "ocr_success": 0, "ocr_engine": "none"}
    elif ocr_method == "surya":
        new_pages = await surya_recognition(
            doc, ocr_idxs, langs, rec_model, pages, batch_multiplier=batch_multiplier,
            batch_recognition_override=batch_recognition_override,
        )
    elif ocr_method == "ocrmypdf":
        new_pages = tesseract_recognition(doc, ocr_idxs, langs)
    else:
        raise ValueError(f"Unknown OCR method {ocr_method}")

    for orig_idx, page in zip(ocr_idxs, new_pages):
        if detect_bad_ocr(page.prelim_text) or len(page.prelim_text) == 0:
            ocr_failed += 1
        else:
            ocr_success += 1
            pages[orig_idx] = page

    return pages, {"ocr_pages": ocr_pages, "ocr_failed": ocr_failed, "ocr_success": ocr_success, "ocr_engine": ocr_method}


async def surya_recognition(
    doc, page_idxs, langs: List[str], rec_model, pages: List[Page], batch_multiplier=1,
    batch_recognition_override=None,
) -> List[Optional[Page]]:
    # Slice images in higher resolution than detection happened in
    images = [render_image(doc[pnum], dpi=settings.SURYA_OCR_DPI) for pnum in page_idxs]
    box_scale = settings.SURYA_OCR_DPI / settings.SURYA_DETECTOR_DPI

    processor = rec_model.processor if rec_model is not None else None
    selected_pages = [p for i, p in enumerate(pages) if i in page_idxs]

    surya_langs = [langs] * len(page_idxs)
    detection_results = [p.text_lines.bboxes for p in selected_pages]
    polygons = deepcopy([[b.polygon for b in bboxes] for bboxes in detection_results])

    # Scale polygons to get correct image slices
    for j, poly in enumerate(polygons):
        skip_idxs = []
        for z, p in enumerate(poly):
            for i in range(len(p)):
                p[i] = [int(p[i][0] * box_scale), int(p[i][1] * box_scale)]
            x_coords = [p[i][0] for i in range(len(p))]
            y_coords = [p[i][1] for i in range(len(p))]
            bbox = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
            if (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) == 0:
                skip_idxs.append(z)
        if len(skip_idxs) > 0:
            polygons[j] = [p for i, p in enumerate(poly) if i not in skip_idxs]

    
    results = await run_recognition(
        images, surya_langs, rec_model, processor, polygons=polygons, batch_size=int(get_batch_size() * batch_multiplier),
        batch_recognition_override=batch_recognition_override,
    )
    
    new_pages = []
    for idx, (page_idx, result, old_page) in enumerate(zip(page_idxs, results, selected_pages)):
        text_lines = old_page.text_lines
        ocr_results = result.text_lines
        blocks = []
        for i, line in enumerate(ocr_results):
            scaled_bbox = rescale_bbox([0, 0, images[idx].size[0], images[idx].size[1]], old_page.text_lines.image_bbox, line.bbox)
            block = Block(
                bbox=scaled_bbox,
                pnum=page_idx,
                lines=[Line(
                    bbox=scaled_bbox,
                    spans=[Span(
                        text=line.text,
                        bbox=scaled_bbox,
                        span_id=f"{page_idx}_{i}",
                        font="",
                        font_weight=0,
                        font_size=0,
                    )
                    ]
                )]
            )
            blocks.append(block)
        page = Page(
            blocks=blocks,
            pnum=page_idx,
            bbox=old_page.text_lines.image_bbox,
            rotation=0,
            text_lines=text_lines,
            ocr_method="surya"
        )
        new_pages.append(page)
    return new_pages

def tesseract_recognition(doc, page_idxs, langs: List[str]) -> List[Optional[Page]]:
    pdf_pages = generate_single_page_pdfs(doc, page_idxs)
    with ThreadPoolExecutor(max_workers=settings.OCR_PARALLEL_WORKERS) as executor:
        pages = list(executor.map(_tesseract_recognition, pdf_pages, repeat(langs, len(pdf_pages))))

    return pages


def generate_single_page_pdfs(doc, page_idxs) -> List[io.BytesIO]:
    pdf_pages = []
    for page_idx in page_idxs:
        blank_doc = pdfium.PdfDocument.new()
        blank_doc.import_pages(doc, pages=[page_idx])
        assert len(blank_doc) == 1, "Failed to import page"

        in_pdf = io.BytesIO()
        blank_doc.save(in_pdf)
        in_pdf.seek(0)
        pdf_pages.append(in_pdf)
    return pdf_pages


def _tesseract_recognition(in_pdf, langs: List[str]) -> Optional[Page]:
    import ocrmypdf
    out_pdf = io.BytesIO()

    ocrmypdf.ocr(
        in_pdf,
        out_pdf,
        language=langs[0],
        output_type="pdf",
        redo_ocr=None,
        force_ocr=True,
        progress_bar=False,
        optimize=False,
        fast_web_view=1e6,
        skip_big=15,  # skip images larger than 15 megapixels
        tesseract_timeout=settings.TESSERACT_TIMEOUT,
        tesseract_non_ocr_timeout=settings.TESSERACT_TIMEOUT,
    )

    with tempfile.NamedTemporaryFile() as f:
        f.write(out_pdf.getvalue())
        f.seek(0)
        new_doc = pdfium.PdfDocument(f.name)
        blocks, _ = get_text_blocks(new_doc, f.name, max_pages=1)

    page = blocks[0]
    page.ocr_method = "tesseract"
    return page
