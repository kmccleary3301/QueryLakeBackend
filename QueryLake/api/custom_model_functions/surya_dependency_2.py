from tqdm import tqdm
from pypdfium2 import PdfDocument
from tabled.assignment import assign_rows_columns
from tabled.formats import formatter
from tabled.inference.detection import merge_tables

from surya.input.pdflines import get_page_text_lines
from tabled.inference.recognition import recognize_tables

from marker.pdf.images import render_image
from marker.schema.bbox import rescale_bbox
from marker.schema.block import Line, Span, Block
from marker.schema.page import Page
from marker.tables.table import get_table_boxes
from typing import List
from marker.ocr.recognition import get_batch_size as get_ocr_batch_size
from marker.ocr.detection import get_batch_size as get_detector_batch_size

from marker.settings import settings

from surya.ocr import run_recognition
from surya.schema import TableResult
from surya.tables import batch_table_recognition
import inspect
from surya.input.pdflines import get_table_blocks


def get_batch_size():
    if settings.TABLE_REC_BATCH_SIZE is not None:
        return settings.TABLE_REC_BATCH_SIZE
    elif settings.TORCH_DEVICE_MODEL == "cuda":
        return 6
    return 6


from surya.detection import batch_text_detection

async def get_cells(
    table_imgs, table_bboxes, image_sizes, text_lines, models, detect_boxes=False, detector_batch_size=settings.DETECTOR_BATCH_SIZE,
    run_detection_batch_recognition = None,
):
    try:
        det_model, det_processor = models
    except:
        det_model = None
        det_processor = None
    table_cells = []
    needs_ocr = []

    to_inference_idxs = []
    for idx, (highres_bbox, text_line, image_size) in enumerate(zip(table_bboxes, text_lines, image_sizes)):
        # The text cells inside each table
        table_blocks = get_table_blocks([highres_bbox], text_line, image_size)[0] if text_line is not None else None

        if text_line is None or detect_boxes or len(table_blocks) == 0:
            to_inference_idxs.append(idx)
            table_cells.append(None)
            needs_ocr.append(True)
        else:
            table_cells.append(table_blocks)
            needs_ocr.append(False)

    # Inference tables that need it
    if len(to_inference_idxs) > 0:
        
        if not run_detection_batch_recognition is None:
            if inspect.iscoroutinefunction(run_detection_batch_recognition):
                det_results = await run_detection_batch_recognition(
                    [table_imgs[i] for i in to_inference_idxs], det_model, det_processor, batch_size=detector_batch_size
                )
            else:
                det_results = run_detection_batch_recognition(
                    [table_imgs[i] for i in to_inference_idxs], det_model, det_processor, batch_size=detector_batch_size
                )
        else:
            det_results = batch_text_detection([table_imgs[i] for i in to_inference_idxs], det_model, det_processor, batch_size=detector_batch_size)
        for idx, det_result in zip(to_inference_idxs, det_results):
            cell_bboxes = [{"bbox": tb.bbox, "text": None} for tb in det_result.bboxes if tb.area > 0]
            table_cells[idx] = cell_bboxes

    return table_cells, needs_ocr

async def recognize_tables(
    table_imgs, table_cells, needs_ocr: List[bool], models, table_rec_batch_size=settings.TABLE_REC_BATCH_SIZE, 
    ocr_batch_size=settings.RECOGNITION_BATCH_SIZE,
    run_table_batch_recognition = None,
    run_ocr_batch_recognition = None,
) -> List[TableResult]:
    try:
        table_rec_model, table_rec_processor, ocr_model, ocr_processor = models
    except:
        table_rec_model, table_rec_processor, ocr_model, ocr_processor = None, None, None, None
    
    if sum(needs_ocr) > 0:
        needs_ocr_idx = [idx for idx, needs in enumerate(needs_ocr) if needs]
        ocr_images = [img for img, needs in zip(table_imgs, needs_ocr) if needs]
        ocr_cells = [[c["bbox"] for c in cells] for cells, needs in zip(table_cells, needs_ocr) if needs]
        ocr_langs = [None] * len(ocr_images)
        
        if not run_ocr_batch_recognition is None:
            if inspect.iscoroutinefunction(run_ocr_batch_recognition):
                ocr_predictions = await run_ocr_batch_recognition(
                    ocr_images, ocr_langs, ocr_model, ocr_processor, bboxes=ocr_cells, batch_size=ocr_batch_size
                )
            else:
                ocr_predictions = run_ocr_batch_recognition(
                    ocr_images, ocr_langs, ocr_model, ocr_processor, bboxes=ocr_cells, batch_size=ocr_batch_size
                )
        else:
            ocr_predictions = run_recognition(ocr_images, ocr_langs, ocr_model, ocr_processor, bboxes=ocr_cells, batch_size=ocr_batch_size)

        # Assign text to correct spot
        for orig_idx, ocr_pred in zip(needs_ocr_idx, ocr_predictions):
            for ocr_line, cell in zip(ocr_pred.text_lines, table_cells[orig_idx]):
                cell["text"] = ocr_line.text

    if not run_table_batch_recognition is None:
        if inspect.iscoroutinefunction(run_table_batch_recognition):
            table_preds = await run_table_batch_recognition(
                table_imgs, table_cells, table_rec_model, table_rec_processor, batch_size=table_rec_batch_size
            )
        else:
            table_preds = run_table_batch_recognition(
                table_imgs, table_cells, table_rec_model, table_rec_processor, batch_size=table_rec_batch_size
            )
    else:
        table_preds = batch_table_recognition(table_imgs, table_cells, table_rec_model, table_rec_processor, batch_size=table_rec_batch_size)
    return table_preds


async def format_tables(
    pages: List[Page], doc: PdfDocument, fname: str, detection_model, table_rec_model, ocr_model,
    run_table_batch_recognition = None,
    run_ocr_batch_recognition = None,
    run_detection_batch_recognition = None,
):  
    try:
        det_models = [detection_model, detection_model.processor]
        rec_models = [table_rec_model, table_rec_model.processor, ocr_model, ocr_model.processor]
    except:
        det_models = [None, None]
        rec_models = [None, None, None, None]

    # Don't look at table cell detection tqdm output
    tqdm.disable = True
    table_imgs, table_boxes, table_counts, table_text_lines, img_sizes = get_table_boxes(pages, doc, fname)
    # TODO: Dependency Injection
    # TODO: DEPENDENCY INJECTION ADDED
    cells, needs_ocr = await get_cells(
        table_imgs, table_boxes, img_sizes, table_text_lines, det_models, detect_boxes=settings.OCR_ALL_PAGES, 
        detector_batch_size=get_detector_batch_size(),
        run_detection_batch_recognition=run_detection_batch_recognition,
    )
    tqdm.disable = False

    # This will redo OCR if OCR is forced, since we need to redetect bounding boxes, etc.
    # TODO: Dependency Injection
    # TODO: DEPENDENCY INJECTION ADDED
    table_rec = await recognize_tables(
        table_imgs, cells, needs_ocr, rec_models, table_rec_batch_size=get_batch_size(), ocr_batch_size=get_ocr_batch_size(),
        run_table_batch_recognition=run_table_batch_recognition,
        run_ocr_batch_recognition=run_ocr_batch_recognition,
    )
    cells = [assign_rows_columns(tr, im_size) for tr, im_size in zip(table_rec, img_sizes)]
    table_md = [formatter("markdown", cell)[0] for cell in cells]

    table_count = 0
    for page_idx, page in enumerate(pages):
        page_table_count = table_counts[page_idx]
        if page_table_count == 0:
            continue
        
        table_insert_points = {}
        blocks_to_remove = set()
        pnum = page.pnum
        highres_size = img_sizes[table_count]
        page_table_boxes = table_boxes[table_count:table_count + page_table_count]

        for table_idx, table_box in enumerate(page_table_boxes):
            lowres_table_box = rescale_bbox([0, 0, highres_size[0], highres_size[1]], page.bbox, table_box)

            for block_idx, block in enumerate(page.blocks):
                intersect_pct = block.intersection_pct(lowres_table_box)
                if intersect_pct > settings.TABLE_INTERSECTION_THRESH and block.block_type == "Table":
                    if table_idx not in table_insert_points:
                        table_insert_points[table_idx] = max(0, block_idx - len(blocks_to_remove)) # Where to insert the new table
                    blocks_to_remove.add(block_idx)

        new_page_blocks = []
        for block_idx, block in enumerate(page.blocks):
            if block_idx in blocks_to_remove:
                continue
            new_page_blocks.append(block)

        for table_idx, table_box in enumerate(page_table_boxes):
            if table_idx not in table_insert_points:
                table_count += 1
                continue

            markdown = table_md[table_count]
            table_block = Block(
                bbox=table_box,
                block_type="Table",
                pnum=pnum,
                lines=[Line(
                    bbox=table_box,
                    spans=[Span(
                        bbox=table_box,
                        span_id=f"{table_idx}_table",
                        font="Table",
                        font_size=0,
                        font_weight=0,
                        block_type="Table",
                        text=markdown
                    )]
                )]
            )
            insert_point = table_insert_points[table_idx]
            insert_point = min(insert_point, len(new_page_blocks))
            new_page_blocks.insert(insert_point, table_block)
            table_count += 1
        page.blocks = new_page_blocks
    return table_count