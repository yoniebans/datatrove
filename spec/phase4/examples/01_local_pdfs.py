#!/usr/bin/env python3
"""
Example 01: Dynamic PDF Processing from Local Directory

Dynamically process all PDFs from local data folder using two-tiered extraction.

Components:
- PDFReader: Dynamically discover and load all PDFs from data folder
- PDFRouter: Classify PDFs by OCR probability using XGBoost model
- DoclingExtractor: Extract text from low OCR probability PDFs
- RolmOCR: Extract text from high OCR probability PDFs via inference
- PersistentContextJsonlWriter: Save results with proper context management

Usage:
    # Place your PDFs in spec/phase4/data/ then run:
    python spec/phase4/examples/01_local_pdfs.py
"""

import base64
from pathlib import Path
from typing import Iterable

import fitz

from datatrove.data import Document, Media, MediaType
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.filters.lambda_filter import LambdaFilter
from datatrove.pipeline.filters.pdf_router import PDFRouter
from datatrove.pipeline.inference.post_process import ExtractInferenceText
from datatrove.pipeline.inference.query_builders.vision import rolmocr_query_builder
from datatrove.pipeline.inference.run_inference import InferenceConfig, InferenceRunner
from datatrove.pipeline.inference.utils.page_rendering import render_page_to_base64png_pymupdf
from datatrove.pipeline.media.extractors.extractors import DoclingExtractor
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers.jsonl import JsonlWriter, PersistentContextJsonlWriter
from datatrove.utils.logging import logger

# Configuration
DATA_DIR = "spec/phase4/data"
MODEL_PATH = "spec/phase3/data/pdf_classifier_real_data.xgb"
OUTPUT_DIR = "spec/phase4/output/01_local_pdfs"
LOGS_DIR = "spec/phase4/logs/01_local_pdfs"


# ============================================================================
# Helper Classes for Saving PDFs/PNGs
# ============================================================================


class SavePDFsToDisk(PipelineStep):
    """Save PDF bytes from Media objects to disk as .pdf files."""

    def __init__(self, output_dir: str):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, data: Iterable[Document], rank: int = 0, world_size: int = 1):
        for document in data:
            # Save PDF to disk if media bytes exist
            if document.media and document.media[0].media_bytes:
                pdf_path = self.output_dir / f"{document.id}.pdf"
                with open(pdf_path, "wb") as f:
                    f.write(document.media[0].media_bytes)
                self.stat_update("pdfs_saved")
            yield document


class SaveOCRPagesAsPNG(PipelineStep):
    """Save rendered PDF pages as PNG images (as sent to RolmOCR)."""

    def __init__(self, output_dir: str):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, data: Iterable[Document], rank: int = 0, world_size: int = 1):
        for document in data:
            if document.media and document.media[0].media_bytes:
                pdf_bytes = document.media[0].media_bytes
                pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")

                # Render pages (match RolmOCR processing)
                max_pages = min(1, len(pdf_doc))  # Same as rolmocr_query_builder
                for page_num in range(max_pages):
                    page = pdf_doc.load_page(page_num)

                    # Use same rendering as RolmOCR
                    base64_image = render_page_to_base64png_pymupdf(
                        page,
                        resize_longest_side_pixels=1280,
                        max_visual_tokens=2048
                    )

                    # Save PNG
                    png_path = self.output_dir / f"{document.id}_page{page_num + 1:03d}.png"
                    with open(png_path, "wb") as f:
                        f.write(base64.b64decode(base64_image))

                    self.stat_update("pages_saved")

                pdf_doc.close()
            yield document


# ============================================================================
# Load Local PDFs Dynamically
# ============================================================================

def load_pdf_documents():
    """Dynamically discover and load all PDFs from data directory."""
    data_path = Path(DATA_DIR)

    if not data_path.exists():
        logger.warning(f"Data directory {DATA_DIR} does not exist. Creating it...")
        data_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created {DATA_DIR}. Please add PDF files and run again.")
        return []

    # Find all PDF files
    pdf_files = list(data_path.glob("*.pdf"))

    if not pdf_files:
        logger.warning(f"No PDF files found in {DATA_DIR}")
        logger.info(f"Please add PDF files to {DATA_DIR} and run again.")
        return []

    logger.info(f"Found {len(pdf_files)} PDF files in {DATA_DIR}")

    documents = []
    for pdf_path in pdf_files:
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

        doc = Document(
            text="",  # Empty until extracted
            id=pdf_path.stem,
            media=[
                Media(
                    id=pdf_path.stem,
                    type=MediaType.DOCUMENT,
                    media_bytes=pdf_bytes,
                    url=f"file://{pdf_path}",
                )
            ],
            metadata={"source": str(pdf_path)}
        )
        documents.append(doc)
        logger.info(f"Loaded: {pdf_path.name}")

    return documents


# ============================================================================
# Pipeline
# ============================================================================

def main():
    """Process all PDFs from data directory through FinePDFs pipeline."""

    logger.info("Phase 4 Example 01: Dynamic PDF Processing")

    # Load PDFs dynamically
    documents = load_pdf_documents()

    if not documents:
        logger.error("No PDFs to process. Exiting.")
        return

    logger.info(f"Processing {len(documents)} PDFs")

    # ========================================================================
    # Stage 1: Classification
    # ========================================================================
    logger.info("Stage 1: PDF Classification and Routing")

    stage1_classification = LocalPipelineExecutor(
        pipeline=[
            documents,
            PDFRouter(
                model_path=MODEL_PATH,
                threshold=0.5
            ),
            JsonlWriter(OUTPUT_DIR + "/classified", save_media_bytes=True),
        ],
        tasks=1,
        logging_dir=LOGS_DIR + "/classification"
    )

    stage1_classification.run()

    # ========================================================================
    # Stage 2: Text Extraction (Low OCR)
    # ========================================================================
    logger.info("Stage 2: Text Extraction (Low OCR Probability)")

    stage2_text_extraction = LocalPipelineExecutor(
        pipeline=[
            JsonlReader(OUTPUT_DIR + "/classified"),
            LambdaFilter(
                filter_function=lambda doc: doc.metadata.get("processing_route") == "text_extraction"
            ),
            SavePDFsToDisk(OUTPUT_DIR + "/text_extraction_pdfs"),
            DoclingExtractor(timeout=1200),
            JsonlWriter(OUTPUT_DIR + "/text_extraction"),
        ],
        tasks=1,
        logging_dir=LOGS_DIR + "/text_extraction",
        depends=stage1_classification
    )

    stage2_text_extraction.run()

    # ========================================================================
    # Stage 3: OCR Extraction (High OCR)
    # ========================================================================
    logger.info("Stage 3: OCR Extraction (High OCR Probability)")

    stage3_ocr_extraction = LocalPipelineExecutor(
        pipeline=[
            JsonlReader(OUTPUT_DIR + "/classified"),
            LambdaFilter(
                filter_function=lambda doc: doc.metadata.get("processing_route") == "ocr_extraction"
            ),
            InferenceRunner(
                query_builder=rolmocr_query_builder,
                config=InferenceConfig(
                    server_type="lmdeploy",
                    model_name_or_path="Reducto/RolmOCR",
                    model_max_context=8096,
                    max_concurrent_requests=1,
                    max_concurrent_tasks=1,
                    model_kwargs={
                        "chat_template": "internlm",
                        "vision_max_batch_size": 128
                    }
                ),
                post_process_steps=[
                    ExtractInferenceText(),
                    SavePDFsToDisk(OUTPUT_DIR + "/ocr_extraction_pdfs"),
                    SaveOCRPagesAsPNG(OUTPUT_DIR + "/ocr_extraction_pages_png"),
                    PersistentContextJsonlWriter(OUTPUT_DIR + "/ocr_extraction")
                ]
            ),
        ],
        tasks=1,
        logging_dir=LOGS_DIR + "/ocr_extraction",
        depends=stage1_classification
    )

    try:
        stage3_ocr_extraction.run()
    finally:
        # Explicitly close the writer to ensure gzip file is properly finalized
        writer = None
        for step in stage3_ocr_extraction.pipeline:
            if isinstance(step, InferenceRunner):
                for post_step in step.post_process_steps:
                    if isinstance(post_step, PersistentContextJsonlWriter):
                        writer = post_step
                        break
        if writer and writer._context_entered:
            logger.info("Closing OCR writer context...")
            writer.__exit__(None, None, None)

    logger.info("Pipeline Complete!")
    logger.info(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
