#!/usr/bin/env python3
"""
Chandra OCR to HuggingFace Dataset

Process PDFs from local directory using Chandra OCR and upload to HuggingFace dataset.

Components:
- PDF loading: Dynamic discovery from local data directory
- Chandra OCR extraction: Page-by-page OCR using InferenceRunner with vLLM
- Markdown conversion: Convert HTML output to markdown
- HuggingFace upload: Direct upload to HF dataset

Usage:
    # Set environment variables and place PDFs in spec/pdf-processing/chandra/data/
    export HF_TOKEN=your_token_here
    export HF_DATASET_REPO=your-org/your-dataset-name
    python spec/pdf-processing/chandra/chandra_to_hf.py
"""

import os

from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.inference.post_process import ExtractChandraMarkdown
from datatrove.pipeline.inference.query_builders.vision import chandra_ocr_query_builder
from datatrove.pipeline.inference.run_inference import InferenceConfig, InferenceRunner
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers.huggingface import HuggingFaceDatasetWriter
from datatrove.pipeline.writers.jsonl import PersistentContextJsonlWriter
from datatrove.utils.logging import logger

from local_pdf_loader import load_pdf_documents

# Configuration
DATA_DIR = "spec/pdf-processing/chandra/data"
OUTPUT_DIR = "spec/pdf-processing/chandra/output/chandra_to_hf"
LOGS_DIR = "spec/pdf-processing/chandra/logs/chandra_to_hf"
JSONL_OUTPUT = OUTPUT_DIR + "/ocr_results"

# OCR Configuration
MAX_PAGES_PER_OCR_REQUEST = 1  # Chandra: 1 page per request for stability


def main():
    """Process PDFs with Chandra OCR and upload to HuggingFace."""

    logger.info("Chandra OCR to HuggingFace Dataset Pipeline")

    # Check for required environment variable
    hf_dataset_repo = os.getenv("HF_DATASET_REPO")
    if not hf_dataset_repo:
        logger.error("HF_DATASET_REPO environment variable not set")
        logger.info("Usage: export HF_DATASET_REPO=your-org/your-dataset-name")
        return

    logger.info(f"Target HuggingFace dataset: {hf_dataset_repo}")

    # Load PDFs dynamically
    documents = load_pdf_documents(DATA_DIR)

    if not documents:
        logger.error("No PDFs to process. Exiting.")
        return

    logger.info(f"Processing {len(documents)} PDFs with Chandra OCR")

    # Add OCR metadata to documents
    for doc in documents:
        doc.metadata["ocr_model"] = "chandra/chandra-v1-9b"
        doc.metadata["max_pages_per_request"] = MAX_PAGES_PER_OCR_REQUEST

    # Stage 1: OCR extraction -> JSONL
    logger.info("Stage 1: Running OCR extraction and saving to JSONL")
    stage1_ocr = LocalPipelineExecutor(
        pipeline=[
            documents,
            InferenceRunner(
                query_builder=chandra_ocr_query_builder,
                config=InferenceConfig(
                    server_type="vllm",
                    model_name_or_path="chandra/chandra-v1-9b",
                    model_max_context=8192,
                    max_concurrent_requests=4,
                    max_concurrent_tasks=1,
                ),
                post_process_steps=[
                    ExtractChandraMarkdown(
                        remove_inference_results=True,
                        include_images=False
                    ),
                    PersistentContextJsonlWriter(JSONL_OUTPUT, save_media_bytes=False)
                ]
            ),
        ],
        tasks=1,
        logging_dir=LOGS_DIR + "/ocr"
    )

    try:
        stage1_ocr.run()
    finally:
        # Explicitly close the writer to ensure gzip file is properly finalized
        writer = None
        for step in stage1_ocr.pipeline:
            if isinstance(step, InferenceRunner):
                for post_step in step.post_process_steps:
                    if isinstance(post_step, PersistentContextJsonlWriter):
                        writer = post_step
                        break
        if writer and writer._context_entered:
            logger.info("Closing JSONL writer context...")
            writer.__exit__(None, None, None)

    # Stage 2: Read JSONL and upload to HuggingFace
    logger.info("Stage 2: Uploading OCR results to HuggingFace")
    stage2_upload = LocalPipelineExecutor(
        pipeline=[
            JsonlReader(JSONL_OUTPUT),
            HuggingFaceDatasetWriter(
                dataset=hf_dataset_repo,
                private=False,
                local_working_dir=OUTPUT_DIR + "/hf_upload_temp",
                expand_metadata=False,
                cleanup=False
            )
        ],
        tasks=1,
        logging_dir=LOGS_DIR + "/upload",
        depends=stage1_ocr
    )

    stage2_upload.run()

    logger.info("Pipeline Complete!")
    logger.info(f"Dataset uploaded to: https://huggingface.co/datasets/{hf_dataset_repo}")


if __name__ == "__main__":
    main()
