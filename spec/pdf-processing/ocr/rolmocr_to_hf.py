#!/usr/bin/env python3
"""
RolmOCR to HuggingFace Dataset

Process PDFs from local directory using RolmOCR and upload to HuggingFace dataset.

Components:
- PDF loading: Dynamic discovery from local data directory
- RolmOCR extraction: Page-by-page OCR using InferenceRunner
- HuggingFace upload: Direct upload to private HF dataset

Usage:
    # Set environment variables and place PDFs in spec/pdf-processing/ocr/data/
    export HF_TOKEN=your_token_here
    export HF_DATASET_REPO=your-org/your-dataset-name
    python spec/pdf-processing/ocr/rolmocr_to_hf.py
"""

import os

from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.inference.post_process import ExtractInferenceText
from datatrove.pipeline.inference.query_builders.vision import rolmocr_query_builder
from datatrove.pipeline.inference.run_inference import InferenceConfig, InferenceRunner
from datatrove.pipeline.writers.huggingface import HuggingFaceDatasetWriter
from datatrove.utils.logging import logger

from local_pdf_loader import load_pdf_documents

# Configuration
DATA_DIR = "spec/pdf-processing/ocr/data"
OUTPUT_DIR = "spec/pdf-processing/ocr/output/rolmocr_to_hf"
LOGS_DIR = "spec/pdf-processing/ocr/logs/rolmocr_to_hf"

# OCR Configuration
MAX_PAGES_PER_OCR_REQUEST = 3  # Pages per chunk: leaves ~4K tokens for output in 8K context


def main():
    """Process PDFs with RolmOCR and upload to HuggingFace."""

    logger.info("RolmOCR to HuggingFace Dataset Pipeline")

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

    logger.info(f"Processing {len(documents)} PDFs with RolmOCR")

    # Add OCR metadata to documents
    for doc in documents:
        doc.metadata["ocr_model"] = "Reducto/RolmOCR"
        doc.metadata["max_pages_per_request"] = MAX_PAGES_PER_OCR_REQUEST

    # Pipeline: OCR extraction -> HuggingFace upload
    pipeline = LocalPipelineExecutor(
        pipeline=[
            documents,
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
                        "vision_max_batch_size": 32,
                        "max_pages_per_request": MAX_PAGES_PER_OCR_REQUEST
                    }
                ),
                post_process_steps=[
                    ExtractInferenceText(),
                    HuggingFaceDatasetWriter(
                        dataset=hf_dataset_repo,
                        private=True,
                        local_working_dir=OUTPUT_DIR + "/hf_upload_temp",
                        expand_metadata=False,
                        cleanup=False
                    )
                ]
            )
        ],
        tasks=1,
        logging_dir=LOGS_DIR
    )

    try:
        pipeline.run()
    finally:
        # Explicitly close the HuggingFace writer to ensure files are uploaded
        writer = None
        for step in pipeline.pipeline:
            if isinstance(step, InferenceRunner):
                for post_step in step.post_process_steps:
                    if isinstance(post_step, HuggingFaceDatasetWriter):
                        writer = post_step
                        break
        if writer:
            logger.info("Closing HuggingFace writer and uploading files...")
            try:
                writer.close(rank=0)
            except Exception as e:
                logger.error(f"Error closing writer: {e}")

    logger.info("Pipeline Complete!")
    logger.info(f"Dataset uploaded to: https://huggingface.co/datasets/{hf_dataset_repo}")


if __name__ == "__main__":
    main()
