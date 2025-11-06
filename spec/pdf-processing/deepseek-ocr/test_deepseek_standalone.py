#!/usr/bin/env python3
"""
Standalone DeepSeek-OCR Test

Test DeepSeek-OCR with vLLM directly (without DataTrove) to verify model loads and works.

Components:
- PDF rendering: Convert PDF pages to images using PyMuPDF
- DeepSeek-OCR: Direct vLLM Python API with NGramPerReqLogitsProcessor
- Output: OCR results saved to text files

Usage:
    python spec/pdf-processing/deepseek-ocr/test_deepseek_standalone.py
"""

import sys
from pathlib import Path

from datatrove.utils._import_utils import check_required_dependencies
from datatrove.utils.logging import logger

# Check dependencies before importing
check_required_dependencies("DeepSeek OCR test", [("fitz", "pymupdf"), ("PIL", "pillow"), "vllm"])

import fitz
from PIL import Image
from vllm import LLM, SamplingParams
from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor

# Configuration
DATA_DIR = "spec/pdf-processing/deepseek-ocr/data"
OUTPUT_DIR = "spec/pdf-processing/deepseek-ocr/output/test_standalone"


def render_pdf_page_to_image(pdf_path: str, page_num: int = 0, dpi: int = 150) -> Image.Image:
    """Render a PDF page to PIL Image."""
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    doc.close()
    return img


def test_deepseek_ocr(pdf_path: str, max_pages: int = 3):
    """Test DeepSeek-OCR on a PDF file."""
    logger.info("Starting DeepSeek-OCR standalone test")

    if not Path(pdf_path).exists():
        logger.error(f"PDF not found: {pdf_path}")
        return

    logger.info(f"PDF: {pdf_path}")

    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    doc.close()

    pages_to_process = min(max_pages, total_pages)
    logger.info(f"Total pages: {total_pages}, processing: {pages_to_process}")

    logger.info("Loading DeepSeek-OCR model")

    try:
        llm = LLM(
            model="deepseek-ai/DeepSeek-OCR",
            enable_prefix_caching=False,
            mm_processor_cache_gb=0,
            logits_processors=[NGramPerReqLogitsProcessor],
            trust_remote_code=True,
        )
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=8192,
        extra_args=dict(
            ngram_size=30,
            window_size=90,
            whitelist_token_ids={128821, 128822},
        ),
        skip_special_tokens=False,
    )

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    for page_num in range(pages_to_process):
        logger.info(f"Processing page {page_num + 1}/{pages_to_process}")

        try:
            img = render_pdf_page_to_image(pdf_path, page_num)
            logger.info(f"Rendered page: {img.width}x{img.height}px")
        except Exception as e:
            logger.error(f"Error rendering page: {e}")
            continue

        prompt = "<image>\nFree OCR."
        model_input = [{
            "prompt": prompt,
            "multi_modal_data": {"image": img}
        }]

        try:
            outputs = llm.generate(model_input, sampling_params)
            text = outputs[0].outputs[0].text

            logger.info(f"Extracted {len(text)} characters")

            output_file = output_dir / f"page_{page_num + 1}.txt"
            output_file.write_text(text)
            logger.info(f"Saved to {output_file}")

            preview = text[:300]
            if len(text) > 300:
                preview += "..."
            logger.info(f"Preview: {preview}")

        except Exception as e:
            logger.error(f"Error during OCR: {e}")
            continue

    logger.info("Test completed")


def main():
    """Main test execution."""
    logger.info("DeepSeek-OCR Standalone Test")

    data_dir = Path(DATA_DIR)

    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        logger.info("Please create the directory and add PDF files")
        return

    pdf_files = list(data_dir.glob("*.pdf"))

    if not pdf_files:
        logger.error(f"No PDF files found in {data_dir}")
        logger.info("Please add at least one PDF file to test")
        return

    pdf_path = str(min(pdf_files, key=lambda p: p.stat().st_size))
    logger.info(f"Using smallest PDF for testing: {Path(pdf_path).name}")

    test_deepseek_ocr(pdf_path, max_pages=3)


if __name__ == "__main__":
    main()
