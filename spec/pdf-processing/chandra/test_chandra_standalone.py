#!/usr/bin/env python3
"""
Standalone Chandra OCR Test

Test Chandra OCR with InferenceManager to verify model loads and works.

Components:
- PDF rendering: Convert PDF pages to images using PyMuPDF
- Chandra OCR: InferenceManager with vLLM backend
- Output: Markdown results saved to files

Usage:
    python spec/pdf-processing/chandra/test_chandra_standalone.py
"""

from pathlib import Path

from datatrove.utils._import_utils import check_required_dependencies
from datatrove.utils.logging import logger

# Check dependencies before importing
check_required_dependencies("Chandra OCR test", [("fitz", "pymupdf"), ("PIL", "pillow")])

import fitz
from PIL import Image

# Configuration
DATA_DIR = "spec/pdf-processing/chandra/data"
OUTPUT_DIR = "spec/pdf-processing/chandra/output/test_standalone"


def render_pdf_page_to_image(pdf_path: str, page_num: int = 0, dpi: int = 150) -> Image.Image:
    """Render a PDF page to PIL Image."""
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    doc.close()
    return img


def test_chandra_ocr(pdf_path: str, max_pages: int = 3):
    """Test Chandra OCR on a PDF file."""
    logger.info("Starting Chandra OCR standalone test")

    if not Path(pdf_path).exists():
        logger.error(f"PDF not found: {pdf_path}")
        return

    logger.info(f"PDF: {pdf_path}")

    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    doc.close()

    pages_to_process = min(max_pages, total_pages)
    logger.info(f"Total pages: {total_pages}, processing: {pages_to_process}")

    logger.info("Loading Chandra OCR model")

    try:
        from chandra.model import InferenceManager
        from chandra.model.schema import BatchInputItem

        manager = InferenceManager(method="vllm")
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

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

        batch = [BatchInputItem(image=img, prompt_type="ocr_layout")]

        try:
            result = manager.generate(batch)[0]
            markdown = result.markdown

            logger.info(f"Extracted {len(markdown)} characters")

            output_file = output_dir / f"page_{page_num + 1}.md"
            output_file.write_text(markdown)
            logger.info(f"Saved to {output_file}")

            preview = markdown[:300]
            if len(markdown) > 300:
                preview += "..."
            logger.info(f"Preview: {preview}")

        except Exception as e:
            logger.error(f"Error during OCR: {e}")
            continue

    logger.info("Test completed")


def main():
    """Main test execution."""
    logger.info("Chandra OCR Standalone Test")

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

    test_chandra_ocr(pdf_path, max_pages=3)


if __name__ == "__main__":
    main()
