#!/usr/bin/env python3
"""
Extract text from Phase 4 JSONL files for easy review.

Creates .txt files with extracted text alongside metadata for manual review.

Usage:
    python spec/phase4/examples/utils/extract_text_for_review.py <base_dir>

Arguments:
    base_dir: Directory containing results (required)
"""

import gzip
import json
import sys
from pathlib import Path

from datatrove.utils.logging import logger


def extract_text_from_jsonl(jsonl_path: Path, output_dir: Path):
    """Extract text and metadata from JSONL to individual .txt files."""

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Processing: {jsonl_path.name}")

    # Open JSONL (handles .gz automatically)
    open_fn = gzip.open if jsonl_path.suffix == ".gz" else open

    with open_fn(jsonl_path, "rt") as f:
        for line in f:
            doc = json.loads(line)

            doc_id = doc.get("id", "unknown")
            text = doc.get("text", "")
            metadata = doc.get("metadata", {})

            # Create text file with metadata header
            output_file = output_dir / f"{doc_id}.txt"

            with open(output_file, "w") as out:
                # Header with metadata
                out.write("=" * 80 + "\n")
                out.write(f"Document ID: {doc_id}\n")
                out.write("=" * 80 + "\n\n")

                # Key metadata
                out.write("METADATA:\n")
                out.write("-" * 80 + "\n")

                # OCR probability and routing
                if "ocr_probability" in metadata:
                    out.write(f"OCR Probability: {metadata['ocr_probability']:.4f}\n")
                if "processing_route" in metadata:
                    out.write(f"Processing Route: {metadata['processing_route']}\n")

                # Additional useful fields
                for key in ["num_pages", "is_form", "garbled_text_ratio", "is_encrypted", "content_length", "source"]:
                    if key in metadata:
                        out.write(f"{key.replace('_', ' ').title()}: {metadata[key]}\n")

                out.write("-" * 80 + "\n\n")

                # Extracted text
                out.write("EXTRACTED TEXT:\n")
                out.write("=" * 80 + "\n\n")
                out.write(text)
                out.write("\n\n")
                out.write("=" * 80 + "\n")
                out.write(f"END OF DOCUMENT: {doc_id}\n")
                out.write("=" * 80 + "\n")

            logger.info(f"Created {doc_id}.txt ({len(text)} chars)")


def main():
    if len(sys.argv) < 2:
        logger.error("Missing required argument")
        logger.info("Usage: python spec/phase4/examples/utils/extract_text_for_review.py <base_dir>")
        logger.info(
            "Example: python spec/phase4/examples/utils/extract_text_for_review.py /Users/you/Downloads/phase4_results"
        )
        sys.exit(1)

    base_dir = Path(sys.argv[1])

    if not base_dir.exists():
        logger.error(f"Directory does not exist: {base_dir}")
        sys.exit(1)

    logger.info("Extracting Text from Phase 4 Results")
    logger.info(f"Base directory: {base_dir.absolute()}")

    # Process text extraction results (Docling - low OCR)
    text_extraction_jsonl = base_dir / "text_extraction" / "00000.jsonl.gz"
    if text_extraction_jsonl.exists():
        logger.info("Text Extraction Path (Docling - Low OCR)")
        extract_text_from_jsonl(text_extraction_jsonl, base_dir / "text_extraction_review")
    else:
        logger.warning(f"Text extraction file not found: {text_extraction_jsonl}")

    # Process OCR extraction results (RolmOCR - high OCR)
    ocr_extraction_jsonl = base_dir / "ocr_extraction" / "00000.jsonl.gz"
    if ocr_extraction_jsonl.exists():
        logger.info("OCR Extraction Path (RolmOCR - High OCR)")
        extract_text_from_jsonl(ocr_extraction_jsonl, base_dir / "ocr_extraction_review")
    else:
        logger.warning(f"OCR extraction file not found: {ocr_extraction_jsonl}")

    # Also extract classified metadata (no extracted text yet)
    classified_jsonl = base_dir / "classified" / "00000.jsonl.gz"
    if classified_jsonl.exists():
        logger.info("Classification Results (Routing Metadata)")
        extract_text_from_jsonl(classified_jsonl, base_dir / "classified_review")
    else:
        logger.warning(f"Classification file not found: {classified_jsonl}")

    logger.info("Text extraction complete")
    logger.info(f"Review files created in: {base_dir}")
    logger.info(f"  - {base_dir}/text_extraction_review/")
    logger.info(f"  - {base_dir}/ocr_extraction_review/")
    logger.info(f"  - {base_dir}/classified_review/")


if __name__ == "__main__":
    main()
