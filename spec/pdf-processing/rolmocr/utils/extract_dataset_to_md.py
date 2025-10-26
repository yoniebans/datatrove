#!/usr/bin/env python3
"""
Extract HuggingFace Dataset to Markdown Files

Downloads a HuggingFace dataset and creates markdown files for each row,
using the 'id' column as the filename and 'text' column as the content.

Components:
- HuggingFace datasets: Load dataset from HF Hub
- File writer: Save text to markdown files

Usage:
    export HF_DATASET_REPO=your-org/your-dataset-name
    python spec/pdf-processing/rolmocr/utils/extract_dataset_to_md.py
"""

import os
from pathlib import Path

from datasets import load_dataset

from datatrove.utils.logging import logger

# Configuration
OUTPUT_DIR = "spec/pdf-processing/rolmocr/output/extracted_markdown"


def main():
    """Download dataset and extract text to markdown files."""

    # Check for required environment variable
    dataset_repo = os.getenv("HF_DATASET_REPO")
    if not dataset_repo:
        logger.error("HF_DATASET_REPO environment variable not set")
        logger.info("Usage: export HF_DATASET_REPO=your-org/your-dataset-name")
        return

    logger.info(f"Loading dataset: {dataset_repo}")
    dataset = load_dataset(dataset_repo, split="train")

    logger.info(f"Found {len(dataset)} documents")

    # Create output directory
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    # Extract each document
    for row in dataset:
        doc_id = row.get("id", "unknown")
        text = row.get("text", "")

        # Create markdown filename
        md_filename = f"{doc_id}.md"
        md_path = output_path / md_filename

        # Write text to markdown file
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(text)

        logger.info(f"Created: {md_filename} ({len(text):,} chars)")

    logger.info(f"Extraction complete! Files saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
