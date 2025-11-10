"""PDF loading utilities from local filesystem for spec examples."""

import json
from pathlib import Path

import fitz

from datatrove.data import Document, Media, MediaType
from datatrove.utils.logging import logger


def load_pdf_documents(data_dir: str):
    """
    Dynamically discover and load all PDFs from a local directory.

    Args:
        data_dir: Path to directory containing PDF files

    Returns:
        List of Document objects with PDF bytes in media field
    """
    data_path = Path(data_dir)

    if not data_path.exists():
        logger.warning(f"Data directory {data_dir} does not exist. Creating it...")
        data_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created {data_dir}. Please add PDF files and run again.")
        return []

    # Load URL mapping if available
    url_mapping = {}
    mapping_path = data_path / "url_mapping.json"
    if mapping_path.exists():
        with open(mapping_path, encoding="utf-8") as f:
            url_mapping = json.load(f)
        logger.info(f"Loaded URL mapping for {len(url_mapping)} files")

    # Find all PDF files
    pdf_files = list(data_path.glob("*.pdf"))

    if not pdf_files:
        logger.warning(f"No PDF files found in {data_dir}")
        logger.info(f"Please add PDF files to {data_dir} and run again.")
        return []

    logger.info(f"Found {len(pdf_files)} PDF files in {data_dir}")

    documents = []
    for pdf_path in pdf_files:
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

        # Extract PDF metadata
        try:
            with fitz.open(pdf_path) as pdf_doc:
                num_pages = len(pdf_doc)
        except Exception as e:
            logger.warning(f"Could not read PDF metadata for {pdf_path.name}: {e}")
            num_pages = None

        file_size = pdf_path.stat().st_size

        # Get source URL from mapping if available
        source_url = url_mapping.get(pdf_path.name)

        doc = Document(
            text="",  # Empty until extracted
            id=pdf_path.stem,
            media=[
                Media(
                    id=pdf_path.stem,
                    type=MediaType.DOCUMENT,
                    media_bytes=pdf_bytes,
                    url=f"file://{pdf_path}",
                    metadata={
                        "filename": pdf_path.name,
                        "num_pages": num_pages,
                        "file_size_bytes": file_size,
                        "content_type": "application/pdf",
                    },
                )
            ],
            metadata={
                "source": str(pdf_path),
                "source_url": source_url,
            }
        )
        documents.append(doc)
        logger.info(f"Loaded: {pdf_path.name} ({num_pages} pages, {file_size:,} bytes)")

    return documents
