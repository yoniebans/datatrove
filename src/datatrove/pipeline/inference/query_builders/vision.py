"""Query builders for vision models (OCR, image understanding, etc.)."""

import fitz
from typing import AsyncGenerator

from datatrove.data import Document
from datatrove.pipeline.inference.run_inference import InferenceRunner


async def rolmocr_query_builder(runner: InferenceRunner, doc: Document) -> AsyncGenerator[dict, None]:
    """Convert PDF document to chunked RolmOCR vision requests.

    Follows FinePDFs specification:
    - Rescale PDFs so longest dimension ≥ 1280px
    - Ensure representation doesn't exceed 2048 image tokens
    - Total context length set to 8096 tokens
    - Processes pages in chunks to handle large multi-page PDFs

    Args:
        runner: InferenceRunner instance (provides config.model_name_or_path)
        doc: Document with Media object containing PDF bytes

    Yields:
        OpenAI-compatible vision request dicts, one per page chunk

    Raises:
        ValueError: If document has no media bytes

    Notes:
        - Use `max_pages_per_request` in model_kwargs to set chunk size
        - Default chunk size processes all pages (may cause OOM/truncation with large PDFs)
        - Results from all chunks are automatically concatenated by ExtractInferenceText
    """
    from datatrove.pipeline.inference.utils.page_rendering import render_page_to_base64png_pymupdf

    # Get PDF bytes from Media object
    if not doc.media or not doc.media[0].media_bytes:
        raise ValueError(f"Document {doc.id} has no media bytes")

    pdf_bytes = doc.media[0].media_bytes
    pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    total_pages = len(pdf_doc)

    # Check if max_pages_per_request is configured (for chunking)
    pages_per_chunk = runner.config.model_kwargs.get('max_pages_per_request', total_pages)

    # Process pages in chunks
    for chunk_start in range(0, total_pages, pages_per_chunk):
        chunk_end = min(chunk_start + pages_per_chunk, total_pages)
        page_images = []

        # Render pages for this chunk
        for page_num in range(chunk_start, chunk_end):
            page = pdf_doc.load_page(page_num)

            # Use FinePDFs specification resolution
            base64_image = render_page_to_base64png_pymupdf(
                page,
                resize_longest_side_pixels=1280,  # FinePDFs spec
                max_visual_tokens=2048  # FinePDFs spec
            )

            page_images.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_image}"}
            })

        # Yield request for this chunk
        yield {
            "model": runner.config.model_name_or_path,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract all text from this PDF document using OCR. Return only the extracted text."},
                        *page_images
                    ]
                }
            ],
            "max_tokens": 4096,  # Leave room for input in 8096 total context
            "temperature": 0.0
        }

    pdf_doc.close()
