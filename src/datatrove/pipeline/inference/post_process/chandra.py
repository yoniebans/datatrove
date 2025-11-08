"""Chandra OCR post-processing steps."""

from pathlib import Path
from typing import Iterable

import fitz

from datatrove.data import Document
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.inference.run_inference import InferenceSuccess
from datatrove.utils.logging import logger


class ProcessChandraOutput(PipelineStep):
    """Process Chandra OCR output: save raw HTML, extract images, convert to markdown.

    Combines all Chandra-specific post-processing in a single efficient pass:
    - Saves raw HTML output to disk
    - Extracts and saves images using Chandra's utilities
    - Converts HTML to markdown for document text

    Uses Chandra's official utilities:
    - parse_chunks(): Parses HTML into layout blocks with bounding boxes
    - extract_images(): Crops images from original page using bbox coordinates
    - parse_markdown(): Converts HTML to clean markdown

    Args:
        output_dir: Base directory for saving raw HTML and images
        remove_inference_results: If True, removes inference_results metadata after processing. Default: True.
        include_images: If True, includes image references in markdown. Default: False.
        resize_longest_side_pixels: Resolution to render PDF pages (default: 1280, matches query builder)

    Raises:
        ImportError: If chandra-ocr package is not installed.

    Output structure:
        {output_dir}/{doc_id}/
        ├── page_0.html
        ├── page_0_images/
        │   ├── {hash}_0_img.webp
        │   └── ...
        ├── page_1.html
        └── ...
    """

    def __init__(
        self,
        output_dir: str,
        remove_inference_results: bool = True,
        include_images: bool = False,
        resize_longest_side_pixels: int = 1280,
    ):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.remove_inference_results = remove_inference_results
        self.include_images = include_images
        self.resize_longest_side_pixels = resize_longest_side_pixels

        # Import Chandra utilities
        try:
            from chandra.model import parse_markdown
            from chandra.output import extract_images, parse_chunks
            from PIL import Image

            self.parse_markdown = parse_markdown
            self.parse_chunks = parse_chunks
            self.extract_images = extract_images
            self.Image = Image
        except ImportError:
            raise ImportError(
                "chandra-ocr package required for ProcessChandraOutput. "
                "Install with: pip install chandra-ocr"
            )

    def _render_page_to_pil(self, page: fitz.Page) -> "Image.Image":
        """Render PDF page to PIL Image at specified resolution."""
        # Calculate scale to achieve target resolution
        page_rect = page.rect
        current_longest = max(page_rect.width, page_rect.height)
        scale = self.resize_longest_side_pixels / current_longest

        # Render page as pixmap
        mat = fitz.Matrix(scale, scale)
        pix = page.get_pixmap(matrix=mat, alpha=False)

        # Convert to PIL Image
        img_data = pix.tobytes("png")
        from io import BytesIO

        img = self.Image.open(BytesIO(img_data))
        return img

    def run(self, data: Iterable[Document], rank: int = 0, world_size: int = 1):
        for document in data:
            # Get PDF bytes and inference results
            if not document.media or not document.media[0].media_bytes:
                logger.warning(f"Document {document.id} has no media bytes, skipping")
                self.stat_update("no_media")
                yield document
                continue

            pdf_bytes = document.media[0].media_bytes
            inference_results = document.metadata.get("inference_results", [])

            if not inference_results:
                logger.warning(f"Document {document.id} has no inference results, skipping")
                self.stat_update("no_results")
                yield document
                continue

            # Create document output directory for raw files
            doc_dir = self.output_dir / document.id
            doc_dir.mkdir(parents=True, exist_ok=True)

            # Open PDF
            pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")

            # Process each page - save raw HTML, extract images, convert to markdown
            markdown_pages = []
            page_num = 0

            for result in inference_results:
                if not isinstance(result, InferenceSuccess):
                    logger.warning(f"Document {document.id} page {page_num} failed inference")
                    markdown_pages.append(result.error)
                    self.stat_update("inference_failures")
                    page_num += 1
                    continue

                # Get raw HTML
                html = result.text

                # Save raw HTML
                html_path = doc_dir / f"page_{page_num}.html"
                html_path.write_text(html, encoding="utf-8")
                self.stat_update("html_pages_saved")

                # Render page as PIL Image and extract images
                if page_num < len(pdf_doc):
                    page = pdf_doc.load_page(page_num)
                    page_image = self._render_page_to_pil(page)

                    # Parse chunks and extract images
                    try:
                        chunks = self.parse_chunks(html, page_image)
                        images = self.extract_images(html, chunks, page_image)

                        # Save extracted images
                        if images:
                            images_dir = doc_dir / f"page_{page_num}_images"
                            images_dir.mkdir(exist_ok=True)

                            for img_name, pil_image in images.items():
                                img_path = images_dir / img_name
                                pil_image.save(img_path)
                                self.stat_update("images_saved")

                    except Exception as e:
                        logger.warning(f"Error extracting images for {document.id} page {page_num}: {e}")
                        self.stat_update("image_extraction_errors")

                # Convert HTML to markdown
                try:
                    markdown = self.parse_markdown(html, include_images=self.include_images)
                    markdown_pages.append(markdown)
                except Exception as e:
                    markdown_pages.append(f"[Chandra parsing error: {e}]")
                    self.stat_update("parsing_errors")

                page_num += 1

            pdf_doc.close()

            # Set document text to concatenated markdown
            document.text = "\n\n".join(markdown_pages)

            # Optionally clean up inference_results metadata
            if self.remove_inference_results and "inference_results" in document.metadata:
                del document.metadata["inference_results"]

            self.stat_update("documents_processed")
            yield document
