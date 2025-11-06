"""Post-processing steps for InferenceRunner results."""

from typing import Iterable

from datatrove.data import Document
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.inference.run_inference import InferenceSuccess


class ExtractInferenceText(PipelineStep):
    """Extract text from InferenceRunner results.

    InferenceRunner stores results in document.metadata["inference_results"] as a list of
    InferenceSuccess/InferenceFailure objects. This step extracts the text from successful
    results and sets it as the document's main text field.

    For vision models (e.g., RolmOCR), each page generates one InferenceSuccess result.
    This step concatenates all page texts with newlines.

    Args:
        remove_inference_results: If True, removes the inference_results metadata field
            after extraction to save memory. Default: True.
    """

    def __init__(self, remove_inference_results: bool = True):
        super().__init__()
        self.remove_inference_results = remove_inference_results

    def run(self, data: Iterable[Document], rank: int = 0, world_size: int = 1):
        for document in data:
            # Extract text from inference results
            inference_results = document.metadata.get("inference_results", [])

            # Concatenate successful results, include error messages for failures
            document.text = "\n".join([
                x.text if isinstance(x, InferenceSuccess) else x.error
                for x in inference_results
            ])

            # Optionally clean up inference_results metadata
            if self.remove_inference_results and "inference_results" in document.metadata:
                del document.metadata["inference_results"]

            self.stat_update("documents_processed")
            yield document


class ExtractChandraMarkdown(PipelineStep):
    """Extract and convert Chandra OCR HTML output to markdown.

    Chandra OCR outputs HTML with bbox layout information. This step uses Chandra's
    parse_markdown() function to convert the HTML to clean markdown format.

    InferenceRunner stores results in document.metadata["inference_results"] as a list of
    InferenceSuccess/InferenceFailure objects. This step processes the HTML output and
    converts it to markdown.

    Args:
        remove_inference_results: If True, removes the inference_results metadata field
            after extraction to save memory. Default: True.
        include_images: If True, enables image extraction from Chandra output.
            Default: False (images disabled for now).

    Raises:
        ImportError: If chandra-ocr package is not installed.
    """

    def __init__(self, remove_inference_results: bool = True, include_images: bool = False):
        super().__init__()
        self.remove_inference_results = remove_inference_results
        self.include_images = include_images

        # Import Chandra's parsing function
        try:
            from chandra.model import parse_markdown
            self.parse_markdown = parse_markdown
        except ImportError:
            raise ImportError(
                "chandra-ocr package required for ExtractChandraMarkdown. "
                "Install with: pip install chandra-ocr"
            )

    def run(self, data: Iterable[Document], rank: int = 0, world_size: int = 1):
        for document in data:
            # Extract text from inference results
            inference_results = document.metadata.get("inference_results", [])

            markdown_pages = []

            for result in inference_results:
                if isinstance(result, InferenceSuccess):
                    # Convert Chandra's HTML output to markdown
                    try:
                        markdown = self.parse_markdown(
                            result.text,
                            include_images=self.include_images
                        )
                        markdown_pages.append(markdown)
                    except Exception as e:
                        # If parsing fails, include error message
                        markdown_pages.append(f"[Chandra parsing error: {e}]")
                        self.stat_update("parsing_errors")
                else:
                    # Include error messages for failed inferences
                    markdown_pages.append(result.error)
                    self.stat_update("inference_errors")

            # Concatenate all pages
            document.text = "\n\n".join(markdown_pages)

            # Optionally clean up inference_results metadata
            if self.remove_inference_results and "inference_results" in document.metadata:
                del document.metadata["inference_results"]

            self.stat_update("documents_processed")
            yield document
