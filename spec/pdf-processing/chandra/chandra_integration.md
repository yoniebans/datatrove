# Chandra OCR Integration

## Objective
Integrate Chandra OCR (9B parameter vision-language model) for PDF processing with markdown output and image extraction.

## Components
- ChandraOCRServer: Interface between DataTrove and Chandra
- chandra_ocr_query_builder: Query builder for Chandra input format
- Standalone test: Verify basic functionality

## Implementation
**Files:**
- `spec/pdf-processing/chandra/test_chandra_standalone.py`
- `spec/pdf-processing/chandra/init_runpod.sh`
- `src/datatrove/pipeline/inference/servers/chandra_ocr_server.py`
- `src/datatrove/pipeline/inference/query_builders/vision.py`

## Data Requirements
- Input: PDFs in `spec/pdf-processing/chandra/data/`
- Output: Markdown + extracted images

## Expected Results
- Markdown text with layout preservation
- Extracted images saved separately
- Image references in markdown

## Status
- [ ] Standalone test
- [ ] Init script
- [ ] Server integration
- [ ] Full pipeline

## Notes
- Uses `chandra-ocr` package with InferenceManager
- Extracts images (diagrams, charts) as separate files
- Supports vLLM and HuggingFace methods
- Image handling needs design decision for HF dataset upload
