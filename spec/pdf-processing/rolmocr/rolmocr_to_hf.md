# RolmOCR to HuggingFace Dataset

## Objective
Process PDFs from a local directory using RolmOCR vision model and upload results to a private HuggingFace dataset for later use.

## Components
- PDF loading: Dynamic discovery from local data directory
- RolmOCR extraction: Page-by-page OCR using InferenceRunner with lmdeploy
- HuggingFace upload: Direct upload to private HF dataset repository

## Implementation
**File:** `spec/pdf-processing/rolmocr/rolmocr_to_hf.py`

## Data Requirements
- Input: PDFs in `spec/pdf-processing/rolmocr/data/`
- Output:
  - Local: `spec/pdf-processing/rolmocr/output/rolmocr_to_hf/`
  - Remote: HuggingFace dataset (private repository)

## Expected Results
- All PDFs processed with RolmOCR
- Extracted text saved to HuggingFace dataset in parquet format
- Metadata includes: document ID, page count, source path
- Processing logs available in `spec/pdf-processing/rolmocr/logs/rolmocr_to_hf/`

## Configuration
- Model: `Reducto/RolmOCR`
- Server: `lmdeploy`
- Context: 8096 tokens
- Pages per request: 3 (prevents truncation)
- Output format: Parquet (via HuggingFaceDatasetWriter)

## Status
- [ ] Implemented
- [ ] Tested
- [ ] Documentation updated

## Notes
- Reuses existing `rolmocr_query_builder` and page chunking logic from Phase 4
- No classification/routing - pure OCR pipeline
- PDFs should be downloaded locally or SCP'd to RunPod before running
- HuggingFace token must be set in environment: `HF_TOKEN`
