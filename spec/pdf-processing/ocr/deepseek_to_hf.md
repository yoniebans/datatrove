# DeepSeek-OCR to HuggingFace Dataset

## Objective
Process PDFs from a local directory using DeepSeek-OCR vision model and upload results to a private HuggingFace dataset for comparison with RolmOCR.

## Components
- PDF loading: Dynamic discovery from local data directory
- DeepSeek-OCR extraction: Page-by-page OCR using InferenceRunner with vllm
- Query builder: New `deepseek_ocr_query_builder` for model-specific prompt format
- HuggingFace upload: Direct upload to private HF dataset repository

## Implementation
**File:** `spec/pdf-processing/ocr/deepseek_to_hf.py`

## Data Requirements
- Input: PDFs in `spec/pdf-processing/ocr/data/` (same PDFs as RolmOCR for comparison)
- Output:
  - Local: `spec/pdf-processing/ocr/output/deepseek_to_hf/`
  - Remote: HuggingFace dataset (private repository)

## Expected Results
- All PDFs processed with DeepSeek-OCR
- Extracted text saved to HuggingFace dataset in parquet format
- Metadata includes: document ID, page count, source path, model name
- Processing logs available in `spec/pdf-processing/ocr/logs/deepseek_to_hf/`
- Results comparable with RolmOCR output for quality assessment

## Configuration
- Model: `deepseek-ai/DeepSeek-OCR` (3B parameters)
- Server: `vllm`
- Prompt format: `"<image>\nFree OCR."`
- Resolution: Configurable (default: 1024x1024 base scale)
- Output format: Parquet (via HuggingFaceDatasetWriter)

## Status
- [ ] Implemented
- [ ] Tested
- [ ] Documentation updated

## New Components Required
- `deepseek_ocr_query_builder`: Query builder function in `src/datatrove/pipeline/inference/query_builders/vision.py`
  - Handle DeepSeek-specific prompt format
  - Configure resolution/scale parameters
  - Page rendering compatible with model requirements

## Notes
- DeepSeek-OCR is smaller (3B) than RolmOCR but may have different strengths
- Uses vLLM server instead of lmdeploy
- Same input PDFs as RolmOCR pipeline for direct comparison
- HuggingFace token must be set in environment: `HF_TOKEN`
- Check model context length and adjust pages per request if needed
