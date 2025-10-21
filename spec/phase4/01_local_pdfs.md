# Example 01: Dynamic PDF Processing from Local Directory

## Objective
Test FinePDFs two-tiered pipeline with user-provided PDFs by dynamically discovering and processing all PDF files from a local data directory.

## Components
- PDFReader: Dynamically discover and load all PDFs from data folder
- PDFRouter: Classify PDFs by OCR probability using XGBoost model
- DoclingExtractor: Extract text from low OCR probability PDFs
- RolmOCR: Extract text from high OCR probability PDFs via inference
- PersistentContextJsonlWriter: Save results with proper context management

## Implementation
**File:** `spec/phase4/examples/01_local_pdfs.py`

## Data Requirements
- Input: User-provided PDFs in `spec/phase4/data/` (excluded from git)
- Output: `spec/phase4/output/01_local_pdfs/`
  - `classified/` - Classified PDFs with routing metadata
  - `text_extraction/` - Docling extraction results (low OCR)
  - `text_extraction_pdfs/` - Original PDFs routed to Docling
  - `ocr_extraction/` - RolmOCR extraction results (high OCR)
  - `ocr_extraction_pdfs/` - Original PDFs routed to RolmOCR
  - `ocr_extraction_pages_png/` - Rendered pages sent to RolmOCR

## Expected Results
- All PDFs in data folder are discovered and processed
- PDFs correctly routed based on XGBoost classification (threshold 0.5)
- Low OCR PDFs extracted via Docling with text content
- High OCR PDFs extracted via RolmOCR with text content
- Original PDFs and rendered pages saved for cross-reference
- Statistics logged for each stage

## Status
- [ ] Implemented
- [ ] Tested
- [ ] Documentation updated

## Notes
- Pattern is reusable - user simply adds PDFs to `spec/phase4/data/` and runs script
- No hardcoded file paths - dynamically discovers all `.pdf` files
- Follows same 3-stage architecture as Phase 3 Example 08
- Uses Phase 3 XGBoost model: `spec/phase3/data/pdf_classifier_real_data.xgb`
