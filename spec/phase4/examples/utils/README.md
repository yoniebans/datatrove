# Phase 4 Utilities

Helper scripts for pulling and reviewing Phase 4 pipeline results.

## Usage

### 1. Pull Results from Remote Server

```bash
# Set environment variables
export REMOTE_HOST="root@your-host"
export REMOTE_PORT="22"                    # Optional, defaults to 22
export REMOTE_SSH_KEY="~/.ssh/id_rsa"      # Optional, defaults to ~/.ssh/id_rsa

# Pull results and extract text
./spec/phase4/examples/utils/pull_results.sh
```

### 2. Extract Text (Manual)

If you already have the results locally:

```bash
python spec/phase4/examples/utils/extract_text_for_review.py
```

## Output

Results are saved to `spec/phase4/data/results/`:

- `text_extraction_review/` - Extracted text from Docling (low OCR PDFs)
- `ocr_extraction_review/` - Extracted text from RolmOCR (high OCR PDFs)
- `classified_review/` - Classification metadata
- `text_extraction_pdfs/` - Saved PDFs processed by Docling
- `ocr_extraction_pdfs/` - Saved PDFs processed by RolmOCR
- `ocr_extraction_pages_png/` - Rendered page images sent to RolmOCR

## Review Files

Each `.txt` file contains:
- Document ID
- Metadata (OCR probability, routing decision, page count, etc.)
- Extracted text

Cross-reference extracted text with original PDFs and rendered PNGs for quality verification.
