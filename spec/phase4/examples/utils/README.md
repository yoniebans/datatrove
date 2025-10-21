# Phase 4 Utilities

Helper scripts for managing files between local and remote servers, and reviewing Phase 4 pipeline results.

## Scripts

### push_inputs.sh
Upload PDFs to remote server for processing.

```bash
# Set environment variables
export REMOTE_HOST="root@<remote-ip>"
export REMOTE_PORT="<port>"

# Upload PDFs
./spec/phase4/examples/utils/push_inputs.sh ~/.ssh/id_ed25519 "/path/to/pdfs/*.pdf" "/remote/path/to/data/"
```

### pull_outputs.sh
Download results from remote server and extract text for review.

```bash
# Set environment variables
export REMOTE_HOST="root@<remote-ip>"
export REMOTE_PORT="<port>"

# Pull results and extract text
./spec/phase4/examples/utils/pull_outputs.sh ~/.ssh/id_ed25519 "/remote/path/to/output" "/local/path/to/results"
```

### extract_text_for_review.py
Extract text from JSONL files for manual review.

```bash
# Extract text from downloaded results
python spec/phase4/examples/utils/extract_text_for_review.py /local/path/to/results
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
