# Chandra OCR Integration

## Objective
Integrate Chandra OCR (9B parameter vision-language model) for PDF processing with markdown output.

## Architecture

### How Chandra Works
1. **Model Output**: Raw HTML with bbox data (data-bbox, data-label attributes)
2. **Post-Processing**: Chandra's functions convert HTML → markdown/images/chunks
3. **vLLM Server**: Uses OpenAI-compatible API (not Python API like DeepSeek)
4. **Image Encoding**: Base64 PNG embedded in chat messages

### Integration Approach
**Reuse Chandra's parsing functions** (confirmed importable):
```python
from chandra.model import parse_markdown, parse_html, extract_images, parse_chunks
```

## Components

### 1. Server: Use VLLMServer (no custom server needed)
- DataTrove's `VLLMServer` already provides OpenAI-compatible API
- Chandra connects via OpenAI Python client
- No need for custom `ChandraOCRServer`

### 2. Query Builder: `chandra_ocr_query_builder`
**Location**: `src/datatrove/pipeline/inference/query_builders/vision.py`

**Responsibilities**:
- Encode images as base64 PNG
- Build OpenAI chat format with multimodal content
- Use Chandra's `ocr_layout` prompt from their prompts.py
- Format: `{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}, {"type": "text", "text": "prompt"}]}`

**Chandra Prompt** (from `chandra/prompts.py`):
- Type: `ocr_layout`
- Outputs: HTML with bbox coordinates and block labels
- Labels: Caption, Footnote, Equation-Block, List-Group, Page-Header, Table, Text, etc.

### 3. Post-Process Step: `ExtractChandraMarkdown`
**Location**: `src/datatrove/pipeline/inference/post_process/`

**Responsibilities**:
- Import `from chandra.model import parse_markdown`
- Convert raw HTML output → clean markdown
- Optional: Use `extract_images()` if needed (with `--no-images` flag for now)
- Store markdown in document.text

### 4. Dependencies
- Add `chandra-ocr` to DataTrove optional dependencies: `pip install -e ".[chandra]"`
- Already installed in test environment

## Implementation Files
- `spec/pdf-processing/chandra/test_chandra_standalone.py` - Standalone test (uses HF method)
- `spec/pdf-processing/chandra/init_runpod.sh` - RunPod setup
- `src/datatrove/pipeline/inference/query_builders/vision.py` - Add `chandra_ocr_query_builder`
- `src/datatrove/pipeline/inference/post_process/chandra.py` - New post-process step
- `spec/pdf-processing/chandra/chandra_to_hf.py` - Full pipeline example

## Data Flow
```
PDF → Images → InferenceRunner(VLLMServer + chandra_ocr_query_builder)
    → Raw HTML with bbox
    → ExtractChandraMarkdown(parse_markdown)
    → Clean markdown
    → HuggingFaceDatasetWriter
```

## Image Extraction Decision
**Current approach**: Use `--no-images` flag, only extract markdown text
**Reason**: HF dataset upload strategy for images TBD
**Future**: Can enable image extraction when storage strategy is defined

## Expected Results
- Markdown text with layout preservation
- No images extracted (flag disabled)
- Compatible with HuggingFace dataset upload

## Status
- [x] Standalone test (HF method working)
- [x] Init script
- [ ] Query builder implementation
- [ ] Post-process step implementation
- [ ] Full pipeline test
- [ ] Documentation

## Key Findings
- ✅ Can import `chandra.model.parse_markdown` and other parsing functions
- ✅ HF method works on RunPod (tested)
- ✅ Model outputs 119 chars on test (503 error page PDF)
- ⚠️ vLLM method requires server (chandra_vllm uses Docker - not available in container)
- ✅ Standard VLLMServer is sufficient (provides OpenAI API)
