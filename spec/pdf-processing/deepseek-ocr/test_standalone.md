# DeepSeek-OCR Standalone Test

## Objective
Verify DeepSeek-OCR works with vLLM nightly build before integrating into DataTrove's inference pipeline.

## Components
- PyTorch 2.9.0: Required for vLLM nightly compatibility
- vLLM nightly: Provides DeepSeek-OCR support (post-v0.11.0)
- NGramPerReqLogitsProcessor: Custom logits processor for DeepSeek-OCR
- Standalone test script: Direct vLLM Python API usage

## Implementation
**Files:**
- Script: `spec/pdf-processing/deepseek-ocr/test_deepseek_standalone.py`
- Environment: `spec/pdf-processing/deepseek-ocr/init_deepseek_test.sh`

## Data Requirements
- Input: PDFs in `spec/pdf-processing/deepseek-ocr/data/`
- Output: `spec/pdf-processing/deepseek-ocr/output/test_standalone/page_*.txt`

## Expected Results
- Model loads successfully with vLLM nightly
- NGramPerReqLogitsProcessor works correctly
- OCR extraction completes for 3 test pages
- Output text files contain extracted content
- No dependency conflicts

## Environment Setup

### Key Requirements
- Python 3.12
- PyTorch 2.9.0 (exact version required for vLLM nightly)
- vLLM nightly build (0.11.1rc6+)
- CUDA 12.1+

### Installation
```bash
# On your RunPod instance
cd /workspace/repos/datatrove
bash spec/pdf-processing/deepseek-ocr/init_deepseek_test.sh
```

### Why a Separate Environment?
The main `datatrove-docling` environment has:
- Docling dependencies
- RolmOCR (lmdeploy)
- Many additional packages
- PyTorch 2.8.0 (incompatible with vLLM nightly)

This causes pip to backtrack and install stable vLLM 0.11.0 instead of nightly.

A minimal test environment isolates DeepSeek-OCR to confirm it works before integration.

## Status
- [ ] Environment created
- [ ] Dependencies installed
- [ ] Model loads successfully
- [ ] Test script runs
- [ ] OCR results verified

## Known Issues

### Issue 1: vLLM Nightly Dependency Conflict
**Problem:** `pip install vllm --pre` finds nightly but downgrades to stable 0.11.0 due to PyTorch version conflict.

**Solution:** Install PyTorch 2.9.0 first, then vLLM nightly.

### Issue 2: DeepSeek-OCR Not in Stable Release
**Problem:** vLLM 0.11.0 (stable) doesn't include DeepSeek-OCR support.

**Solution:** Must use nightly build until v0.11.1 stable is released.

## Notes
- DeepSeek-OCR requires specific initialization parameters:
  - `enable_prefix_caching=False`
  - `mm_processor_cache_gb=0`
  - `logits_processors=[NGramPerReqLogitsProcessor]`
- Sampling parameters need `extra_args` for ngram settings:
  - `ngram_size=30`
  - `window_size=90`
  - `whitelist_token_ids={128821, 128822}` (table tags)
- This test uses direct vLLM Python API, not DataTrove's VLLMServer wrapper
- After confirming this works, we can modify DataTrove's VLLMServer to support it
