# OCR Hallucination Cleaning

## Objective
Implement post-processing step to detect and clean hallucinations from vision OCR models (DeepSeek-OCR and RolmOCR) on a per-page basis before concatenation.

## Components
- CleanOCRHallucinations: Post-processing step that cleans individual page results
- Hallucination detection: Pattern matching for conversational loops and token repetition
- Statistics tracking: Record cleaning actions and severity

## Implementation
**File:** `src/datatrove/pipeline/inference/post_process.py` (extend existing file)

## Problem Description

Vision OCR models exhibit two types of hallucinations on sparse/minimal content pages:

1. **RolmOCR Conversational Hallucinations**: Generates fake Q&A pairs
   ```
   <|User|>:What is the current printing?
   <|Bot|>:The current printing is 10 9 8 7 6 5 4 3 2 1.
   ```
   Repeats dozens of times on pages with minimal text.

2. **DeepSeek-OCR Token Repetition**: Repeats detected text excessively
   ```
   Ahkate Akhate, Ahkate Akhate, Ahkate Akhate, ...
   ```
   Repeats thousands of times on near-blank pages with watermarks.

## Solution Approach

Clean hallucinations at the **per-page level** before concatenation:

```python
InferenceRunner(
    query_builder=deepseek_ocr_query_builder,
    config=InferenceConfig(...),
    post_process_steps=[
        CleanOCRHallucinations(),  # Clean individual pages
        ExtractInferenceText(),     # Concatenate cleaned pages
        PersistentContextJsonlWriter(...)
    ]
)
```

## Data Requirements
- Input: Documents with `inference_results` metadata (list of InferenceSuccess/InferenceFailure)
- Output: Same documents with cleaned page results

## Expected Results
- Hallucinated pages cleaned or marked
- Statistics showing: pages_cleaned, conversational_hallucinations_removed, repetitions_cleaned
- Document metadata tracks which pages were cleaned and why
- Optionally save original hallucinated text for analysis

## Configuration Parameters

```python
CleanOCRHallucinations(
    remove_conversational_patterns=True,
    max_repetition_ratio=0.3,
    min_repetition_length=10,
    max_repeat_count=5,
    min_page_length_after_cleaning=50,
    save_original_on_cleaning=True
)
```

## Status
- [ ] Implemented
- [ ] Tested on RolmOCR output
- [ ] Tested on DeepSeek-OCR output
- [ ] Documentation updated

## Notes
- Must clean BEFORE ExtractInferenceText concatenates pages
- Should track cleaning statistics per document
- Consider saving badly hallucinated pages to exclusion writer for analysis
- Need to handle edge case where cleaning removes >80% of page content
