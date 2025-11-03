#!/bin/bash
set -e

echo "🚀 Setting up DeepSeek-OCR Environment (Simplified - CUDA 12.9 driver required)"
echo "Container: RunPod Base 1.0.2-cuda1290-ubuntu2204"
echo ""

# ============================================================================
# Environment Variables
# ============================================================================
export HF_HOME=/workspace/models
export TRANSFORMERS_CACHE=/workspace/models
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Persist to .bashrc
grep -qxF 'export HF_HOME=/workspace/models' ~/.bashrc || cat >> ~/.bashrc <<'EOF'
export HF_HOME=/workspace/models
export TRANSFORMERS_CACHE=/workspace/models
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
EOF

echo "✅ Python 3.12: $(python3.12 --version)"
echo "✅ uv: $(uv --version)"

# ============================================================================
# Create Virtual Environment
# ============================================================================
VENV_DIR="/workspace/envs/deepseek-ocr-simple"

echo "🐍 Creating virtual environment at $VENV_DIR..."
uv venv "$VENV_DIR" --python 3.12

echo "🔄 Activating environment..."
source "$VENV_DIR/bin/activate"

# ============================================================================
# Install vLLM Nightly (WITH dependencies - should work with CUDA 12.9 driver)
# ============================================================================
echo "📦 Installing vLLM nightly (cu129) with all dependencies..."
echo "    Using both vLLM nightly and PyTorch cu129 indexes..."
uv pip install 'vllm>=0.11.1rc0' --pre --extra-index-url https://wheels.vllm.ai/nightly --extra-index-url https://download.pytorch.org/whl/cu129 --index-strategy unsafe-best-match

# ============================================================================
# Install Minimal Dependencies
# ============================================================================
echo "📦 Installing minimal dependencies..."
uv pip install pymupdf pillow

# Install DataTrove
cd /workspace/repos/datatrove
uv pip install -e .

# ============================================================================
# Verification
# ============================================================================
echo "✅ Verifying setup..."

python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'✅ CUDA: {torch.cuda.get_device_name(0)}')"

python -c "import torch; print(f'✅ PyTorch version: {torch.__version__}')"

python -c "import vllm; print(f'✅ vLLM version: {vllm.__version__}')"

python -c "from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor; print('✅ DeepSeek-OCR NGramPerReqLogitsProcessor available')"

python -c "from datatrove.utils.logging import logger; logger.info('✅ DataTrove logger working')"

echo ""
echo "🎉 DeepSeek-OCR test environment ready!"
echo ""
echo "To test:"
echo "  source $VENV_DIR/bin/activate"
echo "  cd /workspace/repos/datatrove"
echo "  python spec/pdf-processing/deepseek-ocr/test_deepseek_standalone.py"
echo ""
