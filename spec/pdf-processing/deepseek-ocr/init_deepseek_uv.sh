#!/bin/bash
set -e

echo "🚀 Initializing DeepSeek-OCR Test Environment with uv..."

# ============================================================================
# Environment Variables
# ============================================================================
export HF_HOME=/workspace/models
export TRANSFORMERS_CACHE=/workspace/models
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Persist to .bashrc if not already there
grep -qxF 'export HF_HOME=/workspace/models' ~/.bashrc || cat >> ~/.bashrc <<'EOF'
export HF_HOME=/workspace/models
export TRANSFORMERS_CACHE=/workspace/models
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
EOF

# ============================================================================
# Install uv if not already installed
# ============================================================================
if ! command -v uv &> /dev/null; then
    echo "📦 Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

echo "✅ uv version: $(uv --version)"

# ============================================================================
# Create Virtual Environment with uv (following vLLM Recipes guide)
# ============================================================================
VENV_DIR="/workspace/envs/deepseek-ocr-uv"

echo "🐍 Creating uv virtual environment at $VENV_DIR..."
uv venv "$VENV_DIR" --python 3.12

echo "🔄 Activating environment..."
source "$VENV_DIR/bin/activate"

# ============================================================================
# Install PyTorch with CUDA 12.1 (compatible with CUDA 12.4 driver)
# ============================================================================
echo "📦 Installing PyTorch with CUDA 12.1 support..."
echo "    (Pre-installing to control CUDA version before vLLM)"
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# ============================================================================
# Install vLLM Nightly (with --no-deps to prevent PyTorch upgrade)
# ============================================================================
echo "📦 Installing vLLM nightly with DeepSeek-OCR support..."
echo "    (Using --no-deps to prevent CUDA version conflicts)"
uv pip install vllm --no-deps --pre --extra-index-url https://wheels.vllm.ai/nightly --index-strategy unsafe-best-match

echo "📦 Installing vLLM runtime dependencies..."
uv pip install xformers --index-url https://download.pytorch.org/whl/cu121
uv pip install ray psutil numpy packaging transformers pyzmq

# ============================================================================
# Install Minimal Dependencies
# ============================================================================
echo "📦 Installing minimal dependencies..."
uv pip install pymupdf pillow

# Install DataTrove (lightweight - just utils)
cd /workspace/repos/datatrove
uv pip install -e .

# ============================================================================
# Verification
# ============================================================================
echo "✅ Verifying setup..."

# Test CUDA
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'✅ CUDA: {torch.cuda.get_device_name(0)}')"

# Test torch version
python -c "import torch; print(f'✅ PyTorch version: {torch.__version__}')"

# Test vLLM version
python -c "import vllm; print(f'✅ vLLM version: {vllm.__version__}')"

# Test DeepSeek-OCR support
python -c "from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor; print('✅ DeepSeek-OCR NGramPerReqLogitsProcessor available')"

# Test DataTrove logger
python -c "from datatrove.utils.logging import logger; logger.info('✅ DataTrove logger working')"

echo ""
echo "🎉 DeepSeek-OCR test environment ready!"
echo ""
echo "To test:"
echo "  source $VENV_DIR/bin/activate"
echo "  cd /workspace/repos/datatrove"
echo "  python spec/pdf-processing/deepseek-ocr/test_deepseek_standalone.py"
echo ""
echo "Installed versions:"
pip list | grep -E "torch|vllm|PIL|pymupdf"
