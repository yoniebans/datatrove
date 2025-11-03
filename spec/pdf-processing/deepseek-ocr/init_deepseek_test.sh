#!/bin/bash
set -e

echo "🚀 Initializing DeepSeek-OCR Test Environment..."

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
# Initialize Conda
# ============================================================================
export PATH="$HOME/miniconda3/bin:$PATH"
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"

# ============================================================================
# Create Minimal Test Environment
# ============================================================================
echo "🐍 Creating minimal DeepSeek-OCR test environment..."
conda create -n deepseek-ocr-test python=3.12 -y

echo "🔄 Activating environment..."
conda activate deepseek-ocr-test

# ============================================================================
# Install PyTorch 2.9.0 First (Required for vLLM nightly)
# ============================================================================
echo "📦 Installing PyTorch 2.9.0 (required for vLLM nightly)..."
# PyTorch 2.9.0 with CUDA support from main PyPI
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0

# ============================================================================
# Install vLLM Nightly (use constraints to prevent PyTorch downgrade)
# ============================================================================
echo "📦 Installing vLLM nightly build..."
# Create constraint file to lock PyTorch versions
cat > /tmp/constraints.txt <<EOF
torch==2.9.0
torchvision==0.24.0
torchaudio==2.9.0
EOF

# Install vLLM nightly with constraints
pip install vllm --pre --extra-index-url https://wheels.vllm.ai/nightly -c /tmp/constraints.txt

# ============================================================================
# Install Minimal Dependencies
# ============================================================================
echo "📦 Installing minimal dependencies..."
pip install pymupdf pillow

# Install DataTrove (lightweight - just utils)
cd /workspace/repos/datatrove
pip install -e .

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
echo "  conda activate deepseek-ocr-test"
echo "  cd /workspace/repos/datatrove"
echo "  python spec/pdf-processing/deepseek-ocr/test_deepseek_standalone.py"
echo ""
echo "Installed versions:"
pip list | grep -E "torch|vllm|PIL|pymupdf"
