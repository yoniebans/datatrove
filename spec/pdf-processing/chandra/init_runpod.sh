#!/bin/bash
set -e

echo "🚀 Setting up Chandra OCR Environment"
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
VENV_DIR="/workspace/envs/chandra"

echo "🐍 Creating virtual environment at $VENV_DIR..."
uv venv "$VENV_DIR" --python 3.12

echo "🔄 Activating environment..."
source "$VENV_DIR/bin/activate"

# ============================================================================
# Install Dependencies
# ============================================================================
echo "📦 Installing Chandra OCR..."
uv pip install chandra-ocr

echo "📦 Installing DataTrove..."
cd /workspace/repos/datatrove
uv pip install -e ".[all]"

echo "📦 Installing PDF processing dependencies..."
uv pip install pymupdf

# ============================================================================
# Verification
# ============================================================================
echo "✅ Verifying setup..."

python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'✅ CUDA: {torch.cuda.get_device_name(0)}')"

python -c "from chandra.model import InferenceManager; print('✅ Chandra InferenceManager available')"

python -c "from datatrove.utils.logging import logger; logger.info('✅ DataTrove logger working')"

echo ""
echo "🎉 Chandra OCR environment ready!"
echo ""
echo "To test:"
echo "  source $VENV_DIR/bin/activate"
echo "  cd /workspace/repos/datatrove"
echo "  python spec/pdf-processing/chandra/test_chandra_standalone.py"
echo ""
