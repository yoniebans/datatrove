#!/bin/bash
set -e

echo "🚀 Initializing DataTrove + RolmOCR Environment..."

# Note: SSH is handled by RunPod's /start.sh script (runs before this)
# Your SSH keys from RunPod account settings are automatically configured

# ============================================================================
# Environment Variables
# ============================================================================
export HF_HOME=/workspace/models
export TRANSFORMERS_CACHE=/workspace/models
export LAYOUT_VINO_PATH="/workspace/repos/Docling-sync/models/v2-quant.xml"
export HF_HUB_ENABLE_HF_TRANSFER=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Persist to .bashrc
cat >> ~/.bashrc <<'EOF'
export HF_HOME=/workspace/models
export TRANSFORMERS_CACHE=/workspace/models
export LAYOUT_VINO_PATH="/workspace/repos/Docling-sync/models/v2-quant.xml"
export HF_HUB_ENABLE_HF_TRANSFER=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
EOF

# ============================================================================
# System Dependencies
# ============================================================================
echo "📦 Installing system dependencies..."
apt update && apt install -y git curl wget build-essential

# ============================================================================
# Miniconda
# ============================================================================
if [ ! -d "$HOME/miniconda3" ]; then
    echo "📥 Installing Miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p $HOME/miniconda3
    rm /tmp/miniconda.sh
fi

# Initialize conda for this shell and all future SSH sessions
export PATH="$HOME/miniconda3/bin:$PATH"
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
$HOME/miniconda3/bin/conda init bash

# Accept Conda TOS
echo "📜 Accepting Conda Terms of Service..."
conda config --set channel_priority flexible
yes | conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
yes | conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true

# ============================================================================
# Conda Environment
# ============================================================================
echo "🐍 Creating conda environment..."
conda create -n datatrove-docling python=3.12 -y
conda activate datatrove-docling

# ============================================================================
# Python Dependencies
# ============================================================================
echo "📚 Installing DataTrove..."
cd /workspace/repos/datatrove
pip install -e ".[dev,all]"

echo "📚 Installing Docling-sync..."
cd /workspace/repos/Docling-sync
pip install -e ./docling-core
pip install -e ./docling
pip install -e ./docling-ibm-models

echo "📚 Installing additional dependencies..."
pip install openvino pymupdf zstandard warcio s3fs orjson xgboost

echo "📚 Installing RolmOCR dependencies..."
pip install lmdeploy[all]
pip install qwen-vl-utils

# ============================================================================
# Verification
# ============================================================================
echo "✅ Verifying setup..."

# Test CUDA
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'✅ CUDA available: {torch.cuda.get_device_name(0)}')"

# Test Docling
cd /workspace/repos/datatrove
python -c "
from src.datatrove.pipeline.media.extractors.extractors import DoclingExtractor
extractor = DoclingExtractor(timeout=60)
print('✅ DoclingExtractor initialized successfully!')
"

echo "🎉 Initialization complete!"
echo ""
echo "To start working:"
echo "  conda activate datatrove-docling"
echo "  cd /workspace/repos/datatrove"
echo "  python spec/phase4/examples/01_local_pdfs.py"
