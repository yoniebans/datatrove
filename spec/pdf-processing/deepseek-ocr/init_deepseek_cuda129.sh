#!/bin/bash
set -e

echo "🚀 Setting up DeepSeek-OCR Environment (Vanilla Ubuntu + CUDA 12.9)..."
echo "Container: RunPod Base 1.0.2 (Ubuntu 22.04)"
echo ""

# ============================================================================
# Install CUDA 12.9 Toolkit
# ============================================================================
echo "📦 Installing CUDA 12.9 Toolkit..."

CUDA_VERSION="12.9.0"
CUDA_INSTALLER_URL="https://developer.download.nvidia.com/compute/cuda/${CUDA_VERSION}/local_installers/cuda_${CUDA_VERSION}_560.35.05_linux.run"

if [ ! -d "/usr/local/cuda-12.9" ]; then
    echo "Downloading CUDA 12.9 installer..."
    wget -q --show-progress "$CUDA_INSTALLER_URL" -O /tmp/cuda_installer.run

    echo "Installing CUDA 12.9 (toolkit only, no driver)..."
    sh /tmp/cuda_installer.run --silent --toolkit --no-man-page

    rm /tmp/cuda_installer.run
    echo "✅ CUDA 12.9 installed"
else
    echo "✅ CUDA 12.9 already installed"
fi

# ============================================================================
# Environment Variables
# ============================================================================
export PATH=/usr/local/cuda-12.9/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-12.9
export HF_HOME=/workspace/models
export TRANSFORMERS_CACHE=/workspace/models
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Persist to .bashrc
grep -qxF 'export PATH=/usr/local/cuda-12.9/bin:$PATH' ~/.bashrc || cat >> ~/.bashrc <<'EOF'
export PATH=/usr/local/cuda-12.9/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-12.9
export HF_HOME=/workspace/models
export TRANSFORMERS_CACHE=/workspace/models
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
EOF

echo "✅ CUDA environment configured"

# ============================================================================
# Verify Python 3.12 and uv (pre-installed in RunPod base container)
# ============================================================================
echo "✅ Python 3.12: $(python3.12 --version)"
echo "✅ uv: $(uv --version)"

# ============================================================================
# Create Virtual Environment
# ============================================================================
VENV_DIR="/workspace/envs/deepseek-ocr-cuda129"

echo "🐍 Creating virtual environment at $VENV_DIR..."
uv venv "$VENV_DIR" --python 3.12

echo "🔄 Activating environment..."
source "$VENV_DIR/bin/activate"

# ============================================================================
# Install vLLM Nightly (built for CUDA 12.9)
# ============================================================================
echo "📦 Installing vLLM nightly (cu129) with DeepSeek-OCR support..."
uv pip install vllm --pre --extra-index-url https://wheels.vllm.ai/nightly --index-strategy unsafe-best-match

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

nvcc --version | grep "release"

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
echo "Installed versions:"
pip list | grep -E "torch|vllm|PIL|pymupdf"
