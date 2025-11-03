#!/bin/bash
set -e

echo "🚀 Setting up DeepSeek-OCR Environment (Vanilla Ubuntu + CUDA 12.9)..."
echo "Container: RunPod Base 1.0.2 (Ubuntu 22.04)"
echo ""

# ============================================================================
# Install CUDA 12.9 Toolkit (via apt - official NVIDIA method)
# ============================================================================
echo "📦 Installing CUDA 12.9 Toolkit..."

if [ ! -d "/usr/local/cuda-12.9" ]; then
    echo "Downloading CUDA repository packages..."
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
    mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600

    wget -q --show-progress https://developer.download.nvidia.com/compute/cuda/12.9.0/local_installers/cuda-repo-ubuntu2204-12-9-local_12.9.0-575.51.03-1_amd64.deb
    dpkg -i cuda-repo-ubuntu2204-12-9-local_12.9.0-575.51.03-1_amd64.deb
    cp /var/cuda-repo-ubuntu2204-12-9-local/cuda-*-keyring.gpg /usr/share/keyrings/

    echo "Installing CUDA toolkit via apt..."
    apt-get update -qq
    apt-get -y install cuda-toolkit-12-9

    rm cuda-repo-ubuntu2204-12-9-local_12.9.0-575.51.03-1_amd64.deb
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
echo "    Forcing nightly version to avoid uv resolver choosing stable..."
uv pip install 'vllm>=0.11.1rc0' --pre --extra-index-url https://wheels.vllm.ai/nightly --index-strategy unsafe-best-match

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
