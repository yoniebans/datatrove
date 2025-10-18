# RunPod Persistent Volume Setup Guide
## DataTrove + Docling + RolmOCR Pipeline

This guide documents how to set up a persistent RunPod environment for the FinePDFs pipeline with minimal setup time on new instances.

---

## Overview

**Goal:** Separate persistent data (repos, models) from compute resources (GPU instances) so you can:
- Spin up/down instances without losing work
- Switch GPU types without re-downloading models
- Reduce setup time from 30+ minutes to 5 minutes

**Architecture:**
- **Persistent Volume**: Repos, HuggingFace models, test data
- **Ephemeral Instance**: Conda environment, Python packages (hardware-specific)
- **Initialization Script**: Automates environment setup on new instances

---

## 1. Create Persistent Volume

### 1.1 In RunPod Dashboard

1. Navigate to **Storage** → **Volumes**
2. Click **Create Volume**
3. Configure:
   - **Name**: `datatrove-workspace`
   - **Size**: 100GB (50GB for models + 50GB for repos/data)
   - **Region**: Choose based on GPU availability

### 1.2 Mount Volume to Instance

When creating a pod:
1. Select your GPU (recommend: RTX PRO 6000 96GB)
2. Under **Volume Disk**: Select `datatrove-workspace`
3. Set **Volume Mount Path**: `/workspace`
4. Set **Container Disk**: 50GB (enough for OS + conda env)

---

## 2. Initial Volume Setup (One-Time)

SSH into your instance and set up the persistent directory structure:

```bash
# Create directory structure on volume
mkdir -p /workspace/repos
mkdir -p /workspace/data
mkdir -p /workspace/models

# Clone repositories to volume
cd /workspace/repos
git clone https://github.com/yoniebans/datatrove.git
git clone https://github.com/yoniebans/Docling-sync.git

# Configure DataTrove
cd datatrove
git checkout feat/varied_local_pdf_testing

# Configure Docling-sync
cd ../Docling-sync
git checkout bug/fix_compilation_issues

# Set up HuggingFace cache on volume
echo 'export HF_HOME=/workspace/models' >> ~/.bashrc
echo 'export TRANSFORMERS_CACHE=/workspace/models' >> ~/.bashrc
source ~/.bashrc

# Create data directories
mkdir -p /workspace/repos/datatrove/spec/phase4/data
```

---

## 3. Initialization Script

Create `/workspace/init.sh` on your volume (one-time setup):

```bash
#!/bin/bash
set -e

echo "🚀 Initializing DataTrove Environment..."

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
    echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.bashrc
    source ~/.bashrc
    conda init bash
fi

source $HOME/miniconda3/bin/activate

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
pip install openvino zstandard warcio s3fs pymupdf orjson xgboost
pip install lmdeploy[all]
pip install qwen_vl_utils

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
```

Make it executable:
```bash
chmod +x /workspace/init.sh
```

---

## 4. RunPod Template Configuration

Create a custom template in RunPod dashboard:

### 4.1 Template Settings

**Container Image:**
```
runpod/pytorch:2.1.0-py3.11-cuda12.1.0-devel-ubuntu22.04
```

**Docker Command:**
```bash
bash -c "source ~/.bashrc && /workspace/init.sh && sleep infinity"
```

**Environment Variables:**
```
HF_HOME=/workspace/models
TRANSFORMERS_CACHE=/workspace/models
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
HF_HUB_ENABLE_HF_TRANSFER=0
```

**Volume Mount:**
- Select: `datatrove-workspace`
- Mount Path: `/workspace`

**Expose Ports:**
```
22/tcp  # SSH
8888/tcp  # Jupyter (optional)
```

### 4.2 Save Template

1. Name: `datatrove-finepdfs`
2. Save template
3. Use for all future instances

---

## 5. Daily Workflow

### 5.1 Start New Instance

1. Go to RunPod → **GPU Pods**
2. Click **Deploy**
3. Select template: `datatrove-finepdfs`
4. Choose GPU: RTX PRO 6000 (or available)
5. Click **Deploy On-Demand**

### 5.2 Wait for Initialization

The init script runs automatically. Monitor startup:
```bash
# SSH into instance
ssh root@<pod-ip> -p <port>

# Check init progress
tail -f /var/log/syslog | grep -E "(init.sh|conda|pip)"
```

Initialization takes ~5 minutes (vs 30+ minutes manual setup).

### 5.3 Start Working

```bash
conda activate datatrove-docling
cd /workspace/repos/datatrove

# Pull latest changes
git pull

# Run pipeline
python spec/phase4/examples/01_local_pdfs.py
```

### 5.4 Stop Instance

When done, **terminate the instance** (not the volume):
1. Your work persists on `/workspace/` volume
2. You're only billed for instance uptime
3. Models stay cached for next time

---

## 6. Volume Directory Structure

```
/workspace/
├── repos/
│   ├── datatrove/                    # Main repo
│   │   ├── spec/phase4/
│   │   │   ├── data/                 # Test PDFs
│   │   │   ├── output/               # Pipeline outputs
│   │   │   └── logs/                 # Pipeline logs
│   │   └── ...
│   └── Docling-sync/                 # Docling fork
│       └── models/
│           └── v2-quant.xml          # Optimized layout model
├── models/                           # HuggingFace cache
│   └── hub/
│       └── models--Reducto--RolmOCR/ # 15GB cached model
└── data/                             # Shared datasets (optional)
```

---

## 7. Managing Multiple Branches

Since repos are on volume, you can work on multiple branches:

```bash
# Main work
cd /workspace/repos/datatrove
git checkout feat/varied_local_pdf_testing
git pull

# Experimental branch
cd /workspace/repos/datatrove
git checkout -b experiment/new-feature
```

---

## 8. Troubleshooting

### Issue: Init script doesn't run

**Solution:**
```bash
# Run manually on first login
/workspace/init.sh
```

### Issue: Conda environment missing after restart

**Solution:** Environment is ephemeral by design. The init script recreates it:
```bash
/workspace/init.sh
```

Or manually:
```bash
conda activate datatrove-docling  # If it exists
# Otherwise, rerun init script
```

### Issue: Volume not mounted

**Solution:**
```bash
# Check mount
df -h | grep workspace

# If missing, stop pod and ensure volume is attached in RunPod dashboard
```

### Issue: Out of volume space

**Solution:**
```bash
# Check usage
du -sh /workspace/*

# Clean HuggingFace cache if needed
rm -rf /workspace/models/hub/.locks
pip cache purge
```

---

## 9. Cost Optimization

### Strategy
- **Keep volume**: Always (low cost ~$0.10/GB/month)
- **Terminate instance**: When not actively working
- **Model caching**: Models download once, reuse forever

### Example Monthly Cost
- Volume (100GB): ~$10/month
- GPU instance (RTX PRO 6000): $1.84/hr × 40 hours = $73.60
- **Total**: ~$83.60/month for 40 hours of compute

Compare to re-downloading models each time:
- 15GB RolmOCR download: ~5 minutes per instance
- 20 instances/month: 100 minutes wasted = $3+ lost to downloads

---

## 10. Advanced: Docker Image (Future)

For even faster deployment, build a custom Docker image with conda pre-installed:

```dockerfile
FROM runpod/pytorch:2.1.0-py3.11-cuda12.1.0-devel-ubuntu22.04

# Install system deps
RUN apt update && apt install -y git wget

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /root/miniconda3 && \
    rm /tmp/miniconda.sh

ENV PATH="/root/miniconda3/bin:$PATH"

# Set environment variables
ENV HF_HUB_ENABLE_HF_TRANSFER=0
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CMD ["/workspace/init.sh"]
```

**Benefits:**
- Miniconda pre-installed
- System deps baked in
- Even faster startup (~2 minutes)

---

## 11. Quick Reference

### Essential Commands
```bash
# Activate environment
conda activate datatrove-docling

# Update repos
cd /workspace/repos/datatrove && git pull
cd /workspace/repos/Docling-sync && git pull

# Run pipeline
cd /workspace/repos/datatrove
python spec/phase4/examples/01_local_pdfs.py

# Check GPU
nvidia-smi

# Monitor GPU in real-time
watch -n 1 nvidia-smi

# Clean up outputs
rm -rf /workspace/repos/datatrove/spec/phase4/output/*
rm -rf /workspace/repos/datatrove/spec/phase4/logs/*
```

### First-Time Setup Checklist
- [ ] Create volume in RunPod dashboard
- [ ] Deploy instance with volume attached
- [ ] Run initial setup (clone repos, create init.sh)
- [ ] Test init script works
- [ ] Create RunPod template
- [ ] Upload test PDFs to `/workspace/repos/datatrove/spec/phase4/data/`
- [ ] Run test pipeline

### Per-Session Checklist
- [ ] Deploy instance from template
- [ ] Wait for init script to complete (~5 min)
- [ ] Activate conda environment
- [ ] Pull latest changes
- [ ] Run pipeline
- [ ] Terminate instance when done

---

## Support

For issues specific to RunPod setup:
- RunPod Docs: https://docs.runpod.io/
- Community: https://discord.gg/runpod

For pipeline issues, see `lambda_setup.md`.
