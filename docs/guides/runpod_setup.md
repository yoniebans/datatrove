# RunPod Setup Guide
## DataTrove OCR Pipelines

This guide documents how to set up RunPod environments for DataTrove OCR pipelines with minimal setup time on new instances.

**Supported OCR Models:**
- **DeepSeek-OCR**: Uses vLLM with uv package manager (recommended for new projects)
- **RolmOCR**: Uses Docling + RolmOCR with conda (legacy setup)

---

## Overview

**Goal:** Separate persistent data (repos, models) from compute resources (GPU instances) so you can:
- Spin up/down instances without losing work
- Switch GPU types without re-downloading models
- Reduce setup time from 30+ minutes to 5 minutes

**Architecture:**
- **Persistent Volume**: Repos, HuggingFace models, test data
- **Ephemeral/Persistent Environments**: Depends on OCR model (see below)
- **Initialization Script**: Automates environment setup on new instances

---

## Environment Approaches

### DeepSeek-OCR (uv-based)
- **Package Manager**: `uv` (faster, modern)
- **Environment Location**: `/workspace/envs/deepseek-ocr` (persistent on volume)
- **Dependencies**: vLLM nightly, PyTorch with CUDA 12.9
- **Container**: `runpod/base:1.0.2-cuda1290-ubuntu2204`
- **Setup Script**: `spec/pdf-processing/deepseek-ocr/init_runpod.sh`
- **Pros**: Environment persists across instances, faster package resolution

### RolmOCR (conda-based)
- **Package Manager**: `conda` (traditional)
- **Environment Location**: `~/miniconda3/envs/datatrove-docling` (ephemeral, recreated each time)
- **Dependencies**: Docling, lmdeploy, qwen-vl-utils
- **Container**: `runpod/pytorch:2.1.0-py3.11-cuda12.1.0-devel-ubuntu22.04`
- **Setup Script**: `spec/pdf-processing/rolmocr/init_runpod.sh`
- **Pros**: Familiar conda workflow, works with Docling stack

**Note**: Future versions will standardize on uv for all models.

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

# Create data directories
cd /workspace/repos/datatrove
mkdir -p spec/phase4/data
```

---

## 3. Initialization Scripts

Each OCR model has its own initialization script. Copy the appropriate one to your volume:

### DeepSeek-OCR
```bash
# Copy init script to volume
cp /workspace/repos/datatrove/spec/pdf-processing/deepseek-ocr/init_runpod.sh /workspace/init.sh
chmod +x /workspace/init.sh
```

**Script location**: `spec/pdf-processing/deepseek-ocr/init_runpod.sh`

### RolmOCR
```bash
# Copy init script to volume
cp /workspace/repos/datatrove/spec/pdf-processing/rolmocr/init_runpod.sh /workspace/init.sh
chmod +x /workspace/init.sh
```

**Script location**: `spec/pdf-processing/rolmocr/init_runpod.sh`

**Note**: These scripts handle all environment setup automatically. Review them to understand what gets installed.

---

## 4. RunPod Template Configuration

Create a custom template in RunPod dashboard for your chosen OCR model.

### 4.1 Template Settings

**Container Image:**

Choose based on your OCR model (see [RunPod Official Containers](https://github.com/runpod/containers/tree/main/official-templates)):

```
# For DeepSeek-OCR:
runpod/base:1.0.2-cuda1290-ubuntu2204

# For RolmOCR:
runpod/pytorch:2.1.0-py3.11-cuda12.1.0-devel-ubuntu22.04
```

**Docker Command:**
```bash
bash -c "/start.sh > /tmp/start.log 2>&1 & tail -f /tmp/start.log | grep -q 'Pod is ready to use' && source ~/.bashrc && /workspace/init.sh && wait"
```

**Note:** This runs RunPod's `/start.sh` first (which sets up SSH using your account keys), waits for it to complete, then runs your custom init.sh.

**Environment Variables:**
```
HF_HOME=/workspace/models
TRANSFORMERS_CACHE=/workspace/models
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
HF_HUB_ENABLE_HF_TRANSFER=0
```

**Note:** SSH keys are automatically configured from your RunPod account settings (Settings → SSH Public Keys)

**Volume Mount:**
- Select: `datatrove-workspace`
- Mount Path: `/workspace`

**Expose Ports:**
```
22/tcp  # SSH
8888/tcp  # Jupyter (optional)
```

### 4.2 Save Template

1. Name: `datatrove-deepseek-ocr` or `datatrove-rolmocr` (based on your choice)
2. Save template
3. Use for all future instances of that model type

---

## 5. Daily Workflow

### 5.1 Start New Instance

1. Go to RunPod → **GPU Pods**
2. Click **Deploy**
3. Select your template (`datatrove-deepseek-ocr` or `datatrove-rolmocr`)
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

**For DeepSeek-OCR:**
```bash
source /workspace/envs/deepseek-ocr/bin/activate
cd /workspace/repos/datatrove

# Pull latest changes
git pull

# Run pipeline
python spec/pdf-processing/deepseek-ocr/deepseek_to_hf.py
```

**For RolmOCR:**
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
│   ├── datatrove/                               # Main repo
│   │   ├── spec/pdf-processing/
│   │   │   ├── deepseek-ocr/
│   │   │   │   ├── data/                        # DeepSeek PDFs
│   │   │   │   ├── output/                      # DeepSeek outputs
│   │   │   │   └── logs/                        # DeepSeek logs
│   │   │   └── rolmocr/
│   │   │       ├── data/                        # RolmOCR PDFs
│   │   │       └── utils/                       # PDF utilities
│   │   ├── spec/phase4/
│   │   │   ├── data/                            # Legacy RolmOCR PDFs
│   │   │   ├── output/                          # Legacy outputs
│   │   │   └── logs/                            # Legacy logs
│   │   └── ...
│   └── Docling-sync/                            # Docling fork (RolmOCR only)
│       └── models/
│           └── v2-quant.xml                     # Optimized layout model
├── envs/
│   └── deepseek-ocr/                            # DeepSeek uv environment (persistent)
├── models/                                      # HuggingFace cache
│   └── hub/
│       ├── models--deepseek-ai--DeepSeek-OCR/   # DeepSeek model
│       └── models--Reducto--RolmOCR/            # RolmOCR model
└── data/                                        # Shared datasets (optional)
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

### Issue: Environment missing after restart

**Solution:**

For RolmOCR (conda - ephemeral by design):
```bash
# Recreate with init script
/workspace/init.sh

# Or manually activate if it exists
conda activate datatrove-docling
```

For DeepSeek-OCR (uv - should persist):
```bash
# Environment should still exist on volume
source /workspace/envs/deepseek-ocr/bin/activate

# If missing, rerun init script
/workspace/init.sh
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

**DeepSeek-OCR:**
```bash
# Activate environment
source /workspace/envs/deepseek-ocr/bin/activate

# Update repos
cd /workspace/repos/datatrove && git pull

# Run pipeline
python spec/pdf-processing/deepseek-ocr/deepseek_to_hf.py

# Check GPU
nvidia-smi

# Monitor GPU in real-time
watch -n 1 nvidia-smi

# Clean up outputs
rm -rf /workspace/repos/datatrove/spec/pdf-processing/deepseek-ocr/output/*
rm -rf /workspace/repos/datatrove/spec/pdf-processing/deepseek-ocr/logs/*
```

**RolmOCR:**
```bash
# Activate environment
conda activate datatrove-docling

# Update repos
cd /workspace/repos/datatrove && git pull
cd /workspace/repos/Docling-sync && git pull

# Run pipeline
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
- [ ] Run initial setup (clone repos)
- [ ] Copy appropriate init script to `/workspace/init.sh`
- [ ] Test init script works
- [ ] Create RunPod template with correct container image
- [ ] Upload test PDFs to appropriate data directory
- [ ] Run test pipeline

### Per-Session Checklist
- [ ] Deploy instance from template
- [ ] Wait for init script to complete (~5 min)
- [ ] Activate environment (uv or conda depending on model)
- [ ] Pull latest changes
- [ ] Run pipeline
- [ ] Terminate instance when done

---

## Support

For issues specific to RunPod setup:
- RunPod Docs: https://docs.runpod.io/
- RunPod Containers: https://github.com/runpod/containers/tree/main/official-templates
- Community: https://discord.gg/runpod

For model-specific issues, check the respective init scripts:
- DeepSeek-OCR: `spec/pdf-processing/deepseek-ocr/init_runpod.sh`
- RolmOCR: `spec/pdf-processing/rolmocr/init_runpod.sh`
