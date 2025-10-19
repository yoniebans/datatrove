#!/bin/bash
# Pull Phase 4 PDF processing results from remote server for local review

# Configuration - set environment variables before running
# Example: export REMOTE_HOST="root@your-host" REMOTE_PORT="22" REMOTE_SSH_KEY="~/.ssh/id_rsa"
if [ -z "$REMOTE_HOST" ]; then
    echo "Error: REMOTE_HOST environment variable not set"
    echo "Usage:"
    echo "  export REMOTE_HOST='root@your-host'"
    echo "  export REMOTE_PORT='22'  # Optional, defaults to 22"
    echo "  export REMOTE_SSH_KEY='~/.ssh/id_rsa'  # Optional, defaults to ~/.ssh/id_rsa"
    echo "  ./pull_results.sh"
    exit 1
fi

# Default values
REMOTE_PORT="${REMOTE_PORT:-22}"
REMOTE_SSH_KEY="${REMOTE_SSH_KEY:-$HOME/.ssh/id_rsa}"

REMOTE_DIR="datatrove/spec/phase4/output/01_local_pdfs"
LOCAL_DIR="spec/phase4/data/results"

echo "=========================================="
echo "Pulling Phase 4 Results from Server"
echo "=========================================="
echo "Host: $REMOTE_HOST:$REMOTE_PORT"
echo

# Create local directory structure
mkdir -p "$LOCAL_DIR"

# Pull all outputs maintaining structure
echo "📥 Downloading all files (PDFs, PNGs, JSONL)..."
scp -P "$REMOTE_PORT" -i "$REMOTE_SSH_KEY" -r "$REMOTE_HOST:$REMOTE_DIR"/* "$LOCAL_DIR/"

echo
echo "✅ Files downloaded to: $LOCAL_DIR"
echo

# Run text extraction script
echo "📝 Extracting text to readable format..."
python spec/phase4/examples/utils/extract_text_for_review.py

echo
echo "✅ Complete! Review files are ready."
echo
