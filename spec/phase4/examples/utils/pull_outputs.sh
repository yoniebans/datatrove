#!/bin/bash
# Pull Phase 4 PDF processing results from remote server for local review
#
# Usage:
#   ./pull_outputs.sh <ssh_key_path> <remote_path> <local_path>
#
# Notes:
#   - remote_path: absolute path on remote server (e.g., /workspace/repos/datatrove/...)
#   - local_path: resolved relative to where you run the script from (use absolute paths to avoid confusion)
#
# Example (using absolute paths):
#   export REMOTE_HOST="root@213.181.122.235"
#   export REMOTE_PORT="11531"
#   ./pull_outputs.sh ~/.ssh/id_ed25519 "/workspace/repos/datatrove/spec/phase4/output/01_local_pdfs" "/Users/you/datatrove/spec/phase4/data/results"
#
# Example (using relative paths - assumes running from datatrove repo root):
#   export REMOTE_HOST="root@213.181.122.235"
#   export REMOTE_PORT="11531"
#   ./spec/phase4/examples/utils/pull_outputs.sh ~/.ssh/id_ed25519 "/workspace/repos/datatrove/spec/phase4/output/01_local_pdfs" "spec/phase4/data/results"
#
# Environment variables (required):
#   REMOTE_HOST - SSH host (e.g., root@213.181.122.235)
#   REMOTE_PORT - SSH port

set -e

# Check arguments
if [ "$#" -lt 3 ]; then
    echo "Error: Missing arguments"
    echo ""
    echo "Usage:"
    echo "  ./pull_outputs.sh <ssh_key_path> <remote_path> <local_path>"
    echo ""
    echo "Example (using absolute paths):"
    echo "  export REMOTE_HOST='root@213.181.122.235'"
    echo "  export REMOTE_PORT='11531'"
    echo "  ./pull_outputs.sh ~/.ssh/id_ed25519 \"/workspace/repos/datatrove/spec/phase4/output/01_local_pdfs\" \"/Users/you/datatrove/spec/phase4/data/results\""
    exit 1
fi

SSH_KEY_PATH="$1"
REMOTE_PATH="$2"
LOCAL_PATH="$3"

# Check required environment variables
if [ -z "$REMOTE_HOST" ]; then
    echo "Error: REMOTE_HOST environment variable not set"
    exit 1
fi

if [ -z "$REMOTE_PORT" ]; then
    echo "Error: REMOTE_PORT environment variable not set"
    exit 1
fi

echo "=========================================="
echo "Downloading from Remote Server"
echo "=========================================="
echo "Host: $REMOTE_HOST:$REMOTE_PORT"
echo "SSH Key: $SSH_KEY_PATH"
echo "Remote: $REMOTE_PATH"
echo "Local: $LOCAL_PATH"
echo ""

# Create local directory structure
mkdir -p "$LOCAL_PATH"

# Pull all outputs maintaining structure
echo "📥 Downloading all files (PDFs, PNGs, JSONL)..."
scp -P "$REMOTE_PORT" -i "$SSH_KEY_PATH" -r "$REMOTE_HOST:$REMOTE_PATH"/* "$LOCAL_PATH/"

echo ""
echo "✅ Files downloaded to: $LOCAL_PATH"
echo ""

# Run text extraction script
echo "📝 Extracting text to readable format..."
python spec/phase4/examples/utils/extract_text_for_review.py

echo ""
echo "✅ Complete! Review files are ready."
echo ""
