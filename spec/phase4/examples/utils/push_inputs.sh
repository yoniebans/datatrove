#!/bin/bash
# Upload files to RunPod/remote server
#
# Usage:
#   ./push_inputs.sh <ssh_key_path> <local_path> <remote_path>
#
# Notes:
#   - local_path: resolved relative to where you run the script from (use absolute paths to avoid confusion)
#   - remote_path: absolute path on remote server (e.g., /workspace/repos/datatrove/...)
#
# Example (using absolute paths):
#   export REMOTE_HOST="root@213.181.122.235"
#   export REMOTE_PORT="11531"
#   ./push_inputs.sh ~/.ssh/id_ed25519 "/Users/you/datatrove/spec/phase4/data/*.pdf" "/workspace/repos/datatrove/spec/phase4/data/"
#
# Example (using relative paths - assumes running from datatrove repo root):
#   export REMOTE_HOST="root@213.181.122.235"
#   export REMOTE_PORT="11531"
#   ./spec/phase4/examples/utils/push_inputs.sh ~/.ssh/id_ed25519 "spec/phase4/data/*.pdf" "/workspace/repos/datatrove/spec/phase4/data/"
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
    echo "  ./push_inputs.sh <ssh_key_path> <local_path> <remote_path>"
    echo ""
    echo "Example (using absolute paths):"
    echo "  export REMOTE_HOST='root@213.181.122.235'"
    echo "  export REMOTE_PORT='11531'"
    echo "  ./push_inputs.sh ~/.ssh/id_ed25519 \"/Users/you/datatrove/spec/phase4/data/*.pdf\" \"/workspace/repos/datatrove/spec/phase4/data/\""
    exit 1
fi

SSH_KEY_PATH="$1"
LOCAL_PATH="$2"
REMOTE_PATH="$3"

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
echo "Uploading to Remote Server"
echo "=========================================="
echo "Host: $REMOTE_HOST:$REMOTE_PORT"
echo "SSH Key: $SSH_KEY_PATH"
echo "Local: $LOCAL_PATH"
echo "Remote: $REMOTE_PATH"
echo ""

# Upload files
scp -P "$REMOTE_PORT" -i "$SSH_KEY_PATH" $LOCAL_PATH "$REMOTE_HOST:$REMOTE_PATH"

echo ""
echo "✅ Upload complete!"
