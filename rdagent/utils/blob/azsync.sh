#!/bin/bash
# Azure Blob sync script - for syncing FT scenario log files across machines

# ========== Configuration ==========
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/../../.."
LOCAL_LOG_DIR="$PROJECT_ROOT/log"
TOKEN_FILE="$PROJECT_ROOT/git_ignore_folder/.az_sas_token"

# Blob configuration
ACCOUNT="epeastus"
CONTAINER="rdagent"
REMOTE_PATH="FinetuneAgenticLLM/FT_qizheng/logs"  # Modify this to specify the blob path

# Read SAS Token
if [ -f "$TOKEN_FILE" ]; then
    SAS_TOKEN=$(cat "$TOKEN_FILE")
else
    SAS_TOKEN=""
fi
# ========== End Configuration ==========

BLOB_URL="https://${ACCOUNT}.blob.core.windows.net/${CONTAINER}/${REMOTE_PATH}?${SAS_TOKEN}"

usage() {
    echo "Usage: $0 [up|down]"
    echo ""
    echo "  up    Upload local log to blob"
    echo "  down  Download blob to local log"
    echo "  (no args) Show this help"
    echo ""
    echo "Configuration:"
    echo "  Local directory: $LOCAL_LOG_DIR"
    echo "  Remote path: $REMOTE_PATH"
    echo ""
    echo "SAS Token: Run ./gen_token.sh to generate"
    exit 0
}

check_token() {
    if [ -z "$SAS_TOKEN" ]; then
        echo "Error: SAS Token not found"
        echo "Please run: ./gen_token.sh first"
        exit 1
    fi
}

case "${1:-}" in
    up)
        check_token
        echo "Uploading: $LOCAL_LOG_DIR -> $REMOTE_PATH"
        azcopy sync "$LOCAL_LOG_DIR" "$BLOB_URL" --recursive=true \
            --exclude-path="pickle_cache;prompt_cache.db"
        ;;
    down)
        check_token
        mkdir -p "$LOCAL_LOG_DIR"
        echo "Downloading: $REMOTE_PATH -> $LOCAL_LOG_DIR"
        azcopy sync "$BLOB_URL" "$LOCAL_LOG_DIR" --recursive=true
        ;;
    *)
        usage
        ;;
esac
