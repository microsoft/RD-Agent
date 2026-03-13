#!/bin/bash
# Azure Blob sync script - for syncing FT scenario files across machines
# Supports both logs and workspace directories

# ========== Configuration ==========
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/../../.."
TOKEN_FILE="$PROJECT_ROOT/git_ignore_folder/.az_sas_token"

# Blob configuration
ACCOUNT="epeastus"
CONTAINER="rdagent"
REMOTE_BASE="FinetuneAgenticLLM/FT_qizheng"

# Directory mappings (support environment variable override)
# Default to project-relative paths; can be overridden by environment variables
LOCAL_LOG_DIR="${FT_LOG_BASE:-$PROJECT_ROOT/log}"
LOCAL_WORKSPACE_DIR="${FT_WORKSPACE_BASE:-$PROJECT_ROOT/git_ignore_folder/RD-Agent_workspace}"
LOCAL_LITELLM_LOG_DIR="${LITELLM_LOG_DIR:-/workspace/rdagent/litllm_log}"
# Support sub-path for syncing specific job directory (e.g., SYNC_SUBPATH="2024-01-01_12-00")
SYNC_SUBPATH="${SYNC_SUBPATH:-}"
REMOTE_LOG_PATH="${REMOTE_BASE}/logs${SYNC_SUBPATH:+/$SYNC_SUBPATH}"
REMOTE_WORKSPACE_PATH="${REMOTE_BASE}/workspace${SYNC_SUBPATH:+/$SYNC_SUBPATH}"
# litellm_log doesn't use SYNC_SUBPATH since local dir is shared across jobs
REMOTE_LITELLM_LOG_PATH="${REMOTE_BASE}/litellm_log"

# Read SAS Token
if [ -f "$TOKEN_FILE" ]; then
    SAS_TOKEN=$(cat "$TOKEN_FILE")
else
    SAS_TOKEN=""
fi
# ========== End Configuration ==========

# Get paths based on sync type (logs/workspace/litellm_log)
get_paths() {
    local sync_type="${1:-logs}"
    case "$sync_type" in
        logs)
            LOCAL_DIR="$LOCAL_LOG_DIR"
            REMOTE_PATH="$REMOTE_LOG_PATH"
            ;;
        workspace)
            LOCAL_DIR="$LOCAL_WORKSPACE_DIR"
            REMOTE_PATH="$REMOTE_WORKSPACE_PATH"
            ;;
        litellm_log)
            LOCAL_DIR="$LOCAL_LITELLM_LOG_DIR"
            REMOTE_PATH="$REMOTE_LITELLM_LOG_PATH"
            ;;
        *)
            echo "Error: Unknown sync type '$sync_type'. Use 'logs', 'workspace', or 'litellm_log'."
            exit 1
            ;;
    esac
    BLOB_URL="https://${ACCOUNT}.blob.core.windows.net/${CONTAINER}/${REMOTE_PATH}?${SAS_TOKEN}"
}

usage() {
    echo "Usage: $0 [up|down] [logs|workspace|litellm_log]"
    echo ""
    echo "  up    Upload local directory to blob"
    echo "  down  Download blob to local directory"
    echo "  (no args) Show this help"
    echo ""
    echo "Sync types:"
    echo "  logs        Sync log directory (default)"
    echo "  workspace   Sync workspace directory"
    echo "  litellm_log Sync litellm log directory"
    echo ""
    echo "Configuration:"
    echo "  Log directory:         $LOCAL_LOG_DIR"
    echo "  Workspace directory:   $LOCAL_WORKSPACE_DIR"
    echo "  Litellm log directory: $LOCAL_LITELLM_LOG_DIR"
    echo "  Remote base:           $REMOTE_BASE"
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
        get_paths "${2:-logs}"
        echo "Uploading: $LOCAL_DIR -> $REMOTE_PATH"
        azcopy sync "$LOCAL_DIR" "$BLOB_URL" --recursive=true \
            --exclude-path="pickle_cache;prompt_cache.db"
        ;;
    down)
        check_token
        get_paths "${2:-logs}"
        mkdir -p "$LOCAL_DIR"
        echo "Downloading: $REMOTE_PATH -> $LOCAL_DIR"
        azcopy sync "$BLOB_URL" "$LOCAL_DIR" --recursive=true
        ;;
    *)
        usage
        ;;
esac
