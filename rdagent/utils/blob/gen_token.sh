#!/bin/bash
# Generate Azure Blob SAS Token and save it

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/../../.."
TOKEN_FILE="$PROJECT_ROOT/git_ignore_folder/.az_sas_token"

# Blob configuration
ACCOUNT="epeastus"
CONTAINER="rdagent"
REMOTE_PATH="FinetuneAgenticLLM/FT_qizheng/logs"

# Default expiry: 7 days from now
DEFAULT_EXPIRY=$(date -u -d "+7 days" +%Y-%m-%dT00:00Z 2>/dev/null || date -u -v+7d +%Y-%m-%dT00:00Z)
EXPIRY="${1:-$DEFAULT_EXPIRY}"

echo "Generating SAS Token..."
echo "Expires at: $EXPIRY"
echo ""

# Generate token
TOKEN=$(az storage container generate-sas \
    --as-user \
    --auth-mode login \
    --account-name "$ACCOUNT" \
    --name "$CONTAINER" \
    --permissions lrwd \
    --expiry "$EXPIRY" \
    -o tsv)

if [ -z "$TOKEN" ]; then
    echo "Error: Token generation failed, please ensure you are logged in to az cli"
    echo "Run: az login"
    exit 1
fi

# Save token
mkdir -p "$(dirname "$TOKEN_FILE")"
echo "$TOKEN" > "$TOKEN_FILE"
echo "Token saved to: $TOKEN_FILE"
echo ""

# Output full URL
BLOB_URL="https://${ACCOUNT}.blob.core.windows.net/${CONTAINER}/${REMOTE_PATH}?${TOKEN}"
echo "Full Blob URL:"
echo "$BLOB_URL"
