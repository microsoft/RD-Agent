#!/bin/bash
# 生成 Azure Blob SAS Token 并保存

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/../../.."
TOKEN_FILE="$PROJECT_ROOT/git_ignore_folder/.az_sas_token"

# Blob 配置
ACCOUNT="epeastus"
CONTAINER="rdagent"
REMOTE_PATH="FinetuneAgenticLLM/FT_qizheng/logs"

# 默认有效期：7天后
DEFAULT_EXPIRY=$(date -u -d "+7 days" +%Y-%m-%dT00:00Z 2>/dev/null || date -u -v+7d +%Y-%m-%dT00:00Z)
EXPIRY="${1:-$DEFAULT_EXPIRY}"

echo "生成 SAS Token..."
echo "有效期至: $EXPIRY"
echo ""

# 生成 token
TOKEN=$(az storage container generate-sas \
    --as-user \
    --auth-mode login \
    --account-name "$ACCOUNT" \
    --name "$CONTAINER" \
    --permissions lrwd \
    --expiry "$EXPIRY" \
    -o tsv)

if [ -z "$TOKEN" ]; then
    echo "错误: Token 生成失败，请确保已登录 az cli"
    echo "运行: az login"
    exit 1
fi

# 保存 token
mkdir -p "$(dirname "$TOKEN_FILE")"
echo "$TOKEN" > "$TOKEN_FILE"
echo "Token 已保存到: $TOKEN_FILE"
echo ""

# 输出完整 URL
BLOB_URL="https://${ACCOUNT}.blob.core.windows.net/${CONTAINER}/${REMOTE_PATH}?${TOKEN}"
echo "完整 Blob URL:"
echo "$BLOB_URL"
