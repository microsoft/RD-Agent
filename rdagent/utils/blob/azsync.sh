#!/bin/bash
# Azure Blob 同步脚本 - 用于多机器间同步 FT 场景的 log 文件

# ========== 配置区域 ==========
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/../../.."
LOCAL_LOG_DIR="$PROJECT_ROOT/log"
TOKEN_FILE="$PROJECT_ROOT/git_ignore_folder/.az_sas_token"

# Blob 配置
ACCOUNT="epeastus"
CONTAINER="rdagent"
REMOTE_PATH="FinetuneAgenticLLM/FT_qizheng/logs"  # 修改这里指定 blob 上的路径

# 读取 SAS Token
if [ -f "$TOKEN_FILE" ]; then
    SAS_TOKEN=$(cat "$TOKEN_FILE")
else
    SAS_TOKEN=""
fi
# ========== 配置结束 ==========

BLOB_URL="https://${ACCOUNT}.blob.core.windows.net/${CONTAINER}/${REMOTE_PATH}?${SAS_TOKEN}"

usage() {
    echo "用法: $0 [up|down]"
    echo ""
    echo "  up    上传本地 log 到 blob"
    echo "  down  下载 blob 到本地 log"
    echo "  无参数 显示此帮助"
    echo ""
    echo "配置:"
    echo "  本地目录: $LOCAL_LOG_DIR"
    echo "  远程路径: $REMOTE_PATH"
    echo ""
    echo "SAS Token: 运行 ./gen_token.sh 生成"
    exit 0
}

check_token() {
    if [ -z "$SAS_TOKEN" ]; then
        echo "错误: 未找到 SAS Token"
        echo "请先运行: ./gen_token.sh"
        exit 1
    fi
}

case "${1:-}" in
    up)
        check_token
        echo "上传中: $LOCAL_LOG_DIR -> $REMOTE_PATH"
        azcopy sync "$LOCAL_LOG_DIR" "$BLOB_URL" --recursive=true \
            --exclude-path="pickle_cache;prompt_cache.db"
        ;;
    down)
        check_token
        mkdir -p "$LOCAL_LOG_DIR"
        echo "下载中: $REMOTE_PATH -> $LOCAL_LOG_DIR"
        azcopy sync "$BLOB_URL" "$LOCAL_LOG_DIR" --recursive=true
        ;;
    *)
        usage
        ;;
esac
