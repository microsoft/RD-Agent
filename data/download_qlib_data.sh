#!/bin/bash
#
# Qlib Data Download Script
#
# 下载 Qlib 中国 A 股市场数据（1d 频率）
# 数据源：https://github.com/chenditc/investment_data/releases
#
# 用法:
#   ./download_qlib_data.sh [target_directory]
#
# 参数:
#   target_directory: 可选，默认为 ~/.qlib/qlib_data/cn_data
#

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_TARGET_DIR="$HOME/.qlib/qlib_data/cn_data"
TARGET_DIR="${1:-$DEFAULT_TARGET_DIR}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "========================================"
echo "  Qlib Data Download Script"
echo "========================================"
echo ""
echo -e "${BLUE}数据源：${NC}https://github.com/chenditc/investment_data/releases"
echo -e "${BLUE}目标目录：${NC}${TARGET_DIR}"
echo ""

# Check if data already exists
if [ -d "${TARGET_DIR}" ] && [ "$(ls -A ${TARGET_DIR})" ]; then
    echo -e "${YELLOW}警告：数据目录已存在且非空${NC}"
    echo "  目录：${TARGET_DIR}"
    echo ""
    read -p "是否覆盖？[y/N] " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${GREEN}下载已跳过。${NC}"
        exit 0
    fi
    echo -e "${YELLOW}正在删除现有数据...${NC}"
    rm -rf "${TARGET_DIR}"
fi

# Create target directory
echo "创建目标目录..."
mkdir -p "${TARGET_DIR}"

# Download data
echo ""
echo -e "${YELLOW}开始下载数据...${NC}"
echo "数据文件约 1-2GB，下载时间取决于网络速度"
echo ""

# Download from GitHub release
cd /tmp
wget https://github.com/chenditc/investment_data/releases/latest/download/qlib_bin.tar.gz

echo ""
echo -e "${YELLOW}正在解压数据...${NC}"
tar -zxvf qlib_bin.tar.gz -C "${TARGET_DIR}" --strip-components=1

rm -f qlib_bin.tar.gz

# Verify download
if [ -d "${TARGET_DIR}" ] && [ "$(ls -A ${TARGET_DIR})" ]; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  数据下载完成！${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "数据位置：${TARGET_DIR}"
    echo ""
    echo -e "${YELLOW}使用方式（二选一）:${NC}"
    echo ""
    echo "1. 环境变量:"
    echo "   export QLIB_DATA_PATH=\"${TARGET_DIR}\""
    echo ""
    echo "2. 配置文件 (rdagent/config/settings.yaml):"
    echo "   qlib:"
    echo "     data_path: ${TARGET_DIR}"
    echo ""
else
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}  下载可能失败，请检查上方日志。${NC}"
    echo -e "${RED}========================================${NC}"
    exit 1
fi
