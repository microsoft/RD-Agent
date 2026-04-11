#!/bin/bash
#
# Qlib Data Download Script
#
# This script downloads Qlib Chinese A-share market data (1d frequency)
# to the local data directory for use with RD-Agent.
#
# Usage:
#   ./download_qlib_data.sh [target_directory]
#
# Arguments:
#   target_directory: Optional. Default is ./qlib_data/cn_data
#

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_TARGET_DIR="${SCRIPT_DIR}/qlib_data/cn_data"
TARGET_DIR="${1:-$DEFAULT_TARGET_DIR}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================"
echo "  Qlib Data Download Script"
echo "========================================"
echo ""
echo "Target directory: ${TARGET_DIR}"
echo ""

# Check if pyqlib is installed
if ! command -v python &> /dev/null; then
    echo -e "${RED}Error: Python is not installed or not in PATH${NC}"
    exit 1
fi

# Check if qlib is installed
if ! python -c "import qlib" 2>/dev/null; then
    echo -e "${RED}Error: pyqlib is not installed${NC}"
    echo -e "${YELLOW}Please install it first:${NC}"
    echo "  pip install pyqlib"
    echo "  or"
    echo "  pip install git+https://github.com/microsoft/qlib.git"
    exit 1
fi

# Check if data already exists
if [ -d "${TARGET_DIR}" ] && [ "$(ls -A ${TARGET_DIR})" ]; then
    echo -e "${YELLOW}Warning: Data directory already exists and is not empty${NC}"
    echo "  Directory: ${TARGET_DIR}"
    echo ""
    read -p "Do you want to overwrite? [y/N] " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${GREEN}Download skipped.${NC}"
        exit 0
    fi
    echo -e "${YELLOW}Removing existing data...${NC}"
    rm -rf "${TARGET_DIR}"
fi

# Create target directory
echo "Creating target directory..."
mkdir -p "${TARGET_DIR}"

# Download data
echo ""
echo -e "${YELLOW}Starting data download...${NC}"
echo "This may take several minutes depending on your network speed."
echo ""

python -m qlib.run.get_data qlib_data \
    --target_dir "${TARGET_DIR}" \
    --region cn \
    --interval 1d \
    --delete_old False

# Verify download
if [ -d "${TARGET_DIR}" ] && [ "$(ls -A ${TARGET_DIR})" ]; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  Data download completed successfully!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "Data location: ${TARGET_DIR}"
    echo ""
    echo "To use this data with RD-Agent, set the environment variable:"
    echo "  export QLIB_DATA_PATH=\"${TARGET_DIR}\""
    echo ""
    echo "Or add to your config/settings.yaml:"
    echo "  qlib:"
    echo "    data_path: ${TARGET_DIR}"
else
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}  Download may have failed. Please check the logs above.${NC}"
    echo -e "${RED}========================================${NC}"
    exit 1
fi
