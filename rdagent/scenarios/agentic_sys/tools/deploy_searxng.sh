#!/bin/bash
# filepath: /data/userdata/v-wangzhu/RD-Agent/rdagent/scenarios/agentic_sys/tools/deploy_searxng.sh

# SearxNG Deployment Script
set -e

SEARXNG_DIR="${HOME}/apps/searxng"
SEARXNG_PORT=8888
CONTAINER_NAME="searxng"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

deploy() {
    log_info "Deploying SearxNG..."
    mkdir -p "${SEARXNG_DIR}/config" "${SEARXNG_DIR}/data"
    
    if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        log_warn "Container exists, removing..."
        docker rm -f ${CONTAINER_NAME}
    fi
    
    docker run --name ${CONTAINER_NAME} -d \
        -p ${SEARXNG_PORT}:8080 \
        -v "${SEARXNG_DIR}/config:/etc/searxng/" \
        -v "${SEARXNG_DIR}/data:/var/cache/searxng/" \
        --restart unless-stopped \
        docker.io/searxng/searxng:latest
    
    log_info "SearxNG deployed at http://localhost:${SEARXNG_PORT}"
    sleep 5
}

update_config() {
    log_info "Updating configuration..."
    CONFIG_FILE="${SEARXNG_DIR}/config/settings.yml"
    
    local attempts=0
    while [ ! -f "$CONFIG_FILE" ] && [ $attempts -lt 10 ]; do
        log_info "Waiting for config file... ($((attempts+1))/10)"
        sleep 2
        attempts=$((attempts+1))
    done
    
    if [ ! -f "$CONFIG_FILE" ]; then
        log_error "Config file not found!"
        exit 1
    fi
    
    sudo chmod 777 -R "${SEARXNG_DIR}/config/"
    
    if ! command -v yq &> /dev/null; then
        log_warn "Installing yq..."
        sudo wget -qO /usr/local/bin/yq https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64
        sudo chmod +x /usr/local/bin/yq
    fi
    
    yq -i '.search.formats = ["html", "json", "csv"]' "$CONFIG_FILE"
    log_info "Configuration updated"
    
    restart
}

restart() {
    log_info "Restarting SearxNG..."
    docker restart ${CONTAINER_NAME} >/dev/null
    log_info "Restarted successfully"
    sleep 3
}

status() {
    log_info "SearxNG Status:"
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo -e "${GREEN}  Status: Running${NC}"
        echo "  URL: http://localhost:${SEARXNG_PORT}"
    else
        echo -e "${RED}  Status: Not running${NC}"
    fi
}

case "${1:-help}" in
    deploy) deploy ;;
    update_config) update_config ;;
    restart) restart ;;
    status) status ;;
    *) echo "Usage: $0 {deploy|update_config|restart|status}" ;;
esac