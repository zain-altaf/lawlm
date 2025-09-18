#!/bin/bash
set -euo pipefail

# Qdrant local management script
# This script helps manage a local Qdrant instance for development

QDRANT_VERSION="latest"
QDRANT_PORT=6333
QDRANT_STORAGE="./qdrant_storage"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check if Docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}Error: Docker is not installed. Please install Docker first.${NC}"
        exit 1
    fi

    if ! docker info &> /dev/null; then
        echo -e "${RED}Error: Docker daemon is not running. Please start Docker.${NC}"
        exit 1
    fi
}

# Function to check if Qdrant is running
is_running() {
    docker ps --filter "name=qdrant" --filter "status=running" --quiet | grep -q .
}

# Start Qdrant
start_qdrant() {
    check_docker

    if is_running; then
        echo -e "${YELLOW}Qdrant is already running${NC}"
        return 0
    fi

    # Check if container exists but is stopped
    if docker ps -a --filter "name=qdrant" --quiet | grep -q .; then
        echo -e "${GREEN}Starting existing Qdrant container...${NC}"
        docker start qdrant
    else
        echo -e "${GREEN}Creating and starting new Qdrant container...${NC}"

        # Create storage directory if it doesn't exist
        mkdir -p "$QDRANT_STORAGE"

        # Run Qdrant container
        docker run -d \
            --name qdrant \
            -p "${QDRANT_PORT}:6333" \
            -v "$(pwd)/qdrant_storage:/qdrant/storage" \
            "qdrant/qdrant:${QDRANT_VERSION}"
    fi

    # Wait for container to be ready
    echo -e "${GREEN}Waiting for Qdrant to be ready...${NC}"
    for i in {1..30}; do
        if curl -s "http://localhost:${QDRANT_PORT}/health" &> /dev/null; then
            echo -e "${GREEN}Qdrant started successfully on port ${QDRANT_PORT}${NC}"
            echo -e "${GREEN}Dashboard available at: http://localhost:${QDRANT_PORT}/dashboard${NC}"
            return 0
        fi
        sleep 1
    done

    echo -e "${RED}Error: Qdrant failed to start or is not responding${NC}"
    exit 1
}

# Stop Qdrant
stop_qdrant() {
    check_docker

    if ! is_running; then
        echo -e "${YELLOW}Qdrant is not running${NC}"
        return 0
    fi

    echo -e "${GREEN}Stopping Qdrant...${NC}"
    if docker stop qdrant && docker rm qdrant; then
        echo -e "${GREEN}Qdrant stopped successfully${NC}"
    else
        echo -e "${RED}Error: Failed to stop Qdrant${NC}"
        exit 1
    fi
}

# Restart Qdrant
restart_qdrant() {
    echo -e "${GREEN}Restarting Qdrant...${NC}"
    stop_qdrant
    sleep 2
    start_qdrant
}

# Check Qdrant status
status_qdrant() {
    check_docker

    if is_running; then
        echo -e "${GREEN}Status: Qdrant is running${NC}"

        # Check health endpoint
        if curl -s "http://localhost:${QDRANT_PORT}/health" > /dev/null; then
            echo -e "${GREEN}Health: OK${NC}"
        else
            echo -e "${YELLOW}Health: Qdrant is running but health check failed${NC}"
        fi

        # Show container info
        echo -e "\n${GREEN}Container Info:${NC}"
        docker ps --filter "name=qdrant" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

        # Show storage info
        if [ -d "$QDRANT_STORAGE" ]; then
            echo -e "\n${GREEN}Storage: $(du -sh "$QDRANT_STORAGE" | cut -f1)${NC}"
        fi
    else
        echo -e "${RED}Status: Qdrant is not running${NC}"

        # Check if container exists but is stopped
        if docker ps -a --filter "name=qdrant" --quiet | grep -q .; then
            echo -e "${YELLOW}Note: Qdrant container exists but is stopped${NC}"
        fi
    fi
}

# View Qdrant logs
logs_qdrant() {
    check_docker

    if ! docker ps -a --filter "name=qdrant" --quiet | grep -q .; then
        echo -e "${RED}Error: Qdrant container does not exist${NC}"
        exit 1
    fi

    echo -e "${GREEN}Showing Qdrant logs (Ctrl+C to exit)...${NC}"
    docker logs -f qdrant
}

# Clean Qdrant storage
clean_storage() {
    if is_running; then
        echo -e "${RED}Error: Please stop Qdrant before cleaning storage${NC}"
        exit 1
    fi

    if [ ! -d "$QDRANT_STORAGE" ]; then
        echo -e "${YELLOW}Storage directory does not exist${NC}"
        return 0
    fi

    echo -e "${YELLOW}WARNING: This will delete all data in Qdrant storage!${NC}"
    echo -e "Storage location: $QDRANT_STORAGE"
    if [ -d "$QDRANT_STORAGE" ]; then
        echo -e "Current size: $(du -sh "$QDRANT_STORAGE" | cut -f1)"
    fi

    read -p "Are you sure you want to proceed? (y/N) " -n 1 -r
    echo

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        if rm -rf "$QDRANT_STORAGE" && mkdir -p "$QDRANT_STORAGE"; then
            echo -e "${GREEN}Storage cleaned successfully${NC}"
        else
            echo -e "${RED}Error: Failed to clean storage${NC}"
            exit 1
        fi
    else
        echo -e "${YELLOW}Operation cancelled${NC}"
    fi
}

# Show help
show_help() {
    echo "Qdrant Local Management Script"
    echo ""
    echo "Usage: ./manage_qdrant.sh [command]"
    echo ""
    echo "Commands:"
    echo "  start    - Start Qdrant container"
    echo "  stop     - Stop Qdrant container"
    echo "  restart  - Restart Qdrant container"
    echo "  status   - Check Qdrant status"
    echo "  logs     - View Qdrant logs"
    echo "  clean    - Clean Qdrant storage (removes all data)"
    echo "  help     - Show this help message"
    echo ""
    echo "Qdrant will be available at: http://localhost:${QDRANT_PORT}"
    echo "Dashboard: http://localhost:${QDRANT_PORT}/dashboard"
}

# Main script logic
case "$1" in
    start)
        start_qdrant
        ;;
    stop)
        stop_qdrant
        ;;
    restart)
        restart_qdrant
        ;;
    status)
        status_qdrant
        ;;
    logs)
        logs_qdrant
        ;;
    clean)
        clean_storage
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo -e "${RED}Invalid command${NC}"
        echo ""
        show_help
        exit 1
        ;;
esac