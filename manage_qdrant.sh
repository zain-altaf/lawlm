#!/bin/bash

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
        echo -e "${RED}Docker is not installed. Please install Docker first.${NC}"
        exit 1
    fi
}

# Function to check if Qdrant is running
is_running() {
    docker ps | grep qdrant/qdrant > /dev/null 2>&1
    return $?
}

# Start Qdrant
start_qdrant() {
    check_docker
    
    if is_running; then
        echo -e "${YELLOW}Qdrant is already running${NC}"
        return
    fi
    
    echo -e "${GREEN}Starting Qdrant...${NC}"
    
    # Create storage directory if it doesn't exist
    mkdir -p "$QDRANT_STORAGE"
    
    # Run Qdrant container
    docker run -d \
        --name qdrant \
        -p ${QDRANT_PORT}:6333 \
        -v $(pwd)/qdrant_storage:/qdrant/storage \
        qdrant/qdrant:${QDRANT_VERSION}
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Qdrant started successfully on port ${QDRANT_PORT}${NC}"
        echo -e "${GREEN}Dashboard available at: http://localhost:${QDRANT_PORT}/dashboard${NC}"
    else
        echo -e "${RED}Failed to start Qdrant${NC}"
        exit 1
    fi
}

# Stop Qdrant
stop_qdrant() {
    check_docker
    
    if ! is_running; then
        echo -e "${YELLOW}Qdrant is not running${NC}"
        return
    fi
    
    echo -e "${GREEN}Stopping Qdrant...${NC}"
    docker stop qdrant
    docker rm qdrant
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Qdrant stopped successfully${NC}"
    else
        echo -e "${RED}Failed to stop Qdrant${NC}"
        exit 1
    fi
}

# Restart Qdrant
restart_qdrant() {
    stop_qdrant
    sleep 2
    start_qdrant
}

# Check Qdrant status
status_qdrant() {
    check_docker
    
    if is_running; then
        echo -e "${GREEN}Qdrant is running${NC}"
        
        # Check health endpoint
        if curl -s http://localhost:${QDRANT_PORT}/health > /dev/null; then
            echo -e "${GREEN}Qdrant health check: OK${NC}"
        else
            echo -e "${YELLOW}Qdrant is running but health check failed${NC}"
        fi
        
        # Show container info
        echo -e "\n${GREEN}Container Info:${NC}"
        docker ps | grep qdrant/qdrant
    else
        echo -e "${RED}Qdrant is not running${NC}"
    fi
}

# View Qdrant logs
logs_qdrant() {
    check_docker
    
    if ! is_running; then
        echo -e "${RED}Qdrant is not running${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}Showing Qdrant logs (Ctrl+C to exit)...${NC}"
    docker logs -f qdrant
}

# Clean Qdrant storage
clean_storage() {
    if is_running; then
        echo -e "${RED}Please stop Qdrant before cleaning storage${NC}"
        exit 1
    fi
    
    echo -e "${YELLOW}This will delete all data in Qdrant storage!${NC}"
    read -p "Are you sure? (y/N) " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$QDRANT_STORAGE"
        mkdir -p "$QDRANT_STORAGE"
        echo -e "${GREEN}Storage cleaned successfully${NC}"
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