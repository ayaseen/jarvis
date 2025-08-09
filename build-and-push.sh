#!/bin/bash
# build-and-push.sh

set -e

# Docker Hub username
DOCKER_USER="felix971"
VERSION="${1:-latest}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Building and pushing JARVIS Docker images${NC}"
echo "============================================"
echo "Registry: $DOCKER_USER"
echo "Version: $VERSION"
echo ""

# Check if logged in to Docker Hub
if ! docker info 2>/dev/null | grep -q "Username: $DOCKER_USER"; then
    echo -e "${YELLOW}⚠️  Not logged in to Docker Hub${NC}"
    docker login -u $DOCKER_USER
fi

# Array of services to build
services=(
    "llm"
    "rag"
    "voice"
    "orchestrator"
    "web"
)

# Build and push each service
for service in "${services[@]}"; do
    echo -e "${YELLOW}📦 Building $service...${NC}"
    
    # Build the image
    if docker build \
        -t "$DOCKER_USER/jarvis-$service:$VERSION" \
        "./services/$service"; then
        
        echo -e "${GREEN}✅ Build successful${NC}"
        
        echo -e "${YELLOW}📤 Pushing $service to Docker Hub...${NC}"
        
        # Push both tags
        docker push "$DOCKER_USER/jarvis-$service:$VERSION"
        
        echo -e "${GREEN}✅ $service complete!${NC}"
    else
        echo -e "${RED}❌ Failed to build $service${NC}"
        exit 1
    fi
    
    echo ""
done

echo -e "${GREEN}🎉 All images built and pushed successfully!${NC}"
echo ""
echo "📋 Images available:"
for service in "${services[@]}"; do
    echo "   - $DOCKER_USER/jarvis-$service:$VERSION"
done

echo ""
echo -e "${GREEN}🚀 To deploy, run:${NC}"
echo "   ./setup-volumes.sh  # First time only"
echo "   docker-compose up -d"
