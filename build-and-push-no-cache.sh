#!/bin/bash
# build-and-push.sh

set -e

# Load credentials from .env.registry file
if [ -f .env.registry ]; then
    source .env.registry
else
    echo "ERROR: .env.registry file not found!"
    echo "Create .env.registry with:"
    exit 1
fi

# Private Registry Configuration
REGISTRY="hub.rhlab.dev"
REGISTRY_NAMESPACE="jarvis"
VERSION="${1:-latest}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Building and pushing JARVIS Docker images${NC}"
echo "============================================"
echo "Registry: $REGISTRY/$REGISTRY_NAMESPACE"
echo "Version: $VERSION"
echo "User: $REGISTRY_USER"
echo ""

# Login to private registry
echo -e "${YELLOW}🔐 Logging in to $REGISTRY...${NC}"
echo "$REGISTRY_PASS" | docker login $REGISTRY -u "$REGISTRY_USER" --password-stdin
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Successfully logged in to $REGISTRY${NC}"
else
    echo -e "${RED}❌ Failed to login to $REGISTRY${NC}"
    exit 1
fi
echo ""

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
    
    # Build the image with private registry tag
    if docker build \
        --no-cache \
        -t "$REGISTRY/$REGISTRY_NAMESPACE/$service:$VERSION" \
        -t "$REGISTRY/$REGISTRY_NAMESPACE/$service:latest" \
        "./services/$service"; then
        
        echo -e "${GREEN}✅ Build successful${NC}"
        
        echo -e "${YELLOW}📤 Pushing $service to $REGISTRY...${NC}"
        
        # Push both version tag and latest tag
        docker push "$REGISTRY/$REGISTRY_NAMESPACE/$service:$VERSION"
        docker push "$REGISTRY/$REGISTRY_NAMESPACE/$service:latest"
        
        echo -e "${GREEN}✅ $service complete!${NC}"
    else
        echo -e "${RED}❌ Failed to build $service${NC}"
        exit 1
    fi
    
    echo ""
done

# Logout for security
docker logout $REGISTRY
echo -e "${YELLOW}🔒 Logged out from $REGISTRY${NC}"

echo ""
echo -e "${GREEN}🎉 All images built and pushed successfully!${NC}"
echo ""
echo "📋 Images available:"
for service in "${services[@]}"; do
    echo "   - $REGISTRY/$REGISTRY_NAMESPACE/$service:$VERSION"
    echo "   - $REGISTRY/$REGISTRY_NAMESPACE/$service:latest"
done

echo ""
echo -e "${GREEN}🚀 To deploy, run:${NC}"
echo "   ./setup-volumes.sh  # First time only"
echo "   docker-compose up -d"
