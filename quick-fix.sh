#!/bin/bash
# JARVIS Quick Fix Script - Fixes all port issues and restarts services
# Save as: quick-fix.sh

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 JARVIS Quick Fix - Fixing all issues${NC}"
echo "========================================"
echo ""

# Step 1: Check why services are failing
echo -e "${YELLOW}Step 1: Diagnosing failures...${NC}"
echo ""

echo "Voice Service last error:"
docker logs jarvis-voice --tail 5 2>&1 | grep -iE "error|exception|traceback" || echo "No obvious errors"
echo ""

echo "Qdrant last error:"
docker logs jarvis-qdrant --tail 5 2>&1 | grep -iE "error|exception|permission" || echo "No obvious errors"
echo ""

echo "RAG Service last error:"
docker logs jarvis-rag --tail 5 2>&1 | grep -iE "error|exception|traceback" || echo "No obvious errors"
echo ""

# Step 2: Fix permissions for Qdrant
echo -e "${YELLOW}Step 2: Fixing Qdrant permissions...${NC}"
sudo mkdir -p /mnt/appsdata/jarvis/qdrant
sudo chown -R 1000:1000 /mnt/appsdata/jarvis/qdrant
sudo chmod -R 775 /mnt/appsdata/jarvis/qdrant
echo -e "${GREEN}✓ Qdrant permissions fixed${NC}"
echo ""

# Step 3: Clear any corrupted Qdrant data if needed
echo -e "${YELLOW}Step 3: Checking Qdrant data...${NC}"
if [ -f "/mnt/appsdata/jarvis/qdrant/storage/raft_state.bin" ]; then
    echo "Qdrant data exists. If corrupted, you may need to clear it."
    read -p "Clear Qdrant data? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        sudo rm -rf /mnt/appsdata/jarvis/qdrant/*
        echo -e "${GREEN}✓ Qdrant data cleared${NC}"
    fi
else
    echo "No existing Qdrant data found."
fi
echo ""

# Step 4: Stop all services
echo -e "${YELLOW}Step 4: Stopping services...${NC}"
docker-compose down
echo -e "${GREEN}✓ Services stopped${NC}"
echo ""

# Step 5: Apply all Dockerfile fixes
echo -e "${YELLOW}Step 5: Applying Dockerfile fixes...${NC}"

# Fix RAG Dockerfile
cat > services/rag/Dockerfile << 'EOF'
FROM nvidia/cuda:12.9.1-runtime-ubuntu24.04
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv python3-dev \
    build-essential libpq-dev git curl \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
RUN python -m pip install --upgrade pip setuptools wheel
WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
COPY app.py .
EXPOSE 8002
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8002", "--workers", "1"]
EOF

# Simplified Voice Dockerfile without pre-downloading model
cat > services/voice/Dockerfile << 'EOF'
FROM nvidia/cuda:12.9.1-runtime-ubuntu24.04
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv python3-dev \
    build-essential libsndfile1 ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
RUN python -m pip install --upgrade pip setuptools wheel
WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
COPY app.py .
EXPOSE 8003
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8003", "--workers", "1"]
EOF

echo -e "${GREEN}✓ Dockerfiles fixed${NC}"
echo ""

# Step 6: Rebuild only the affected services
echo -e "${YELLOW}Step 6: Rebuilding affected services...${NC}"
docker-compose build rag-service voice-service
echo -e "${GREEN}✓ Services rebuilt${NC}"
echo ""

# Step 7: Start infrastructure first
echo -e "${YELLOW}Step 7: Starting infrastructure services...${NC}"
docker-compose up -d postgres redis ollama
sleep 5
echo -e "${GREEN}✓ Infrastructure started${NC}"
echo ""

# Step 8: Start Qdrant with clean state
echo -e "${YELLOW}Step 8: Starting Qdrant...${NC}"
docker-compose up -d qdrant
sleep 10
# Verify Qdrant is running
if curl -s http://localhost:6333/collections > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Qdrant started successfully${NC}"
else
    echo -e "${RED}✗ Qdrant failed to start. Checking logs...${NC}"
    docker logs jarvis-qdrant --tail 20
fi
echo ""

# Step 9: Start application services
echo -e "${YELLOW}Step 9: Starting application services...${NC}"
docker-compose up -d llm-service rag-service voice-service orchestrator web
echo -e "${GREEN}✓ Application services started${NC}"
echo ""

# Step 10: Load Ollama model if needed
echo -e "${YELLOW}Step 10: Checking Ollama model...${NC}"
if ! docker exec jarvis-ollama ollama list 2>/dev/null | grep -q "mixtral"; then
    echo "Mixtral model not found. Pulling (this will take time)..."
    docker exec jarvis-ollama ollama pull mixtral:8x7b-instruct-v0.1-q4_K_M &
    echo "Model pull started in background. Check progress with:"
    echo "  docker exec jarvis-ollama ollama list"
else
    echo -e "${GREEN}✓ Mixtral model already loaded${NC}"
fi
echo ""

# Step 11: Wait for services to stabilize
echo -e "${YELLOW}Step 11: Waiting for services to stabilize...${NC}"
sleep 20
echo ""

# Step 12: Final health check
echo -e "${YELLOW}Step 12: Running health checks...${NC}"
echo ""

services=(
    "Orchestrator:8000"
    "LLM:8001"
    "RAG:8002"
    "Voice:8003"
)

all_healthy=true
for service_port in "${services[@]}"; do
    IFS=':' read -r service port <<< "$service_port"
    echo -n "$service Service: "
    if curl -s -f http://localhost:$port/health > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Healthy${NC}"
    else
        echo -e "${RED}✗ Not responding${NC}"
        all_healthy=false
    fi
done

echo ""
if [ "$all_healthy" = true ]; then
    echo -e "${GREEN}🎉 All services are healthy!${NC}"
    echo ""
    echo "Test JARVIS with:"
    echo "  curl -X POST 'http://localhost:8000/chat?message=Hello%20JARVIS'"
    echo ""
    echo "Or open web interface:"
    echo "  http://localhost:3000"
else
    echo -e "${YELLOW}⚠ Some services are not healthy. Check logs:${NC}"
    echo "  docker-compose logs -f [service-name]"
fi

echo ""
echo "Monitor all logs with:"
echo "  docker-compose logs -f"
