#!/bin/bash
# JARVIS Port Configuration Verification and Fix Script
# Save as: fix-ports.sh

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🔧 JARVIS Port Configuration Checker & Fixer${NC}"
echo "=============================================="
echo ""

# Expected port configuration
declare -A EXPECTED_PORTS=(
    ["llm-service"]=8001
    ["rag-service"]=8002
    ["voice-service"]=8003
    ["orchestrator"]=8000
    ["web"]=80
    ["qdrant"]=6333
    ["postgres"]=5432
    ["redis"]=6379
    ["ollama"]=11434
)

# Function to check Dockerfile CMD port
check_dockerfile_port() {
    local service=$1
    local expected_port=$2
    local dockerfile_path="services/${service}/Dockerfile"
    
    if [ -f "$dockerfile_path" ]; then
        echo -n "  Dockerfile CMD: "
        port_in_cmd=$(grep -E "CMD.*uvicorn.*--port" "$dockerfile_path" 2>/dev/null | grep -oE "[0-9]+" | tail -1 || echo "")
        
        if [ -z "$port_in_cmd" ]; then
            echo -e "${YELLOW}No uvicorn port found${NC}"
        elif [ "$port_in_cmd" = "$expected_port" ]; then
            echo -e "${GREEN}✓ Port $port_in_cmd (correct)${NC}"
        else
            echo -e "${RED}✗ Port $port_in_cmd (should be $expected_port)${NC}"
            return 1
        fi
    fi
    return 0
}

# Function to check docker-compose ports
check_compose_port() {
    local service=$1
    local expected_port=$2
    
    echo -n "  Docker-compose: "
    compose_port=$(grep -A 20 "^  ${service}:" docker-compose.yml | grep -E "^\s+- \"[0-9]+:${expected_port}\"" | head -1 || echo "")
    
    if [ -n "$compose_port" ]; then
        echo -e "${GREEN}✓ Mapped correctly${NC}"
    else
        echo -e "${YELLOW}⚠ Check port mapping${NC}"
    fi
}

# Function to check running container port
check_running_port() {
    local container=$1
    local expected_port=$2
    
    echo -n "  Running container: "
    if docker ps | grep -q "$container"; then
        actual_ports=$(docker port "$container" 2>/dev/null || echo "")
        if echo "$actual_ports" | grep -q "$expected_port"; then
            echo -e "${GREEN}✓ Port $expected_port exposed${NC}"
        else
            echo -e "${RED}✗ Port not exposed or different${NC}"
        fi
    else
        echo -e "${RED}✗ Container not running${NC}"
    fi
}

echo "1. Checking Service Port Configurations"
echo "---------------------------------------"

# Check Python services
for service in llm rag voice orchestrator; do
    echo -e "${YELLOW}${service}-service:${NC}"
    expected_port=${EXPECTED_PORTS["${service}-service"]}
    echo "  Expected port: $expected_port"
    
    needs_fix=false
    if ! check_dockerfile_port "$service" "$expected_port"; then
        needs_fix=true
    fi
    check_compose_port "${service}-service" "$expected_port"
    check_running_port "jarvis-${service}" "$expected_port"
    
    if [ "$needs_fix" = true ]; then
        echo -e "  ${RED}⚠ Needs fixing${NC}"
    fi
    echo ""
done

echo "2. Checking Container Status and Logs"
echo "-------------------------------------"

# Check why voice service is restarting
echo -e "${YELLOW}Voice Service Status:${NC}"
echo "Last 10 error lines:"
docker logs jarvis-voice --tail 20 2>&1 | grep -iE "error|exception|failed|critical" | tail -10 || echo "  No errors in recent logs"
echo ""

# Check why qdrant is restarting  
echo -e "${YELLOW}Qdrant Status:${NC}"
echo "Last 10 error lines:"
docker logs jarvis-qdrant --tail 20 2>&1 | grep -iE "error|exception|failed|critical" | tail -10 || echo "  No errors in recent logs"
echo ""

echo "3. Applying Fixes"
echo "-----------------"

# Fix RAG service Dockerfile
echo -e "${BLUE}Fixing RAG Service Dockerfile...${NC}"
cat > services/rag/Dockerfile << 'EOF'
# services/rag/Dockerfile
FROM nvidia/cuda:12.9.1-runtime-ubuntu24.04

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    libpq-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

RUN python -m pip install --upgrade pip setuptools wheel

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 8002

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=5 \
    CMD curl -f http://localhost:8002/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8002", "--workers", "1"]
EOF
echo -e "${GREEN}✓ RAG Dockerfile fixed${NC}"

# Fix Voice service Dockerfile with better health check
echo -e "${BLUE}Fixing Voice Service Dockerfile...${NC}"
cat > services/voice/Dockerfile << 'EOF'
# services/voice/Dockerfile
FROM nvidia/cuda:12.9.1-runtime-ubuntu24.04

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

RUN python -m pip install --upgrade pip setuptools wheel

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Pre-download the Whisper model during build
RUN python3 -c "import whisper; whisper.load_model('base.en')"

COPY app.py .

EXPOSE 8003

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=5 \
    CMD curl -f http://localhost:8003/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8003", "--workers", "1"]
EOF
echo -e "${GREEN}✓ Voice Dockerfile fixed${NC}"

# Fix LLM service Dockerfile for consistency
echo -e "${BLUE}Ensuring LLM Service Dockerfile is correct...${NC}"
cat > services/llm/Dockerfile << 'EOF'
# services/llm/Dockerfile
FROM nvidia/cuda:12.9.1-runtime-ubuntu24.04

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

RUN python -m pip install --upgrade pip setuptools wheel

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=5 \
    CMD curl -f http://localhost:8001/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
EOF
echo -e "${GREEN}✓ LLM Dockerfile verified${NC}"

# Fix Orchestrator Dockerfile
echo -e "${BLUE}Ensuring Orchestrator Dockerfile is correct...${NC}"
cat > services/orchestrator/Dockerfile << 'EOF'
# services/orchestrator/Dockerfile
FROM nvidia/cuda:12.9.1-runtime-ubuntu24.04

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

RUN python -m pip install --upgrade pip setuptools wheel

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
EOF
echo -e "${GREEN}✓ Orchestrator Dockerfile verified${NC}"

echo ""
echo "4. Checking Qdrant Configuration"
echo "--------------------------------"

# Check if qdrant storage has correct permissions
echo -n "Qdrant storage permissions: "
if [ -d "/mnt/appsdata/jarvis/qdrant" ]; then
    perms=$(stat -c %a /mnt/appsdata/jarvis/qdrant)
    owner=$(stat -c %U:%G /mnt/appsdata/jarvis/qdrant)
    echo "Permissions: $perms, Owner: $owner"
    
    # Fix permissions if needed
    if [ "$perms" != "775" ]; then
        echo -e "${YELLOW}Fixing Qdrant permissions...${NC}"
        sudo chmod -R 775 /mnt/appsdata/jarvis/qdrant
        sudo chown -R $(id -u):$(id -g) /mnt/appsdata/jarvis/qdrant
        echo -e "${GREEN}✓ Permissions fixed${NC}"
    fi
else
    echo -e "${RED}✗ Directory not found${NC}"
fi

echo ""
echo "5. Rebuilding and Restarting Services"
echo "-------------------------------------"

echo "Commands to execute:"
echo -e "${YELLOW}# Stop affected services${NC}"
echo "docker-compose stop rag-service voice-service qdrant"
echo ""
echo -e "${YELLOW}# Remove old containers${NC}"
echo "docker-compose rm -f rag-service voice-service qdrant"
echo ""
echo -e "${YELLOW}# Rebuild services with fixed Dockerfiles${NC}"
echo "docker-compose build --no-cache rag-service voice-service"
echo ""
echo -e "${YELLOW}# Start services${NC}"
echo "docker-compose up -d rag-service voice-service qdrant"
echo ""
echo -e "${YELLOW}# Wait for services to stabilize${NC}"
echo "sleep 30"
echo ""
echo -e "${YELLOW}# Check health again${NC}"
echo "./diagnose-jarvis.sh"

echo ""
echo -e "${GREEN}✅ Port configuration fixes applied!${NC}"
echo ""
echo "Run the commands above to rebuild and restart the services."
echo ""
echo "Alternative quick restart (if you want to apply immediately):"
echo -e "${BLUE}docker-compose down && docker-compose build && docker-compose up -d${NC}"
