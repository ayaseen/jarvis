#!/bin/bash
# JARVIS Comprehensive Diagnostic Script
# Save as: diagnose-jarvis.sh

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🔍 JARVIS System Diagnostic${NC}"
echo "=============================="
echo ""

# Function to check service
check_service() {
    local name=$1
    local url=$2
    echo -n "Checking $name... "
    if curl -s -f -o /dev/null "$url" 2>/dev/null; then
        echo -e "${GREEN}✓ OK${NC}"
        return 0
    else
        echo -e "${RED}✗ FAILED${NC}"
        return 1
    fi
}

# Function to check container
check_container() {
    local name=$1
    echo -n "Container $name: "
    if docker ps | grep -q "$name"; then
        status=$(docker inspect -f '{{.State.Status}}' "$name" 2>/dev/null || echo "unknown")
        if [ "$status" = "running" ]; then
            echo -e "${GREEN}✓ Running${NC}"
        else
            echo -e "${YELLOW}⚠ Status: $status${NC}"
        fi
    else
        echo -e "${RED}✗ Not running${NC}"
    fi
}

echo "1. Checking Docker Containers"
echo "-----------------------------"
check_container "jarvis-orchestrator"
check_container "jarvis-llm"
check_container "jarvis-rag"
check_container "jarvis-voice"
check_container "jarvis-web"
check_container "jarvis-ollama"
check_container "jarvis-redis"
check_container "jarvis-qdrant"
check_container "jarvis-postgres"
echo ""

echo "2. Checking Service Health Endpoints"
echo "------------------------------------"
check_service "Orchestrator" "http://localhost:8000/health"
check_service "LLM Service" "http://localhost:8001/health"
check_service "RAG Service" "http://localhost:8002/health"
check_service "Voice Service" "http://localhost:8003/health"
check_service "Web Interface" "http://localhost:3000/"
echo ""

echo "3. Checking Infrastructure Services"
echo "-----------------------------------"
echo -n "Redis: "
if docker exec jarvis-redis redis-cli ping 2>/dev/null | grep -q PONG; then
    echo -e "${GREEN}✓ Connected${NC}"
else
    echo -e "${RED}✗ Not responding${NC}"
fi

echo -n "PostgreSQL: "
if docker exec jarvis-postgres pg_isready -U jarvis 2>/dev/null | grep -q "accepting connections"; then
    echo -e "${GREEN}✓ Ready${NC}"
else
    echo -e "${RED}✗ Not ready${NC}"
fi

echo -n "Qdrant: "
if curl -s http://localhost:6333/collections >/dev/null 2>&1; then
    echo -e "${GREEN}✓ Accessible${NC}"
else
    echo -e "${RED}✗ Not accessible${NC}"
fi

echo -n "Ollama: "
if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo -e "${GREEN}✓ Running${NC}"
    # Check for model
    echo -n "  Model (mixtral): "
    if docker exec jarvis-ollama ollama list 2>/dev/null | grep -q "mixtral"; then
        echo -e "${GREEN}✓ Loaded${NC}"
    else
        echo -e "${YELLOW}⚠ Not loaded (run: docker exec jarvis-ollama ollama pull mixtral:8x7b-instruct-v0.1-q4_K_M)${NC}"
    fi
else
    echo -e "${RED}✗ Not running${NC}"
fi
echo ""

echo "4. Testing API Endpoints"
echo "-----------------------"

# Test orchestrator root
echo -n "Orchestrator API: "
response=$(curl -s http://localhost:8000/ 2>/dev/null || echo "{}")
if echo "$response" | grep -q "JARVIS Orchestrator"; then
    echo -e "${GREEN}✓ Responding${NC}"
else
    echo -e "${RED}✗ Not responding${NC}"
fi

# Test WebSocket upgrade
echo -n "WebSocket endpoint: "
ws_response=$(curl -s -o /dev/null -w "%{http_code}" \
    -H "Connection: Upgrade" \
    -H "Upgrade: websocket" \
    -H "Sec-WebSocket-Version: 13" \
    -H "Sec-WebSocket-Key: test" \
    http://localhost:8000/ws 2>/dev/null)
if [ "$ws_response" = "101" ]; then
    echo -e "${GREEN}✓ WebSocket upgrade working${NC}"
else
    echo -e "${YELLOW}⚠ WebSocket status: $ws_response${NC}"
fi
echo ""

echo "5. Recent Error Logs"
echo "-------------------"
echo "Orchestrator errors (last 5):"
docker logs jarvis-orchestrator 2>&1 | grep -i error | tail -5 || echo "  No errors found"
echo ""
echo "LLM Service errors (last 5):"
docker logs jarvis-llm 2>&1 | grep -i error | tail -5 || echo "  No errors found"
echo ""

echo "6. Network Connectivity"
echo "----------------------"
echo -n "Inter-service communication (orchestrator -> llm): "
if docker exec jarvis-orchestrator curl -s http://llm-service:8001/health >/dev/null 2>&1; then
    echo -e "${GREEN}✓ OK${NC}"
else
    echo -e "${RED}✗ Failed${NC}"
fi

echo -n "Inter-service communication (orchestrator -> rag): "
if docker exec jarvis-orchestrator curl -s http://rag-service:8002/health >/dev/null 2>&1; then
    echo -e "${GREEN}✓ OK${NC}"
else
    echo -e "${RED}✗ Failed${NC}"
fi
echo ""

echo "7. Quick Fix Suggestions"
echo "------------------------"
issues_found=false

# Check if Ollama model is loaded
if ! docker exec jarvis-ollama ollama list 2>/dev/null | grep -q "mixtral"; then
    echo -e "${YELLOW}⚠ Ollama model not loaded. Run:${NC}"
    echo "  docker exec jarvis-ollama ollama pull mixtral:8x7b-instruct-v0.1-q4_K_M"
    issues_found=true
fi

# Check if RAG service is on wrong port
if docker logs jarvis-rag 2>&1 | grep -q "port 8001"; then
    echo -e "${YELLOW}⚠ RAG service running on wrong port. Rebuild with:${NC}"
    echo "  docker-compose stop rag-service"
    echo "  docker-compose build rag-service"
    echo "  docker-compose up -d rag-service"
    issues_found=true
fi

if [ "$issues_found" = false ]; then
    echo -e "${GREEN}✓ No immediate issues detected${NC}"
fi
echo ""

echo "8. Test Message Send"
echo "-------------------"
echo "Attempting to send test message..."
response=$(curl -s -X POST "http://localhost:8000/chat?message=Hello%20JARVIS" 2>/dev/null || echo "{}")
if echo "$response" | grep -q "session_id"; then
    echo -e "${GREEN}✓ Chat endpoint working${NC}"
    echo "Response preview: $(echo "$response" | head -c 100)..."
else
    echo -e "${RED}✗ Chat endpoint not responding properly${NC}"
    echo "Response: $response"
fi

echo ""
echo "=============================="
echo -e "${GREEN}Diagnostic Complete${NC}"
echo ""
echo "For detailed logs, run:"
echo "  docker-compose logs -f [service-name]"
echo ""
echo "To restart all services:"
echo "  docker-compose restart"
