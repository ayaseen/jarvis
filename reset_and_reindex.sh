#!/bin/bash
# reset_and_reindex.sh - Complete reset of RAG system

echo "=========================================="
echo "COMPLETE RAG SYSTEM RESET"
echo "=========================================="

# 1. Stop all services
echo -e "\n1. Stopping all services..."
docker-compose down

# 2. Clear all data
echo -e "\n2. Clearing all data..."
sudo rm -rf /mnt/appsdata/jarvis/milvus/*
sudo rm -rf /mnt/appsdata/jarvis/documents/*
sudo rm -rf /mnt/appsdata/jarvis/redis/*

# 3. Start infrastructure
echo -e "\n3. Starting infrastructure..."
docker-compose up -d postgres redis etcd minio

# Wait for infrastructure
echo "   Waiting for infrastructure..."
sleep 20

# 4. Start Milvus
echo -e "\n4. Starting Milvus..."
docker-compose up -d milvus

# Wait for Milvus to be fully ready
echo "   Waiting for Milvus to initialize..."
for i in {1..30}; do
    if curl -s http://localhost:19530 > /dev/null 2>&1; then
        echo "   ✓ Milvus is ready!"
        break
    fi
    echo "   Waiting... ($i/30)"
    sleep 2
done

# 5. Start vLLM with proper settings
echo -e "\n5. Starting vLLM..."
docker-compose up -d vllm

# Wait for vLLM
echo "   Waiting for vLLM to load model..."
for i in {1..60}; do
    if curl -s http://localhost:8080/health > /dev/null 2>&1; then
        echo "   ✓ vLLM is ready!"
        break
    fi
    echo "   Waiting... ($i/60)"
    sleep 2
done

# 6. Start RAG service
echo -e "\n6. Starting RAG service..."
docker-compose up -d rag-service

# Wait for RAG to initialize
echo "   Waiting for RAG service..."
sleep 10

# Check RAG health
echo -e "\n7. Checking RAG health..."
curl -s http://localhost:8002/health | jq '.status, .milvus_status'

# 8. Start remaining services
echo -e "\n8. Starting remaining services..."
docker-compose up -d

# 9. Final health check
echo -e "\n9. Final system health check..."
sleep 5
curl -s http://localhost:3000/api/health | jq

echo -e "\n=========================================="
echo "RESET COMPLETE"
echo "=========================================="
echo ""
echo "Now you can:"
echo "1. Go to http://localhost:3000/document.html"
echo "2. Upload your Amjad Yaseen document"
echo "3. Test the search"
echo ""
echo "To debug, run:"
echo "  python3 debug_milvus.py"
echo "  python3 test_rag.py"
