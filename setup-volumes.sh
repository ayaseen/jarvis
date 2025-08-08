#!/bin/bash
# setup-volumes.sh

set -e

echo "🚀 Setting up JARVIS volumes and permissions"
echo "==========================================="

# Get current user info
CURRENT_UID=$(id -u)
CURRENT_GID=$(getent group appsdata | cut -d: -f3)

echo "Current UID: $CURRENT_UID"
echo "Current GID: $CURRENT_GID"

# Base directory
BASE_DIR="/mnt/appsdata/jarvis"

# Create all required directories
echo "📁 Creating directory structure..."
sudo mkdir -p $BASE_DIR/{models,data,logs,documents,grafana,prometheus,shared}
sudo mkdir -p $BASE_DIR/models/{ollama,whisper,tts,transformers}
sudo mkdir -p $BASE_DIR/logs/{llm,rag,voice,orchestrator,nginx}
sudo mkdir -p $BASE_DIR/{postgres,redis,qdrant}
sudo mkdir -p $BASE_DIR/config/grafana/provisioning/{dashboards,datasources}
sudo mkdir -p $BASE_DIR/scripts
sudo mkdir -p $BASE_DIR/init-scripts

# Set ownership
echo "🔐 Setting permissions..."
sudo chown -R $CURRENT_UID:$CURRENT_GID $BASE_DIR
sudo chmod -R 775 $BASE_DIR

# Create .env file with proper UID/GID
echo "📝 Creating .env file..."
cat > .env << EOF
# User and Group IDs
PUID=$CURRENT_UID
PGID=$CURRENT_GID
TZ=$(timedatectl show -p Timezone --value 2>/dev/null || echo "UTC")

# GPU Configuration
CUDA_VISIBLE_DEVICES=0
NVIDIA_VISIBLE_DEVICES=all
NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Model Configuration
LLM_MODEL=mixtral:8x7b-instruct-v0.1-q4_K_M
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
WHISPER_MODEL=base.en
TTS_MODEL=tts_models/en/ljspeech/tacotron2-DDC

# Database Configuration
POSTGRES_DB=jarvis
POSTGRES_USER=jarvis
POSTGRES_PASSWORD=jarvis123

# Monitoring
GRAFANA_PASSWORD=admin
EOF

# Create PostgreSQL init script
echo "📄 Creating PostgreSQL init script..."
sudo tee $BASE_DIR/init-scripts/init.sql > /dev/null << 'EOF'
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id VARCHAR(255),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    user_message TEXT,
    assistant_response TEXT,
    context JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_session ON conversations(session_id);
CREATE INDEX idx_timestamp ON conversations(timestamp);
CREATE INDEX idx_created_at ON conversations(created_at);

CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filename VARCHAR(255),
    content TEXT,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_filename ON documents(filename);
CREATE INDEX idx_created_at_docs ON documents(created_at);
EOF

# Create Prometheus config
echo "📊 Creating Prometheus configuration..."
sudo tee $BASE_DIR/config/prometheus.yml > /dev/null << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'llm-service'
    static_configs:
      - targets: ['llm-service:8001']
    metrics_path: '/metrics'
  
  - job_name: 'rag-service'
    static_configs:
      - targets: ['rag-service:8002']
    metrics_path: '/metrics'
  
  - job_name: 'voice-service'
    static_configs:
      - targets: ['voice-service:8003']
    metrics_path: '/metrics'
  
  - job_name: 'orchestrator'
    static_configs:
      - targets: ['orchestrator:8000']
    metrics_path: '/metrics'
EOF

# Set final permissions
sudo chown -R $CURRENT_UID:$CURRENT_GID $BASE_DIR

echo "✅ Setup complete!"
echo ""
echo "📍 Volume structure created at: $BASE_DIR"
echo "🔑 Permissions set to UID:$CURRENT_UID GID:$CURRENT_GID"
echo ""
echo "Next steps:"
echo "1. Build and push images: ./build-and-push.sh"
echo "2. Start services: docker-compose up -d"
echo "3. Check logs: docker-compose logs -f"
