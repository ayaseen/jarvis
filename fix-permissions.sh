#!/bin/bash
# fix-permissions.sh - Fix Docker volume permissions and git issues

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}=== JARVIS Permission Fix Script ===${NC}"
echo ""

# Get current user info
CURRENT_USER=$(whoami)
CURRENT_UID=$(id -u)
CURRENT_GID=$(id -g)
CURRENT_GROUP=$(id -gn)

echo -e "${YELLOW}Current user: $CURRENT_USER (UID: $CURRENT_UID, GID: $CURRENT_GID)${NC}"
echo ""

# Step 1: Stop services that use volumes
echo -e "${YELLOW}Step 1: Stopping services to fix permissions...${NC}"
docker-compose stop postgres redis grafana prometheus qdrant

# Step 2: Fix PostgreSQL permissions
echo -e "${YELLOW}Step 2: Fixing PostgreSQL permissions...${NC}"
# PostgreSQL needs to be owned by UID 999 (postgres user in container)
# But we can set group permissions for host access
if [ -d "/mnt/appsdata/jarvis/postgres" ]; then
    sudo chown -R 999:$CURRENT_GID /mnt/appsdata/jarvis/postgres
    sudo chmod -R 775 /mnt/appsdata/jarvis/postgres
    # Add sticky bit to maintain permissions
    sudo chmod g+s /mnt/appsdata/jarvis/postgres
    echo -e "${GREEN}✓ PostgreSQL permissions fixed${NC}"
fi

# Step 3: Fix Redis permissions
echo -e "${YELLOW}Step 3: Fixing Redis permissions...${NC}"
if [ -d "/mnt/appsdata/jarvis/redis" ]; then
    # Redis also runs as UID 999 in Alpine containers
    sudo chown -R 999:$CURRENT_GID /mnt/appsdata/jarvis/redis
    sudo chmod -R 775 /mnt/appsdata/jarvis/redis
    sudo chmod g+s /mnt/appsdata/jarvis/redis
    echo -e "${GREEN}✓ Redis permissions fixed${NC}"
fi

# Step 4: Fix Grafana permissions
echo -e "${YELLOW}Step 4: Fixing Grafana permissions...${NC}"
if [ -d "/mnt/appsdata/jarvis/grafana" ]; then
    # Grafana runs as UID 472 by default
    sudo chown -R 472:$CURRENT_GID /mnt/appsdata/jarvis/grafana
    sudo chmod -R 775 /mnt/appsdata/jarvis/grafana
    sudo chmod g+s /mnt/appsdata/jarvis/grafana
    echo -e "${GREEN}✓ Grafana permissions fixed${NC}"
fi

# Step 5: Fix Prometheus permissions
echo -e "${YELLOW}Step 5: Fixing Prometheus permissions...${NC}"
if [ -d "/mnt/appsdata/jarvis/prometheus" ]; then
    # Prometheus runs as nobody (65534) by default
    sudo chown -R 65534:$CURRENT_GID /mnt/appsdata/jarvis/prometheus
    sudo chmod -R 775 /mnt/appsdata/jarvis/prometheus
    sudo chmod g+s /mnt/appsdata/jarvis/prometheus
    echo -e "${GREEN}✓ Prometheus permissions fixed${NC}"
fi

# Step 6: Fix Qdrant permissions
echo -e "${YELLOW}Step 6: Fixing Qdrant permissions...${NC}"
if [ -d "/mnt/appsdata/jarvis/qdrant" ]; then
    # Qdrant needs consistent permissions
    sudo chown -R $CURRENT_UID:$CURRENT_GID /mnt/appsdata/jarvis/qdrant
    sudo chmod -R 775 /mnt/appsdata/jarvis/qdrant
    echo -e "${GREEN}✓ Qdrant permissions fixed${NC}"
fi

# Step 7: Fix other directories
echo -e "${YELLOW}Step 7: Fixing other directory permissions...${NC}"
for dir in models logs shared documents scripts init-scripts config; do
    if [ -d "/mnt/appsdata/jarvis/$dir" ]; then
        sudo chown -R $CURRENT_UID:$CURRENT_GID /mnt/appsdata/jarvis/$dir
        sudo chmod -R 775 /mnt/appsdata/jarvis/$dir
    fi
done
echo -e "${GREEN}✓ Other directories fixed${NC}"

# Step 8: Create comprehensive .gitignore
echo -e "${YELLOW}Step 8: Creating proper .gitignore...${NC}"
cat > /mnt/appsdata/jarvis/.gitignore << 'EOF'
# Data directories (should never be in git)
postgres/
redis/
qdrant/
grafana/
prometheus/

# Model directories (too large for git)
models/

# Log files
logs/
*.log

# Temporary files
*.tmp
*.temp
*.swp
*.swo
*~

# Environment files with secrets
.env
.env.*
!.env.example

# Docker volumes
volumes/
data/

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/

# IDE
.vscode/
.idea/
*.iml

# OS
.DS_Store
Thumbs.db

# Backup files
*.backup
*.bak

# Build artifacts
build/
dist/
*.egg-info/
EOF

echo -e "${GREEN}✓ .gitignore created${NC}"

# Step 9: Add git safe directory (for permission mismatch)
echo -e "${YELLOW}Step 9: Configuring git safe directory...${NC}"
git config --global --add safe.directory /mnt/appsdata/jarvis
echo -e "${GREEN}✓ Git safe directory configured${NC}"

# Step 10: Clean git cache for ignored files
echo -e "${YELLOW}Step 10: Cleaning git cache...${NC}"
if [ -d ".git" ]; then
    # Remove cached entries for directories that should be ignored
    git rm -r --cached postgres/ 2>/dev/null || true
    git rm -r --cached redis/ 2>/dev/null || true
    git rm -r --cached qdrant/ 2>/dev/null || true
    git rm -r --cached grafana/ 2>/dev/null || true
    git rm -r --cached prometheus/ 2>/dev/null || true
    git rm -r --cached models/ 2>/dev/null || true
    git rm -r --cached logs/ 2>/dev/null || true
    echo -e "${GREEN}✓ Git cache cleaned${NC}"
fi

# Step 11: Create ACL rules for mixed permissions (optional but recommended)
echo -e "${YELLOW}Step 11: Setting up ACL for mixed permissions...${NC}"
if command -v setfacl &> /dev/null; then
    # Give your user full access to all directories regardless of owner
    sudo setfacl -R -m u:$CURRENT_USER:rwx /mnt/appsdata/jarvis
    sudo setfacl -R -d -m u:$CURRENT_USER:rwx /mnt/appsdata/jarvis
    
    # Give group members read/write access
    sudo setfacl -R -m g:$CURRENT_GROUP:rwx /mnt/appsdata/jarvis
    sudo setfacl -R -d -m g:$CURRENT_GROUP:rwx /mnt/appsdata/jarvis
    
    echo -e "${GREEN}✓ ACL rules applied${NC}"
else
    echo -e "${YELLOW}⚠ ACL not available. Install with: sudo apt-get install acl${NC}"
fi

echo ""
echo -e "${GREEN}=== Permission Fix Complete ===${NC}"
echo ""
echo "Summary:"
echo "  • PostgreSQL: owned by 999:$CURRENT_GID (required by container)"
echo "  • Redis: owned by 999:$CURRENT_GID (required by container)"
echo "  • Grafana: owned by 472:$CURRENT_GID (required by container)"
echo "  • Prometheus: owned by 65534:$CURRENT_GID (required by container)"
echo "  • Other dirs: owned by $CURRENT_UID:$CURRENT_GID"
echo ""
echo "Git should now work properly. The data directories are in .gitignore"
echo ""
echo -e "${YELLOW}Restart services with:${NC}"
echo "  docker-compose up -d postgres redis grafana prometheus qdrant"
echo ""
echo -e "${YELLOW}Test git with:${NC}"
echo "  git status"
echo "  git add ."
echo "  git commit -m 'Fixed permissions and updated services'"
