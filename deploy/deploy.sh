#!/bin/bash
# =============================================================================
# Deploy Script for Face ID (Employee Attendance System)
# Server: 10.10.0.129
#
# Usage: bash deploy/deploy.sh
# =============================================================================

set -e  # Exit on any error

# Auto-detect project directory from script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="${APP_DIR}/venv"

echo "  Project path: ${APP_DIR}"
echo "  Running as:   $(whoami)"

echo "=========================================="
echo "  Face ID — Deployment Script"
echo "=========================================="
echo ""

# ----- Step 1: System Dependencies -----
echo "[1/6] Installing system dependencies..."
sudo apt update -qq
sudo apt install -y -qq nginx python3-venv python3-pip cmake build-essential \
    libgl1-mesa-glx libglib2.0-0 > /dev/null 2>&1
echo "  ✓ System dependencies installed"

# ----- Step 2: Python Virtual Environment -----
echo "[2/6] Setting up Python environment..."
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    echo "  ✓ Virtual environment created"
fi
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip -q
pip install -r "${APP_DIR}/requirements.txt" -q
echo "  ✓ Python dependencies installed"

# ----- Step 3: Environment File -----
echo "[3/6] Setting up environment..."
if [ ! -f "${APP_DIR}/.env" ]; then
    cp "${APP_DIR}/.env.example" "${APP_DIR}/.env"
    
    # Generate a secure SECRET_KEY
    SECRET=$(python3 -c "import secrets; print(secrets.token_hex(32))")
    sed -i "s/change-me-in-production-use-a-strong-random-key/${SECRET}/" "${APP_DIR}/.env"
    
    # Set ENV to production
    sed -i "s/^ENV=dev/ENV=prod/" "${APP_DIR}/.env"
    
    echo "  ✓ .env created with secure SECRET_KEY"
    echo "  ⚠  EDIT .env to set your CAMERA_ID/RTSP_URL and other settings!"
else
    echo "  ✓ .env already exists (not overwritten)"
fi

# ----- Step 4: Data Directories -----
echo "[4/6] Creating data directories..."
mkdir -p "${APP_DIR}/data/images"
mkdir -p "${APP_DIR}/data/attendance_snapshots"
echo "  ✓ Data directories ready"

# ----- Step 5: Nginx -----
echo "[5/6] Configuring Nginx..."
# Update paths in nginx.conf to match actual deployment path
sed "s|/home/suxrob/Documents/face_id|${APP_DIR}|g" \
    "${APP_DIR}/deploy/nginx.conf" | sudo tee /etc/nginx/sites-available/face_id > /dev/null

sudo ln -sf /etc/nginx/sites-available/face_id /etc/nginx/sites-enabled/face_id
sudo rm -f /etc/nginx/sites-enabled/default

# Test nginx config
if sudo nginx -t 2>&1; then
    sudo systemctl reload nginx
    echo "  ✓ Nginx configured and reloaded"
else
    echo "  ✗ Nginx config test failed! Check /etc/nginx/sites-available/face_id"
    exit 1
fi

# ----- Step 6: Systemd Service -----
echo "[6/6] Setting up systemd service..."
# Update paths in service file
sed "s|/home/suxrob/Documents/face_id|${APP_DIR}|g; s|User=suxrob|User=$(whoami)|g; s|Group=suxrob|Group=$(whoami)|g" \
    "${APP_DIR}/deploy/face_id.service" | sudo tee /etc/systemd/system/face_id.service > /dev/null

sudo systemctl daemon-reload
sudo systemctl enable face_id
sudo systemctl restart face_id
echo "  ✓ face_id service started"

echo ""
echo "=========================================="
echo "  ✓ Deployment Complete!"
echo "=========================================="
echo ""
echo "  App URL:     http://10.10.0.129"
echo "  App Status:  sudo systemctl status face_id"
echo "  App Logs:    sudo journalctl -u face_id -f"
echo ""
echo "  ⚠  Don't forget to:"
echo "     1. Edit .env with your actual CAMERA_ID / RTSP_URL"
echo "     2. Restart after changes: sudo systemctl restart face_id"
echo ""
echo "  Optional — Enable HTTPS (self-signed):"
echo "     sudo mkdir -p /etc/nginx/ssl"
echo "     sudo openssl req -x509 -nodes -days 365 -newkey rsa:2048 \\"
echo "       -keyout /etc/nginx/ssl/face_id.key \\"
echo "       -out /etc/nginx/ssl/face_id.crt \\"
echo "       -subj '/CN=10.10.0.129'"
echo "     Then uncomment the HTTPS block in /etc/nginx/sites-available/face_id"
echo "     sudo systemctl reload nginx"
echo ""
