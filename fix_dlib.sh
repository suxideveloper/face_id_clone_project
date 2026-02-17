#!/bin/bash
set -e

echo "=== Fixing Face ID Deployment (dlib SIGILL Error - NUCLEAR OPTION) ==="

# 1. Stop services
echo "[1/4] Stopping face_id service..."
sudo systemctl stop face_id || true

# 2. Reinstall dlib from source (FORCE COMPILATION)
echo "[2/4] Reinstalling dlib from source (This WILL take 5-10 minutes)..."
source venv/bin/activate

# Uninstall twice to be sure
pip uninstall -y dlib || true
pip uninstall -y dlib || true

# Clear pip cache to avoid using the same broken wheel
echo "Clearing pip cache..."
pip cache purge

# Install dlib from source, forcing compilation
# --no-binary :all: ensures NO wheels are used
# --no-cache-dir ensures we download fresh source
# --force-reinstall ensures we overwrite anything existing
echo "Compiling dlib..."
pip install --no-cache-dir --force-reinstall --no-binary :all: dlib

# 3. Start the service
echo "[3/4] Restarting face_id service..."
sudo systemctl restart face_id

# 4. Check status
echo "[4/4] Checking status..."
sudo systemctl status face_id --no-pager

echo "=== Fix Complete! Check logs with: sudo journalctl -u face_id -f ==="
