#!/bin/bash
set -e

echo "=== Fixing Face ID Deployment (dlib SIGILL Error) ==="

# 1. Stop services
echo "[1/3] Stopping face_id service..."
sudo systemctl stop face_id || true

# 2. Reinstall dlib from source (This fixes SIGILL error)
echo "[2/3] Reinstalling dlib from source (takes a few minutes)..."
source venv/bin/activate
pip uninstall -y dlib
# --no-binary dlib forces compilation on this machine, matching CPU features exactly
pip install dlib --no-binary dlib --verbose

# 3. Start the service
echo "[3/3] Restarting face_id service..."
sudo systemctl restart face_id

# 4. Check status
echo "[4/4] Checking status..."
sudo systemctl status face_id --no-pager

echo "=== Fix Complete! Check logs with: sudo journalctl -u face_id -f ==="
