#!/bin/bash

SERVICE_NAME="faceid.service"
SERVICE_PATH="/etc/systemd/system/$SERVICE_NAME"
CURRENT_DIR=$(pwd)

echo "=== Installing Face ID Service ==="

if [ ! -f "$SERVICE_NAME" ]; then
    echo "❌ Error: $SERVICE_NAME not found in current directory!"
    exit 1
fi

echo "1. Copying service file to /etc/systemd/system/..."
sudo cp "$SERVICE_NAME" "$SERVICE_PATH"

echo "2. Reloading systemd daemon..."
sudo systemctl daemon-reload

echo "3. Enabling service to start on boot..."
sudo systemctl enable faceid

echo "4. Starting service..."
sudo systemctl restart faceid

echo "5. Checking status..."
sudo systemctl status faceid --no-pager

echo "=== Installation Complete ==="
