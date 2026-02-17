#!/bin/bash
set -e

echo "=== Fixing Face ID Systemd Service (Switching to Direct Python) ==="

SERVICE_FILE="/etc/systemd/system/face_id.service"

# Backup existing service file
if [ -f "$SERVICE_FILE" ]; then
    echo "Creating backup of $SERVICE_FILE..."
    sudo cp "$SERVICE_FILE" "$SERVICE_FILE.bak"
fi

# Create new service content locally
cat <<EOF > face_id.service.new
[Unit]
Description=Face ID App (Direct Python)
After=network.target

[Service]
User=suhrob
Group=www-data
WorkingDirectory=/home/suhrob/faceid/face_id_clone_project
Environment="PATH=/home/suhrob/faceid/face_id_clone_project/venv/bin"
Environment="PYTHONUNBUFFERED=1"
Environment="CAMERA_MODE=ip_camera"
Environment="OPENBLAS_CORETYPE=NEHALEM"
# Ensure no lingering processes (- prefix = ignore errors)
ExecStartPre=-/usr/bin/pkill -f gunicorn
# Run directly with python (Uvicorn handles workers internally if configured, or single process)
ExecStart=/home/suhrob/faceid/face_id_clone_project/venv/bin/python main.py
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# Install the new service file
echo "Updating $SERVICE_FILE..."
sudo mv face_id.service.new "$SERVICE_FILE"

# Reload systemd
echo "Reloading systemd..."
sudo systemctl daemon-reload

# Restart service
echo "Restarting face_id..."
sudo systemctl restart face_id

# Check status
echo "Checking status..."
sudo systemctl status face_id --no-pager

echo "=== Service Updated! Check logs with: sudo journalctl -u face_id -f ==="
