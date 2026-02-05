# Face Auth System

Real-time face recognition system using FastAPI, YOLOv8, and Face Recognition.

## Features
- Real-time video feed via FastAPI StreamingResponse.
- Face Detection using YOLOv8.
- Face Verification using `dlib` based `face_recognition`.
- User Registration (Captures 5 images per user).

## Setup

1. **Install Dependencies**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
   *Note: This includes PyTorch and can take a while.*

2. **Run the Application**:
   ```bash
   source venv/bin/activate
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```
   *Or simply run `python main.py`.*

## Usage
1. Open [http://localhost:8000](http://localhost:8000).
2. Allow camera access if prompted (though it runs on server, for local dev it uses server's webcam).
3. Click "Register New User".
4. Enter name and click "Capture". Wait for 5 images to be captured.
5. Go back to Home. You should see your face with your name in a green box.

## Configuration
Edit `app/core/config.py` to change:
- `CAMERA_ID`: Set to `0` for webcam, or an RTSP URL (e.g., `rtsp://user:pass@ip:port/stream`) for IP Cameras.
- `YOLO_MODEL`: Path to YOLO model (default `yolov8n-face.pt`).
