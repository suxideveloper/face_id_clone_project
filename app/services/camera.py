import cv2
import threading
import time
import os
import numpy as np
from app.core.config import settings

class CameraService:
    def __init__(self):
        self.source = settings.CAMERA_ID
        self.cap = None
        self.lock = threading.Lock()
        self.last_frame = None
        
        # Do not connect immediately to avoid lock contention during uvicorn reload
        # self.connect_camera()

    def _device_exists(self, src):
        """Check if a camera device exists before trying to open it (prevents SEGV on headless servers)."""
        if isinstance(src, int):
            # Check if /dev/videoN exists
            device_path = f"/dev/video{src}"
            if not os.path.exists(device_path):
                print(f"Camera device {device_path} does not exist. Skipping.")
                return False
        # For RTSP URLs or other strings, we can't pre-check — just try to open
        return True

    def connect_camera(self):
        """Attempts to connect to the configured camera source, falling back to other indices if needed."""
        if self.cap is not None:
            self.cap.release()
            
        # Try the configured source first
        sources_to_try = [self.source]
        if isinstance(self.source, int):
            # If it's an index, try other common indices
            sources_to_try.extend([1, 2, -1])
            # Ensure unique
            sources_to_try = sorted(list(set(sources_to_try)))

        for src in sources_to_try:
            # Skip if device doesn't exist (prevents SEGV crash)
            if not self._device_exists(src):
                continue

            print(f"Attempting to open camera source: {src}")
            try:
                cap = cv2.VideoCapture(src)
                if cap.isOpened():
                    # Try to set HD resolution
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    
                    # Read a frame to be sure
                    ret, frame = cap.read()
                    if ret:
                        self.cap = cap
                        self.source = src
                        print(f"Successfully opened camera source: {src}")
                        return
                    else:
                        print(f"Opened source {src} but failed to read frame.")
                        cap.release()
                else:
                    print(f"Failed to open camera source: {src}")
                    cap.release()
            except Exception as e:
                print(f"Error opening camera source {src}: {e}")
        
        print("Could not open any camera source. Using dummy frame.")
        self.cap = None

    def get_dummy_frame(self):
        """Returns a black frame with 'No Signal' text."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, "No Camera Signal", (160, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, "Check Connection", (180, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        return frame

    def get_frame(self):
        with self.lock:
            if self.cap is None:
                self.connect_camera()
                
            if self.cap is None or not self.cap.isOpened():
                return self.get_dummy_frame()
                
            ret, frame = self.cap.read()
            if ret:
                self.last_frame = frame
                return frame
            else:
                # If reading fails, return last good frame or dummy
                print("Failed to read frame from camera.")
                return self.last_frame if self.last_frame is not None else self.get_dummy_frame()

    def change_source(self, new_source):
        """Changes the camera source and reconnects."""
        with self.lock:
            print(f"Switching camera source to: {new_source}")
            self.source = new_source
            # Force reconnection
            self.connect_camera()

    def release(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()

camera_service = CameraService()
