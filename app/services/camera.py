import cv2
import threading
import time
import numpy as np
from app.core.config import settings

class CameraService:
    def __init__(self):
        self.source = settings.CAMERA_ID
        self.cap = None
        self.lock = threading.Lock()
        self.last_frame = None
        
        # Virtual Camera State
        self.use_virtual_camera = False
        self.last_virtual_frame_time = 0
        self.virtual_frame_timeout = 2.0 # Seconds before reverting to dummy
        
        # Retry Logic
        self.last_connection_attempt = 0
        self.connection_retry_interval = 5.0 # Wait 5s before retrying physical camera

    def connect_camera(self):
        """Attempts to connect to the configured camera source, falling back to other indices if needed."""
        # Rate limit connection attempts
        if time.time() - self.last_connection_attempt < self.connection_retry_interval:
            return

        self.last_connection_attempt = time.time()

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
            print(f"Attempting to open camera source: {src}")
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
                    self.use_virtual_camera = False
                    print(f"Successfully opened camera source: {src}")
                    return
                else:
                    print(f"Opened source {src} but failed to read frame.")
                    cap.release()
            else:
                print(f"Failed to open camera source: {src}")
        
        print("Could not open any camera source. Switching to Virtual/Dummy mode.")
        self.cap = None
        self.use_virtual_camera = True

    def get_dummy_frame(self):
        """Returns a black frame with 'No Signal' text."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Check if we are waiting for virtual frames
        if self.use_virtual_camera:
             cv2.putText(frame, "Waiting for Client Camera...", (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        else:
             cv2.putText(frame, "No Camera Signal", (160, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
             
        cv2.putText(frame, "Check Connection", (180, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        return frame

    def process_input_frame(self, frame_bytes):
        """Process a frame received from an external source (client)."""
        try:
            # Decode image
            nparr = np.frombuffer(frame_bytes, np.uint8)
            # Use imdecode to read JPEG from bytes
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is not None:
                with self.lock:
                    self.last_frame = frame
                    self.use_virtual_camera = True
                    self.last_virtual_frame_time = time.time()
                    # Debug log occasional frames
                    if int(self.last_virtual_frame_time) % 10 == 0:
                        print(f"Processed virtual frame: {frame.shape}")
            else:
                print("Failed to decode frame bytes.")
        except Exception as e:
            print(f"Error processing input frame: {e}")

    def get_frame(self):
        with self.lock:
            # If we have a physical camera, try to read from it
            if self.cap is not None and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    self.last_frame = frame
                    return frame
                else:
                    print("Physical camera read failed.")
                    self.cap.release()
                    self.cap = None
                    self.use_virtual_camera = True
            
            # If no physical camera, check if we should try to connect (but not too often)
            if not self.use_virtual_camera and (time.time() - self.last_connection_attempt > self.connection_retry_interval):
                 self.connect_camera()

            # If we are in virtual mode or connection failed
            if self.use_virtual_camera:
                # Check if we have a recent virtual frame
                if self.last_frame is not None and (time.time() - self.last_virtual_frame_time < self.virtual_frame_timeout):
                    return self.last_frame
            
            # Fallback to dummy
            return self.get_dummy_frame()

    def change_source(self, new_source):
        """Changes the camera source and reconnects."""
        with self.lock:
            print(f"Switching camera source to: {new_source}")
            self.source = new_source
            # Reset retry timer to force immediate attempt
            self.last_connection_attempt = 0
            self.connect_camera()

    def release(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()

camera_service = CameraService()
