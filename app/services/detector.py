from ultralytics import YOLO
from app.core.config import settings
import cv2
import threading
import time
import numpy as np

class FaceDetector:
    def __init__(self):
        # This will download the model if not found
        self.model = YOLO(settings.YOLO_MODEL)
        
        # Async Processing
        self.latest_detections = []
        self.frame_to_process = None
        self.processing_lock = threading.Lock()
        self.running = True
        
        # Start background thread
        self.thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.thread.start()
        print("FaceDetector background thread started.")

    def _processing_loop(self):
        while self.running:
            if self.frame_to_process is None:
                time.sleep(0.01)
                continue
                
            # Get frame to process
            with self.processing_lock:
                frame = self.frame_to_process
                self.frame_to_process = None
            
            if frame is None:
                continue

            try:
                # Run inference
                # We can resize for speed if needed, but for now rely on YOLO internal resizing (imgsz=640 default)
                results = self.model(frame, verbose=False, imgsz=640)
                
                # Process results
                new_detections = []
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        # bounding box
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        new_detections.append(((int(x1), int(y1), int(x2), int(y2)), float(conf)))
                
                # Update latest detections
                self.latest_detections = new_detections
                
            except Exception as e:
                print(f"Error in detection loop: {e}")
                time.sleep(0.1)

    def detect(self, frame):
        # Update frame for background thread to process
        # We only want to process the LATEST frame, so overwriting is fine
        with self.processing_lock:
            self.frame_to_process = frame
            
        # Return currently available detections immediately (non-blocking)
        # If no detections yet (startup), return empty
        return self.latest_detections

detector = FaceDetector()
