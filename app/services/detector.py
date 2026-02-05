from ultralytics import YOLO
from app.core.config import settings
import cv2

class FaceDetector:
    def __init__(self):
        # This will download the model if not found
        self.model = YOLO(settings.YOLO_MODEL)

    def detect(self, frame):
        # Run inference
        results = self.model(frame, verbose=False)
        # Process results
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # bounding box
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                detections.append(((int(x1), int(y1), int(x2), int(y2)), float(conf)))
        return detections

detector = FaceDetector()
