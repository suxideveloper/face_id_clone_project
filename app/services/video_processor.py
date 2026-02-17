"""
Video Processor Service

Processes video frames received from browser camera via WebSocket.
Reuses existing detector, recognizer, and tracker services.
Each browser session gets its own tracker instance.
"""

import cv2
import numpy as np
import time
import asyncio
from app.services.detector import detector
from app.services.recognizer import recognizer
from app.services.tracker import Tracker


class VideoProcessor:
    """Processes frames from a single browser session."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.tracker = Tracker()
        self.attendance_debounce = {}
        self.visual_debounce = {}
        self.DEBOUNCE_SECONDS = 30
        self.VISUAL_DEBOUNCE_SECONDS = 3

    def decode_frame(self, frame_bytes: bytes) -> np.ndarray:
        """Decode JPEG bytes to OpenCV BGR frame."""
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return frame

    def process_verification_frame(self, frame: np.ndarray, attendance_queue) -> dict:
        """
        Process a single frame for face verification.
        Returns detection/recognition results as dict.

        Same logic as generate_frames() verification mode in routes.py
        """
        if frame is None:
            return {"faces": [], "error": "Invalid frame"}

        h, w = frame.shape[:2]
        # DEBUG: Save first frame of each session to verify quality
        import os
        debug_dir = "data/debug_frames"
        os.makedirs(debug_dir, exist_ok=True)
        debug_path = os.path.join(debug_dir, f"session_{self.session_id}.jpg")
        if not os.path.exists(debug_path):
            cv2.imwrite(debug_path, frame)
            print(f"DEBUG: Saved debug frame to {debug_path}")

        # Detect faces
        try:
            print("LOG: Starting Face Detection (YOLO)...")
            detections = detector.detect(frame)
            print(f"LOG: Finished Face Detection. Found {len(detections)} faces.")
            if len(detections) > 0:
                print(f"DEBUG: Frame {w}x{h}, Faces Detected: {len(detections)}")
        except Exception as e:
            print(f"ERROR: Face detection failed (YOLO): {e}")
            return {"faces": [], "error": str(e)}

        bbox_list = [det[0] for det in detections]
        tracked_faces = self.tracker.update(bbox_list)

        results = []
        for face in tracked_faces:
            tid = face["id"]
            x1, y1, x2, y2 = face["bbox"]
            name = face["name"]
            needs_reverify = face.get("needs_reverify", False)

            # Recognition: if name not cached OR needs re-verification
            if name is None or needs_reverify:
                try:
                    print(f"LOG: Starting Recognition for ID {tid} (Dlib)...")
                    name = recognizer.verify(frame, (x1, y1, x2, y2))
                    self.tracker.set_name(tid, name)
                    print(f"DEBUG: Recognized face ID {tid} as: {name}")
                except Exception as e:
                    print(f"ERROR: Recognition failed for ID {tid}: {e}")
                    name = "Unknown"
                    self.tracker.set_name(tid, name)

            # Attendance logging & Visual Feedback
            if name:
                current_time = time.time()
                last_db = self.attendance_debounce.get(name, 0)
                last_visual = self.visual_debounce.get(name, 0)

                if name != "Unknown":
                    should_log_db = (current_time - last_db >= self.DEBOUNCE_SECONDS)
                    should_show_visual = (current_time - last_visual >= self.VISUAL_DEBOUNCE_SECONDS)

                    if should_show_visual:
                        self.visual_debounce[name] = current_time
                        if should_log_db:
                            self.attendance_debounce[name] = current_time

                        try:
                            attendance_queue.put_nowait({
                                "worker_id": name,
                                "frame": frame.copy(),
                                "log_db": should_log_db
                            })
                        except asyncio.QueueFull:
                            pass
                else:
                    should_show_visual = (current_time - last_visual >= self.VISUAL_DEBOUNCE_SECONDS)
                    if should_show_visual:
                        self.visual_debounce[name] = current_time
                        try:
                            attendance_queue.put_nowait({
                                "worker_id": "Unknown",
                                "frame": frame.copy(),
                                "log_db": False
                            })
                        except asyncio.QueueFull:
                            pass

            # Build result for this face
            face_result = {
                "id": tid,
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "name": name or "Unknown",
            }
            if name and name != "Unknown":
                face_result["color"] = "green"
                face_result["label"] = f"ID:{tid} | {name}"
            else:
                face_result["color"] = "red"
                face_result["label"] = "Unknown"

            results.append(face_result)

        return {"faces": results, "frame_size": [w, h]}

    def process_registration_frame(self, frame: np.ndarray) -> dict:
        """
        Process a single frame for registration mode.
        Returns face detection info for the browser to display overlay.
        """
        if frame is None:
            return {"face_detected": False, "face_in_zone": False}

        h, w = frame.shape[:2]
        detections = detector.detect(frame)

        # Registration zone (same as generate_frames registration mode)
        bracket_w, bracket_h = int(w * 0.45), int(h * 0.7)
        bx1, by1 = (w - bracket_w) // 2, (h - bracket_h) // 2
        bx2, by2 = bx1 + bracket_w, by1 + bracket_h

        face_in_zone = False
        face_bbox = None
        face_valid = False

        for (x1, y1, x2, y2), conf in detections:
            fx, fy = (x1 + x2) // 2, (y1 + y2) // 2
            if bx1 < fx < bx2 and by1 < fy < by2:
                face_w = x2 - x1
                if face_w > bracket_w * 0.3:
                    face_valid = True
                    face_in_zone = True
                    face_bbox = [int(x1), int(y1), int(x2), int(y2)]
                    break
            if not face_in_zone:
                face_bbox = [int(x1), int(y1), int(x2), int(y2)]

        return {
            "face_detected": len(detections) > 0,
            "face_in_zone": face_in_zone,
            "face_valid": face_valid,
            "face_bbox": face_bbox,
            "zone": [bx1, by1, bx2, by2],
            "frame_size": [w, h],
        }


# Active sessions: {session_id: VideoProcessor}
active_processors = {}


def get_or_create_processor(session_id: str) -> VideoProcessor:
    """Get existing processor or create a new one for a session."""
    if session_id not in active_processors:
        active_processors[session_id] = VideoProcessor(session_id)
    return active_processors[session_id]


def remove_processor(session_id: str):
    """Remove processor when session disconnects."""
    if session_id in active_processors:
        del active_processors[session_id]
