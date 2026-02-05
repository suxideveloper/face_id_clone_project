import numpy as np
import time

class Tracker:
    def __init__(self, max_lost=60, iou_threshold=0.4):
        """
        Args:
            max_lost: Frames before a track is deleted (increased from 15 to 60 for ~2 seconds)
            iou_threshold: Minimum IoU to consider a detection as same track
        """
        self.next_id = 1
        self.tracks = {}  # {id: {"bbox": (x1,y1,x2,y2), "name": None, "lost": 0, "last_verified": time, "verify_count": 0}}
        self.max_lost = max_lost
        self.iou_threshold = iou_threshold
        self.reverify_interval = 1.5  # Re-verify "Unknown" faces much faster (every 1.5s)
        self.unknown_retry_limit = 100  # Keep trying to recognize "Unknown" faces for a long time

    def _calculate_iou(self, box1, box2):
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2
        
        xi1 = max(x1, x3)
        yi1 = max(y1, y3)
        xi2 = min(x2, x4)
        yi2 = min(y2, y4)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x4 - x3) * (y4 - y3)
        
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0

    def update(self, detections):
        """
        detections: list of (x1, y1, x2, y2)
        returns: list of {"id": id, "bbox": bbox, "name": name, "is_new": bool, "needs_reverify": bool}
        """
        current_time = time.time()
        current_detections = []
        used_detections = set()
        
        # 1. Matching existing tracks
        for track_id, track_data in list(self.tracks.items()):
            best_iou = 0
            best_det_idx = -1
            
            for i, det in enumerate(detections):
                if i in used_detections:
                    continue
                iou = self._calculate_iou(track_data["bbox"], det)
                if iou > best_iou and iou > self.iou_threshold:
                    best_iou = iou
                    best_det_idx = i
            
            if best_det_idx != -1:
                # Track found!
                self.tracks[track_id]["bbox"] = detections[best_det_idx]
                self.tracks[track_id]["lost"] = 0
                used_detections.add(best_det_idx)
                
                # Check if we need to re-verify (for Unknown faces)
                needs_reverify = False
                name = self.tracks[track_id]["name"]
                last_verified = self.tracks[track_id].get("last_verified", 0)
                verify_count = self.tracks[track_id].get("verify_count", 0)
                
                if name == "Unknown" and verify_count < self.unknown_retry_limit:
                    if current_time - last_verified > self.reverify_interval:
                        needs_reverify = True
                
                current_detections.append({
                    "id": track_id, 
                    "bbox": detections[best_det_idx], 
                    "name": name,
                    "is_new": False,
                    "needs_reverify": needs_reverify
                })
            else:
                # Track lost
                self.tracks[track_id]["lost"] += 1
        
        # 2. Add new tracks
        for i, det in enumerate(detections):
            if i not in used_detections:
                new_id = self.next_id
                self.next_id += 1
                self.tracks[new_id] = {
                    "bbox": det, 
                    "name": None, 
                    "lost": 0,
                    "last_verified": 0,
                    "verify_count": 0
                }
                current_detections.append({
                    "id": new_id, 
                    "bbox": det, 
                    "name": None, 
                    "is_new": True,
                    "needs_reverify": False
                })
        
        # 3. Clean up old tracks
        ids_to_del = [tid for tid, data in self.tracks.items() if data["lost"] > self.max_lost]
        for tid in ids_to_del:
            del self.tracks[tid]
            
        return current_detections

    def set_name(self, track_id, name):
        """Set the recognized name for a track"""
        if track_id in self.tracks:
            self.tracks[track_id]["name"] = name
            self.tracks[track_id]["last_verified"] = time.time()
            self.tracks[track_id]["verify_count"] = self.tracks[track_id].get("verify_count", 0) + 1
    
    def clear_name(self, track_id):
        """Clear the name to trigger re-verification"""
        if track_id in self.tracks:
            self.tracks[track_id]["name"] = None
            self.tracks[track_id]["verify_count"] = 0

    def remove_user(self, name_to_remove):
        """Remove a user from all active tracks (e.g. after deletion)"""
        for tid, data in self.tracks.items():
            if data["name"] == name_to_remove:
                data["name"] = None
                data["verify_count"] = 0
                data["last_verified"] = 0

    def reset_unknown_verifications(self):
        """Reset verification limit for all Unknown tracks to allow re-check after new registration"""
        for tid, data in self.tracks.items():
            if data["name"] == "Unknown":
                data["verify_count"] = 0
                data["last_verified"] = 0  # Force immediate retry
