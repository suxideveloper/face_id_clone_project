import face_recognition
import os
import pickle
import numpy as np
import cv2
from app.core.config import settings

class FaceRecognizer:
    def __init__(self):
        # Store multiple encodings per user for better accuracy
        self.user_encodings = {}  # {name: [encoding1, encoding2, ...]}
        self.known_face_encodings = []  # Flat list for backward compatibility
        self.known_face_names = []
        self.encodings_path = os.path.join(settings.DATA_DIR, "encodings.pkl")
        self.tolerance = 0.45  # Strictor tolerance to reduce false positives
        self.load_encodings()

    def load_encodings(self):
        if os.path.exists(self.encodings_path):
            try:
                with open(self.encodings_path, 'rb') as f:
                    data = pickle.load(f)
                    
                    # Support new format (multi-encoding) and old format
                    if 'user_encodings' in data:
                        self.user_encodings = data['user_encodings']
                        self._rebuild_flat_lists()
                    else:
                        # Legacy format - convert to new format
                        self.known_face_encodings = data.get('encodings', [])
                        self.known_face_names = data.get('names', [])
                        self._migrate_to_multi_encoding()
            except Exception as e:
                print(f"Error loading encodings: {e}")
                self.user_encodings = {}
                self.known_face_encodings = []
                self.known_face_names = []
        else:
            self.user_encodings = {}
            self.known_face_encodings = []
            self.known_face_names = []
    
    def _migrate_to_multi_encoding(self):
        """Convert old single-encoding format to multi-encoding format"""
        for i, name in enumerate(self.known_face_names):
            if name not in self.user_encodings:
                self.user_encodings[name] = []
            self.user_encodings[name].append(self.known_face_encodings[i])
        self.save_encodings()
        print("Migrated to multi-encoding format")
    
    def _rebuild_flat_lists(self):
        """Rebuild flat lists from user_encodings for compatibility"""
        self.known_face_encodings = []
        self.known_face_names = []
        for name, encodings in self.user_encodings.items():
            for enc in encodings:
                self.known_face_encodings.append(enc)
                self.known_face_names.append(name)

    def save_encodings(self):
        data = {
            "user_encodings": self.user_encodings,
            # Also save flat lists for backward compatibility
            "encodings": self.known_face_encodings,
            "names": self.known_face_names
        }
        with open(self.encodings_path, 'wb') as f:
            pickle.dump(data, f)

    def register_user(self, name, images):
        """
        Register a user with list of images (BGR numpy arrays).
        Stores ALL encodings (not averaged) for better recognition.
        """
        user_encodings = []
        user_dir = os.path.join(settings.IMAGES_DIR, name)
        os.makedirs(user_dir, exist_ok=True)
        
        saved_count = 0
        for i, img in enumerate(images):
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            boxes = face_recognition.face_locations(rgb_img)
            
            if boxes:
                # Compute encoding
                encoding = face_recognition.face_encodings(rgb_img, boxes)[0]
                user_encodings.append(encoding)
                
                # Save image
                file_path = os.path.join(user_dir, f"{name}_{i}.jpg")
                cv2.imwrite(file_path, img)
                saved_count += 1
        
        if user_encodings:
            # Store ALL encodings for this user (not averaged)
            if name not in self.user_encodings:
                self.user_encodings[name] = []
            self.user_encodings[name].extend(user_encodings)
            
            # Update flat lists
            self._rebuild_flat_lists()
            self.save_encodings()
            print(f"Registered {name} with {saved_count} encodings (total: {len(self.user_encodings[name])})")
            return True
        return False

    def verify(self, frame, face_location=None):
        """
        Verify face in frame using voting system.
        Checks against ALL encodings per user and uses best match.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if face_location:
            x1, y1, x2, y2 = face_location
            # css (top, right, bottom, left)
            css_location = (y1, x2, y2, x1)
            face_locations = [css_location]
        else:
            face_locations = face_recognition.face_locations(rgb_frame)

        if not face_locations:
            return "Unknown"

        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        if not face_encodings:
            return "Unknown"
            
        encoding = face_encodings[0]
        
        if not self.known_face_encodings:
            return "Unknown"

        # Vectorized comparison using the flat list (MUCH FASTER)
        # Calculate distance to ALL known encodings at once
        distances = face_recognition.face_distance(self.known_face_encodings, encoding)
        best_match_index = np.argmin(distances)
        
        if distances[best_match_index] < self.tolerance:
            return self.known_face_names[best_match_index]
            
        return "Unknown"
    
    def get_all_user_names(self):
        """Get list of all registered user names"""
        return list(self.user_encodings.keys())

recognizer = FaceRecognizer()
