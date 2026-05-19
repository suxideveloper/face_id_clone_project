import face_recognition
import os
import numpy as np
import cv2
from sqlalchemy import text
from app.core.config import settings
from app.core.database import SessionLocal
from app.core.models import FaceEncoding, User
from app.services.detector import detector  # Import YOLO detector


class FaceRecognizer:
    def __init__(self):
        self.tolerance = 0.48  # Webcam + JPEG compression uchun yetarli (eski: 0.42 juda qat'iy edi)
        self.margin_threshold = 0.04  # Min gap between best and second-best different-user match

    @property
    def user_encodings(self):
        """Property for backward compatibility – returns {name: [encoding, ...]} from DB."""
        with SessionLocal() as db:
            rows = db.query(FaceEncoding).all()
            result = {}
            for r in rows:
                if r.username not in result:
                    result[r.username] = []
                result[r.username].append(np.array(r.embedding))
            return result

    def register_user(self, name, images):
        """
        Register a user with list of images (BGR numpy arrays).
        Stores ALL encodings (not averaged) for better recognition.
        """
        new_encodings = []
        user_dir = os.path.join(settings.IMAGES_DIR, name)
        os.makedirs(user_dir, exist_ok=True)

        saved_count = 0
        for i, img in enumerate(images):
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Try standard face_recognition detection (HOG/CNN)
            boxes = face_recognition.face_locations(rgb_img)

            # Fallback to YOLO if standard method fails
            if not boxes:
                detections = detector.detect(img)
                if detections:
                    # YOLO returns (x1, y1, x2, y2), face_recognition needs (top, right, bottom, left)
                    x1, y1, x2, y2 = detections[0][0]
                    boxes = [(y1, x2, y2, x1)]

            if boxes:
                try:
                    # Compute encoding
                    encoding = face_recognition.face_encodings(rgb_img, boxes)[0]
                    new_encodings.append(encoding)

                    # Save image
                    file_path = os.path.join(user_dir, f"{name}_{i}.jpg")
                    cv2.imwrite(file_path, img)
                    saved_count += 1
                except Exception as e:
                    print(f"Error encoding image {i} for {name}: {e}")
            else:
                print(f"Error: No face found in image {i} for {name} even with fallback.")

        if new_encodings:
            # Store ALL encodings into PostgreSQL with pgvector
            with SessionLocal() as db:
                for enc in new_encodings:
                    fe = FaceEncoding(
                        username=name,
                        embedding=enc.tolist(),
                    )
                    db.add(fe)
                db.commit()

            total = len(new_encodings)
            print(f"Registered {name} with {saved_count} encodings (new: {total})")
            return True
        return False

    def verify(self, frame, face_location=None):
        """
        Verify face in frame using pgvector nearest-neighbour search.
        Uses L2 distance operator (<->) for fast similarity lookup.
        Includes top-2 margin check: if the best match and second-best
        match (from a DIFFERENT user) are too close, returns Unknown.
        """
        if frame is None:
            return "Unknown"

        # Convert to proper BGR format first
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)

        # USE PIL to create a clean RGB numpy array
        # This bypasses dlib 19.24 pybind11 2.2.4 numpy buffer issue
        from PIL import Image
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        rgb_frame = np.array(pil_img)

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

        # ── pgvector nearest-neighbour query ──────────────────────────
        # Fetch top 5 results so we can check margin between different users
        vec_str = "[" + ",".join(str(float(v)) for v in encoding) + "]"

        with SessionLocal() as db:
            results = db.execute(
                text(
                    "SELECT username, embedding <-> :vec AS distance "
                    "FROM face_encodings "
                    "ORDER BY embedding <-> :vec "
                    "LIMIT 5"
                ),
                {"vec": vec_str},
            ).fetchall()

            if not results:
                return "Unknown"

            best = results[0]

            # DEBUG: har doim chop etish — qaysi distance da reject/accept bo'layotganini ko'rish
            top_info = [(r.username, round(r.distance, 4)) for r in results[:3]]
            print(f"[RECOGNIZE] Top matches: {top_info} | tolerance={self.tolerance}")

            # Check 1: Is the best match within tolerance?
            if best.distance >= self.tolerance:
                print(f"[RECOGNIZE] REJECTED: {best.username} distance={best.distance:.4f} >= tolerance={self.tolerance}")
                return "Unknown"

            # Check 2: Top-2 margin — find the closest match from a DIFFERENT user
            for r in results[1:]:
                if r.username != best.username:
                    margin = r.distance - best.distance
                    if margin < self.margin_threshold:
                        # Too ambiguous — the two users are too similar
                        print(f"[RECOGNIZE] AMBIGUOUS: {best.username}({best.distance:.4f}) vs {r.username}({r.distance:.4f}), margin={margin:.4f} < {self.margin_threshold}")
                        return "Unknown"
                    break  # Only need to check the first different user

            print(f"[RECOGNIZE] ACCEPTED: {best.username} distance={best.distance:.4f}")
            return best.username

    def delete_user_encodings(self, name: str):
        """Delete all face encodings for a user from the database."""
        with SessionLocal() as db:
            db.query(FaceEncoding).filter(FaceEncoding.username == name).delete()
            db.commit()

    def get_all_user_names(self):
        """Get list of all registered user names"""
        with SessionLocal() as db:
            rows = db.query(FaceEncoding.username).distinct().all()
            return [r[0] for r in rows]


recognizer = FaceRecognizer()
