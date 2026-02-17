"""
Diagnostic script to pinpoint the exact cause of 
'Unsupported image type, must be 8bit gray or RGB image' error.
Run on SERVER: python test_recognition.py
"""
import sys
import os
import traceback
import numpy as np
import cv2

print("=" * 60)
print("DIAGNOSTIC: Face Recognition Pipeline")
print("=" * 60)

# 1. Check dlib version and path
print("\n[1/6] Checking dlib installation...")
import dlib
print(f"  dlib version: {dlib.__version__}")
print(f"  dlib path: {dlib.__file__}")
print(f"  CUDA support: {dlib.DLIB_USE_CUDA}")

# 2. Check face_recognition version
print("\n[2/6] Checking face_recognition...")
import face_recognition
print(f"  face_recognition version: {face_recognition.__version__}")

# Check face_recognition_models
import face_recognition_models
predictor_path = face_recognition_models.pose_predictor_model_location()
encoder_path = face_recognition_models.face_recognition_model_location()
print(f"  Pose predictor model: {predictor_path} (exists: {os.path.exists(predictor_path)})")
print(f"  Face encoder model: {encoder_path} (exists: {os.path.exists(encoder_path)})")

# 3. Load a debug frame
print("\n[3/6] Loading test image...")
debug_dir = "data/debug_frames"
if not os.path.exists(debug_dir) or not os.listdir(debug_dir):
    print("  No debug frames! Creating synthetic test image...")
    test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
else:
    frame_file = sorted(os.listdir(debug_dir))[-1]
    frame_path = os.path.join(debug_dir, frame_file)
    print(f"  Loading: {frame_path}")
    test_img = cv2.imread(frame_path)

print(f"  BGR: shape={test_img.shape}, dtype={test_img.dtype}")

# Convert to RGB
rgb = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
rgb = np.ascontiguousarray(rgb)
print(f"  RGB: shape={rgb.shape}, dtype={rgb.dtype}, contiguous={rgb.flags['C_CONTIGUOUS']}")

# 4. Test dlib shape_predictor DIRECTLY
print("\n[4/6] Testing dlib.shape_predictor directly...")
try:
    sp = dlib.shape_predictor(predictor_path)
    # Use a dummy rectangle in the image
    rect = dlib.rectangle(100, 100, 300, 300)
    shape = sp(rgb, rect)
    print(f"  ✅ shape_predictor works! Got {shape.num_parts} landmarks")
except Exception as e:
    print(f"  ❌ shape_predictor FAILED: {e}")
    traceback.print_exc()

# 5. Test face_recognition.face_locations (uses dlib HOG detector)
print("\n[5/6] Testing face_recognition.face_locations...")
try:
    locations = face_recognition.face_locations(rgb)
    print(f"  ✅ Found {len(locations)} face(s): {locations}")
except Exception as e:
    print(f"  ❌ face_locations FAILED: {e}")
    traceback.print_exc()

# 6. Test face_recognition.face_encodings
print("\n[6/6] Testing face_recognition.face_encodings...")

# 6a. Without locations (auto-detect)
print("  6a. Auto-detect mode:")
try:
    encodings = face_recognition.face_encodings(rgb)
    print(f"  ✅ Auto-detect: {len(encodings)} encoding(s)")
except Exception as e:
    print(f"  ❌ Auto-detect FAILED: {e}")
    traceback.print_exc()

# 6b. With YOLO-detected locations
print("\n  6b. With YOLO locations:")
try:
    from app.services.detector import detector
    detections = detector.detect(test_img)
    if detections:
        x1, y1, x2, y2 = detections[0][0]
        css = (y1, x2, y2, x1)  # top, right, bottom, left
        print(f"  YOLO detection: ({x1},{y1},{x2},{y2}) -> CSS: {css}")
        
        # Test with face_recognition
        encodings = face_recognition.face_encodings(rgb, [css])
        print(f"  ✅ YOLO+encoding: {len(encodings)} encoding(s)")
    else:
        print("  No YOLO detections found")
except Exception as e:
    print(f"  ❌ YOLO+encoding FAILED: {e}")
    traceback.print_exc()

# 6c. Test with a simple copy of the image
print("\n  6c. With np.copy():")
try:
    rgb_copy = rgb.copy()
    print(f"  Copy: shape={rgb_copy.shape}, dtype={rgb_copy.dtype}, contiguous={rgb_copy.flags['C_CONTIGUOUS']}")
    encodings = face_recognition.face_encodings(rgb_copy)
    print(f"  ✅ Copy mode: {len(encodings)} encoding(s)")
except Exception as e:
    print(f"  ❌ Copy mode FAILED: {e}")
    traceback.print_exc()

# 6d. Test loading image directly with face_recognition
print("\n  6d. Loading image with face_recognition.load_image_file:")
try:
    if os.path.exists(debug_dir) and os.listdir(debug_dir):
        frame_file = sorted(os.listdir(debug_dir))[-1]
        frame_path = os.path.join(debug_dir, frame_file)
        fr_img = face_recognition.load_image_file(frame_path)
        print(f"  FR loaded: shape={fr_img.shape}, dtype={fr_img.dtype}")
        encodings = face_recognition.face_encodings(fr_img)
        print(f"  ✅ FR load mode: {len(encodings)} encoding(s)")
except Exception as e:
    print(f"  ❌ FR load mode FAILED: {e}")
    traceback.print_exc()

print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)
