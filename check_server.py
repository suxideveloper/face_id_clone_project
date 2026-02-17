import sys
import os
import platform

print("=== Server Environment Check ===")
print(f"Python: {sys.version}")
print(f"Platform: {platform.platform()}")
print(f"Machine: {platform.machine()}")
print(f"Processor: {platform.processor()}")

print("\n[1] Checking Imports...")
try:
    import dlib
    print(f"✅ dlib imported successfully. Version: {dlib.__version__}")
    print(f"   dlib.DLIB_USE_CUDA: {dlib.DLIB_USE_CUDA}")
except ImportError as e:
    print(f"❌ Failed to import dlib: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error importing dlib (possible SIGILL/AVX issue): {e}")
    sys.exit(1)

try:
    import face_recognition
    print(f"✅ face_recognition imported successfully. Version: {face_recognition.__version__}")
except ImportError as e:
    print(f"❌ Failed to import face_recognition: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error importing face_recognition: {e}")
    sys.exit(1)

try:
    import cv2
    print(f"✅ opencv-python imported successfully. Version: {cv2.__version__}")
except ImportError as e:
    print(f"❌ Failed to import cv2: {e}")

try:
    from ultralytics import YOLO
    print("✅ ultralytics imported successfully.")
except ImportError:
    print("⚠️ ultralytics not found (using default YOLO?)")

print("\n[2] Checking Model Loading (CPU/AVX test)...")
try:
    # This triggers model loading which might cause SIGILL if AVX is missing and dlib is not compiled correctly
    face_encoder = dlib.get_face_recognition_model_v1(face_recognition.api.face_recognition_model_location())
    print("✅ dlib face recognition model loaded successfully.")
except Exception as e:
    print(f"❌ Failed to load dlib model: {e}")
    print("   CRITICAL: This usually means dlib was compiled with AVX instructions but CPU doesn't support them.")

print("\n[3] Functional Test (Face Detection)...")
try:
    import numpy as np
    # Create a simple black image with a white square (dummy face)
    # This won't actually detect a face but tests the pipeline without crashing
    img = np.zeros((300, 300, 3), dtype="uint8")
    
    # Try dlib detection
    detector = dlib.get_frontal_face_detector()
    dets = detector(img, 1)
    print(f"✅ Dlib detector ran successfully (Found {len(dets)} faces on blank image, expected 0).")
    
    # Try face_recognition full pipeline
    # We need a real face for this to return something, but we just want to ensure it doesn't Crash (SIGILL)
    encodings = face_recognition.face_encodings(img)
    print(f"✅ face_recognition.face_encodings ran successfully (Found {len(encodings)} faces).")
    
except Exception as e:
    print(f"❌ Functional test failed: {e}")
    print("   If this crashed with 'Illegal instruction', you NEED to recompile dlib.")

print("\n=== Check Complete ===")
