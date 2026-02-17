#!/bin/bash
set -e

echo "============================================================"
echo "  Fixing dlib: Install latest version without AVX"
echo "============================================================"

# Show current environment
echo ""
echo "[INFO] Current environment:"
python -c "import numpy; print(f'  numpy: {numpy.__version__}')"
python -c "import sys; print(f'  Python: {sys.version}')"
python -c "import dlib; print(f'  dlib (broken): {dlib.__version__}')" 2>/dev/null || echo "  dlib: NOT INSTALLED"

# Step 1: Completely remove old dlib
echo ""
echo "[STEP 1] Removing broken dlib 19.24.0..."
pip uninstall dlib -y 2>/dev/null || true
# Remove all dlib remnants (eggs, wheels, dist-info)
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
echo "  Cleaning: $SITE_PACKAGES"
rm -rf "$SITE_PACKAGES"/dlib* 2>/dev/null || true
rm -rf "$SITE_PACKAGES"/dlib-*.egg 2>/dev/null || true

# Step 2: Download latest dlib source and compile without AVX
echo ""
echo "[STEP 2] Downloading latest dlib source..."
BUILD_DIR="/tmp/dlib_fix_$$"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Download latest source from PyPI
pip download dlib --no-binary :all: --no-deps -d .
echo "  Downloaded: $(ls dlib-*.tar.gz 2>/dev/null || ls dlib-*.zip 2>/dev/null)"

# Extract
tar xzf dlib-*.tar.gz 2>/dev/null || unzip dlib-*.zip 2>/dev/null
cd dlib-*/

echo ""
echo "[STEP 3] Compiling dlib WITHOUT AVX (this takes several minutes)..."
echo "  Flags: --no DLIB_USE_AVX_INSTRUCTIONS --yes DLIB_USE_CUDA"
python setup.py install \
    --no DLIB_USE_AVX_INSTRUCTIONS \
    --no DLIB_USE_SSE4_INSTRUCTIONS \
    --yes DLIB_USE_CUDA

# Return to project dir
cd ~/faceid/face_id_clone_project

# Cleanup
rm -rf "$BUILD_DIR"

# Step 4: Verify
echo ""
echo "============================================================"
echo "  VERIFICATION"
echo "============================================================"
python -c "
import dlib
print(f'  dlib version: {dlib.__version__}')
print(f'  dlib path: {dlib.__file__}')
print(f'  CUDA support: {dlib.DLIB_USE_CUDA}')

import numpy as np

# Test 1: Create simple image and test face detector
img = np.zeros((100, 100, 3), dtype=np.uint8)
det = dlib.get_frontal_face_detector()
det(img)
print('  ✅ Test 1: Face detector works!')

# Test 2: Test shape predictor
import face_recognition_models
sp = dlib.shape_predictor(face_recognition_models.pose_predictor_model_location())
rect = dlib.rectangle(10, 10, 90, 90)
sp(img, rect)
print('  ✅ Test 2: Shape predictor works!')

# Test 3: Test face_recognition
import face_recognition
encs = face_recognition.face_encodings(img)
print('  ✅ Test 3: face_recognition.face_encodings works!')

print('')
print('  🎉 ALL TESTS PASSED! Dlib is fully functional!')
"

echo ""
echo "============================================================"
echo "  SUCCESS! Now run: python main.py"
echo "============================================================"
