#!/bin/bash
set -e

echo "=== Compiling Dlib WITHOUT AVX (Manual Mode) ==="

# 1. Activate venv
source venv/bin/activate

# 2. Uninstall existing dlib (to be sure)
echo "[1/4] Uninstalling existing dlib..."
pip uninstall -y dlib || true
pip uninstall -y dlib || true

# 3. Clone specific stable version of dlib
echo "[2/4] Downloading dlib source (v19.24)..."
if [ -d "dlib_src" ]; then
    rm -rf dlib_src
fi
# Use git to clone exactly what we want
git clone -b v19.24 https://github.com/davisking/dlib.git dlib_src

# 4. Compile with AVX DISABLED explicitly
echo "[3/4] Compiling dlib with --no DLIB_USE_AVX_INSTRUCTIONS..."
cd dlib_src
# This is the critical part: telling setup.py explicitly to disable AVX
python setup.py install --no DLIB_USE_AVX_INSTRUCTIONS --no DLIB_USE_SSE4_INSTRUCTIONS

# 5. Cleanup and Verify
cd ..
rm -rf dlib_src

echo "=== Dlib Installed WITHOUT AVX! ==="
python -c "import dlib; print(f'Dlib installed successfully: {dlib.__version__}')"
