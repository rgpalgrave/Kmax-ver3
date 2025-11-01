#!/bin/bash

# FFT k_max Accumulator - Streamlit Startup Script
# Usage: bash run.sh

set -e

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                 FFT k_max Accumulator - Streamlit App                      ║"
echo "║                      Starting Deployment...                                 ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"

echo ""
echo "📋 Checking dependencies..."

# Check Python version
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "✓ Python $PYTHON_VERSION"

# Check if requirements.txt exists
if [ ! -f "requirements.txt" ]; then
    echo "✗ requirements.txt not found!"
    exit 1
fi

echo "✓ requirements.txt found"

# Check if streamlit_app.py exists
if [ ! -f "streamlit_app.py" ]; then
    echo "✗ streamlit_app.py not found!"
    exit 1
fi

echo "✓ streamlit_app.py found"

# Check if kmax_fft_accumulator.py exists
if [ ! -f "kmax_fft_accumulator.py" ]; then
    echo "✗ kmax_fft_accumulator.py not found!"
    exit 1
fi

echo "✓ kmax_fft_accumulator.py found"

echo ""
echo "📦 Installing/verifying dependencies..."

# Install requirements
pip install -q -r requirements.txt

echo "✓ All dependencies installed"

echo ""
echo "✅ Verification checks:"

python -c "import streamlit; print('  ✓ Streamlit', streamlit.__version__)" 2>/dev/null || { echo "  ✗ Streamlit failed"; exit 1; }
python -c "import numpy; print('  ✓ NumPy', numpy.__version__)" 2>/dev/null || { echo "  ✗ NumPy failed"; exit 1; }
python -c "import scipy; print('  ✓ SciPy', scipy.__version__)" 2>/dev/null || { echo "  ✗ SciPy failed"; exit 1; }
python -c "import pandas; print('  ✓ Pandas', pandas.__version__)" 2>/dev/null || { echo "  ✗ Pandas failed"; exit 1; }
python -c "import plotly; print('  ✓ Plotly', plotly.__version__)" 2>/dev/null || { echo "  ✗ Plotly failed"; exit 1; }
python -c "import kmax_fft_accumulator; print('  ✓ kmax_fft_accumulator module')" 2>/dev/null || { echo "  ✗ kmax_fft_accumulator module failed"; exit 1; }

echo ""
echo "🚀 Starting Streamlit app..."
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  The app is running at: http://localhost:8501"
echo ""
echo "  Tips:"
echo "    • Press Ctrl+C to stop"
echo "    • Press 'R' to rerun the script"
echo "    • Check the terminal for error messages"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

streamlit run streamlit_app.py \
    --server.port 8501 \
    --server.address localhost \
    --logger.level info
