#!/bin/bash
#
# Quick setup verification before running optimization
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Checking optimization setup..."
echo ""

# Default paths
CPP_FILE="${CPP_FILE:-../iree/compiler/src/iree/compiler/DispatchCreation/SetSplitReductionSizes.cpp}"
IREE_BUILD_DIR="${IREE_BUILD_DIR:-../iree-build}"
TURBINE_DIR="${TURBINE_DIR:-../iree-turbine}"

# Check C++ file
if [[ -f "$CPP_FILE" ]]; then
    echo "✓ C++ file found: $CPP_FILE"
else
    echo "✗ C++ file NOT found: $CPP_FILE"
    echo "  Set CPP_FILE environment variable or check path"
    exit 1
fi

# Check IREE build directory
if [[ -d "$IREE_BUILD_DIR" ]]; then
    echo "✓ IREE build directory found: $IREE_BUILD_DIR"
    
    # Check if ninja/make exists
    if [[ -f "$IREE_BUILD_DIR/build.ninja" ]] || [[ -f "$IREE_BUILD_DIR/Makefile" ]]; then
        echo "  ✓ Build system configured"
    else
        echo "  ✗ Build system not configured"
        echo "    Run: cd $IREE_BUILD_DIR && cmake ..."
        exit 1
    fi
else
    echo "✗ IREE build directory NOT found: $IREE_BUILD_DIR"
    echo "  Set IREE_BUILD_DIR environment variable or check path"
    exit 1
fi

# Check iree-turbine directory
if [[ -d "$TURBINE_DIR" ]]; then
    echo "✓ iree-turbine directory found: $TURBINE_DIR"
    
    # Check for virtual environment
    if [[ -f "$TURBINE_DIR/.venv/bin/python" ]]; then
        echo "  ✓ Virtual environment found"
        
        # Check if torch is installed
        if "$TURBINE_DIR/.venv/bin/python" -c "import torch" 2>/dev/null; then
            echo "  ✓ PyTorch installed in venv"
        else
            echo "  ✗ PyTorch NOT installed in venv"
            echo "    Run: cd $TURBINE_DIR && source .venv/bin/activate && pip install torch"
            exit 1
        fi
    else
        echo "  ✗ Virtual environment NOT found: $TURBINE_DIR/.venv"
        echo "    Run: cd $TURBINE_DIR && python -m venv .venv && source .venv/bin/activate && pip install -e ."
        exit 1
    fi
    
    # Check for driver script
    if [[ -f "$TURBINE_DIR/iree/turbine/kernel/boo/driver/driver.py" ]]; then
        echo "  ✓ Driver script found"
    else
        echo "  ✗ Driver script NOT found"
        echo "    Check iree-turbine installation"
        exit 1
    fi
else
    echo "✗ iree-turbine directory NOT found: $TURBINE_DIR"
    echo "  Set TURBINE_DIR environment variable or check path"
    exit 1
fi

# Check test files
TEST_FILES=(
    "../prod_weight_shapes.txt"
    "../proxy_weight_shapes.txt"
)

echo ""
echo "Checking test files..."
FOUND_ANY=false
for file in "${TEST_FILES[@]}"; do
    if [[ -f "$file" ]]; then
        echo "✓ Test file found: $file"
        FOUND_ANY=true
    else
        echo "⚠ Test file not found: $file (optional)"
    fi
done

if [[ "$FOUND_ANY" == "false" ]]; then
    echo "✗ No test files found!"
    echo "  Expected files: ${TEST_FILES[@]}"
    exit 1
fi

# Check GPU
GPU_ID="${GPU_ID:-5}"
if command -v nvidia-smi &> /dev/null; then
    if CUDA_VISIBLE_DEVICES=$GPU_ID nvidia-smi &> /dev/null; then
        echo ""
        echo "✓ GPU $GPU_ID accessible"
    else
        echo ""
        echo "⚠ GPU $GPU_ID may not be accessible"
        echo "  Set GPU_ID environment variable or check GPU availability"
    fi
else
    echo ""
    echo "⚠ nvidia-smi not found (skipping GPU check)"
fi

echo ""
echo "==========================================="
echo "✓ Setup verification passed!"
echo "==========================================="
echo ""
echo "Ready to run optimization:"
echo "  ./run_optimization.sh baseline"
echo "  ./run_optimization.sh quick"
echo "  ./run_optimization.sh full"
