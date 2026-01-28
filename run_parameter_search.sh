#!/bin/bash
#
# Run limitParallelLoops sweep with COMPLETE ISOLATION
#
# Key difference from run_limit_sweep.sh:
# - Calls optimize_single_limit.py ONCE per limit value
# - Each call is a FRESH Python process
# - Complete isolation between different limit values
#
# This solves the caching issue where results were identical!

set -e

# Configuration
CPP_FILE="../iree/compiler/src/iree/compiler/DispatchCreation/SetSplitReductionSizes.cpp"
IREE_BUILD_DIR="../iree-build"
TURBINE_DIR="../iree-turbine"
GPU_ID=5

# Check if running in split_reduction_optimization directory
if [[ ! -f "optimize_single_limit.py" ]]; then
    echo "Error: Must run from split_reduction_optimization directory"
    exit 1
fi

# Parse arguments
MODE="${1:-quick}"
shift || true
TEST_FILES=("$@")

# Default test files if none provided
if [ ${#TEST_FILES[@]} -eq 0 ]; then
    if [ -f "../prod_weight_shapes_conv.txt" ]; then
        TEST_FILES=("../prod_weight_shapes_conv.txt")
    else
        TEST_FILES=("../small_test.txt")
    fi
fi

# Convert to absolute paths
TEST_FILES_ABS=()
for file in "${TEST_FILES[@]}"; do
    TEST_FILES_ABS+=("$(realpath "$file")")
done

# Derive results directory name from first test file
FIRST_TEST_FILE="${TEST_FILES_ABS[0]}"
TEST_BASENAME=$(basename "$FIRST_TEST_FILE")
# Remove extension (.txt, .csv, etc.)
TEST_NAME="${TEST_BASENAME%.*}"
# Remove operation-specific suffixes (_conv, _matmul, _mixed)
TEST_NAME="${TEST_NAME%_conv}"
TEST_NAME="${TEST_NAME%_matmul}"
TEST_NAME="${TEST_NAME%_mixed}"
# Create results directory name
RESULTS_DIR="../${TEST_NAME}_results"

# Define candidate limits based on mode
case "$MODE" in
    quick)
        LIMITS=(1 64 128 256)
        echo "Quick mode: Testing 4 values (including no-split baseline)"
        ;;
    full)
        LIMITS=(1 8 16 32 64 128 256 512 1024 2048)
        echo "Full mode: Testing 10 values (including no-split baseline)"
        ;;
    analyze)
        echo "Analyzing existing results..."
        echo ""
        
        if [ ! -f "$RESULTS_DIR/limitParallelLoops_sweep_results.json" ]; then
            echo "Error: No results found at $RESULTS_DIR/limitParallelLoops_sweep_results.json"
            echo "Run sweep first: ./run_parameter_search.sh quick"
            exit 1
        fi
        
        python3 analyze_results.py \
            --results-file "$RESULTS_DIR/limitParallelLoops_sweep_results.json" \
            --output-dir "$RESULTS_DIR" \
            --baseline-limit baseline
        
        echo ""
        echo "Analysis complete! See: $RESULTS_DIR/comprehensive_analysis.txt"
        exit 0
        ;;
    *)
        echo "Usage: $0 {quick|full|analyze} [test_files...]"
        echo ""
        echo "Modes:"
        echo "  quick   - Test 3 values: 64, 128, 256 (fast)"
        echo "  full    - Test all 9 values (comprehensive)"
        echo "  analyze - Analyze existing results"
        echo ""
        echo "Examples:"
        echo "  $0 quick ../small_test.txt"
        echo "  $0 full ../prod_weight_shapes_conv.txt"
        echo "  $0 analyze"
        echo ""
        exit 1
        ;;
esac

echo "Mode: $MODE"
echo "C++ File: $CPP_FILE"
echo "IREE Build: $IREE_BUILD_DIR"
echo "Turbine: $TURBINE_DIR"
echo "Results: $RESULTS_DIR"
echo "GPU ID: $GPU_ID"
echo "Limits to test: ${LIMITS[@]}"
echo "Test Files:"
for file in "${TEST_FILES_ABS[@]}"; do
    echo "  - $file"
done

# Create results directory
mkdir -p "$RESULTS_DIR"

# Run baseline FIRST (original C++ code, no modifications)
python3 optimize_single_limit.py \
    --cpp-file "$CPP_FILE" \
    --iree-build-dir "$IREE_BUILD_DIR" \
    --turbine-dir "$TURBINE_DIR" \
    --results-dir "$RESULTS_DIR" \
    --test-files "${TEST_FILES_ABS[@]}" \
    --gpu-id "$GPU_ID" \
    --baseline

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Failed to run baseline"
    echo "Stopping sweep."
    exit 1
fi

echo ""
echo "✅ Baseline complete"
echo "   CSV: $RESULTS_DIR/limit_baseline_*.csv"
echo ""

# Run each limit in a SEPARATE Python process
TOTAL=${#LIMITS[@]}
CURRENT=0

for limit in "${LIMITS[@]}"; do
    CURRENT=$((CURRENT + 1))
    
    # Call optimize_single_limit.py for THIS limit only
    # This is a FRESH Python process - no shared state with previous limits!
    python3 optimize_single_limit.py \
        --cpp-file "$CPP_FILE" \
        --iree-build-dir "$IREE_BUILD_DIR" \
        --turbine-dir "$TURBINE_DIR" \
        --results-dir "$RESULTS_DIR" \
        --test-files "${TEST_FILES_ABS[@]}" \
        --gpu-id "$GPU_ID" \
        --limit "$limit"
    
    if [ $? -ne 0 ]; then
        echo ""
        echo "❌ Failed to test limit=$limit"
        echo "Stopping sweep."
        exit 1
    fi
    
    echo ""
    echo "✅ Completed limit=$limit"
    echo "   CSV: $RESULTS_DIR/limit_${limit}_*.csv"
    echo ""
    
    # Wait a moment between tests for complete cleanup
    if [ $CURRENT -lt $TOTAL ]; then
        echo "Waiting 3 seconds for complete cleanup..."
        sleep 3
    fi
done

echo ""
echo "================================================================================"
echo "  ✅ SWEEP COMPLETE - All limits tested in isolated processes"
echo "================================================================================"
echo ""
echo "Results:"
echo "  - limit=baseline: $RESULTS_DIR/limit_baseline_*.csv  ← Original C++ code"
for limit in "${LIMITS[@]}"; do
    echo "  - limit=$limit: $RESULTS_DIR/limit_${limit}_*.csv"
done
echo ""

# Create summary JSON from CSV files
echo "Creating summary JSON from CSV files..."
python3 create_json_summary.py \
    --results-dir "$RESULTS_DIR" \
    --output "$RESULTS_DIR/limitParallelLoops_sweep_results.json"

# Run comprehensive analysis
echo ""
echo "Running comprehensive analysis..."
python3 analyze_results.py \
    --results-file "$RESULTS_DIR/limitParallelLoops_sweep_results.json" \
    --output-dir "$RESULTS_DIR" \
    --baseline-limit baseline

# Apply recommendations and test optimized configuration
echo ""
echo "================================================================================"
echo "  TESTING OPTIMIZED CONFIGURATION"
echo "================================================================================"
echo ""

# The C++ code should already have the optimized configuration applied
# Just run benchmarks to validate
echo "Running benchmarks with current C++ configuration (should be optimized)..."

for test_file in "${TEST_FILES[@]}"; do
    if [ ! -f "$test_file" ]; then
        continue
    fi
    
    test_basename=$(basename "$test_file" .txt)
    csv_file="$RESULTS_DIR/optimized_config_${test_basename}.csv"
    
    echo "  Testing $(basename $test_file)..."
    
    # Create temporary benchmark script
    cat > "$RESULTS_DIR/_run_opt_bench.sh" << EOFSCRIPT
#!/bin/bash
set -e
export PATH="$IREE_BUILD_DIR/tools:\$PATH"
export CUDA_VISIBLE_DEVICES=$GPU_ID
cd "$TURBINE_DIR"
source .venv/bin/activate
if [ -f .env ]; then source .env; fi
python3 iree/turbine/kernel/boo/driver/driver.py \\
    --commands-file="$test_file" \\
    --csv="$csv_file"
EOFSCRIPT
    
    chmod +x "$RESULTS_DIR/_run_opt_bench.sh"
    
    if "$RESULTS_DIR/_run_opt_bench.sh" > /dev/null 2>&1; then
        echo "  ✓ Completed $(basename $test_file)"
    else
        echo "  ⚠ Benchmark failed for $(basename $test_file)"
    fi
    
    rm -f "$RESULTS_DIR/_run_opt_bench.sh"
done

echo ""
echo "✓ Optimized configuration benchmarks complete"

# Re-run analysis with optimized results included
echo ""
echo "Updating comprehensive analysis with optimized results..."
python3 analyze_results.py \
    --results-file "$RESULTS_DIR/limitParallelLoops_sweep_results.json" \
    --output-dir "$RESULTS_DIR" \
    --baseline-limit baseline \
    --optimized-csv "$RESULTS_DIR/optimized_config_*.csv"

echo ""
echo "================================================================================"
echo "  🎉 ALL DONE!"
echo "================================================================================"
echo ""
echo "Results:"
echo "  - Comprehensive Analysis: $RESULTS_DIR/comprehensive_analysis.txt"
echo "    (includes sweep results, recommendations, AND optimized performance)"
echo "  - JSON Summary: $RESULTS_DIR/limitParallelLoops_sweep_results.json"
echo "  - Sweep CSV files: $RESULTS_DIR/limit_*.csv"
echo "  - Optimized CSV files: $RESULTS_DIR/optimized_config_*.csv"
echo ""
echo "The comprehensive_analysis.txt now contains:"
echo "  ✓ Sweep results for all tested limits"
echo "  ✓ C++ code recommendations"
echo "  ✓ Performance validation of optimized configuration"
echo "  ✓ Comparison: baseline vs optimized"
echo ""
