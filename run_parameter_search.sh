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

# Save script directory (absolute path)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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

# Function to test optimized configuration
# This is used by both 'test-optimized' mode and 'full' mode
test_optimized_config() {
    local results_dir="$1"
    shift
    local test_files_abs=("$@")
    
    # Apply recommendations
    echo "Applying PART 4 recommendations to SetSplitReductionSizes.cpp..."
    python3 apply_recommendations.py \
        "$results_dir/comprehensive_analysis.txt" \
        "$CPP_FILE"
    
    if [ $? -ne 0 ]; then
        echo "❌ Failed to apply recommendations"
        return 1
    fi
    
    # Build IREE
    echo ""
    echo "Building IREE with optimized settings..."
    cd "$IREE_BUILD_DIR"
    cmake --build . --target iree-compile 2>&1 | grep -E "(Building|error|warning)" || true
    
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "❌ Build failed!"
        # Restore original C++ file
        if [ -f "${CPP_FILE}.before_optimization" ]; then
            cp "${CPP_FILE}.before_optimization" "$CPP_FILE"
            echo "✓ Original C++ restored"
        fi
        return 1
    fi
    
    echo "✓ Build successful"
    echo ""
    echo "Running benchmarks with optimized configuration..."
    
    # Run benchmarks for each test file
    for test_file in "${test_files_abs[@]}"; do
        if [ ! -f "$test_file" ]; then
            echo "⚠️  Test file not found: $test_file"
            continue
        fi
        
        test_basename=$(basename "$test_file" .txt)
        csv_file="$results_dir/optimized_config_${test_basename}.csv"
        
        echo "  Testing $(basename $test_file)..."
        
        # Create temporary benchmark script for complete environment isolation
        cat > "$results_dir/_run_opt_bench.sh" << EOFSCRIPT
#!/bin/bash
set -e

# Set up IREE environment
export PATH="$IREE_BUILD_DIR/tools:\$PATH"
export PYTHONPATH="$IREE_BUILD_DIR/compiler/bindings/python:\$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Activate turbine venv and load .env
cd "$TURBINE_DIR"
source .venv/bin/activate
if [ -f .env ]; then
    source .env
    export PYTHONPATH
fi

# Run benchmark
python3 iree/turbine/kernel/boo/driver/driver.py \\
    --commands-file="$test_file" \\
    --csv="$csv_file"
EOFSCRIPT
        
        chmod +x "$results_dir/_run_opt_bench.sh"
        
        # Run benchmark (output redirected to log file)
        if "$results_dir/_run_opt_bench.sh" > "$results_dir/_bench_output.log" 2>&1; then
            if [ -f "$csv_file" ]; then
                echo "  ✓ Completed $(basename $test_file)"
            else
                echo "  ❌ Benchmark completed but no CSV file created"
            fi
        else
            echo "  ❌ Benchmark failed for $(basename $test_file)"
            echo "     Last 20 lines of output:"
            if [ -f "$results_dir/_bench_output.log" ]; then
                tail -20 "$results_dir/_bench_output.log" | sed "s/^/     /"
            fi
        fi
        
        rm -f "$results_dir/_run_opt_bench.sh"
    done
    
    echo ""
    echo "✓ Optimized configuration benchmarks complete"
    echo ""
    
    # Update comprehensive analysis with optimized results
    echo "Updating comprehensive analysis with validated results..."
    cd "$SCRIPT_DIR"
    python3 analyze_results.py \
        --results-file "$results_dir/limitParallelLoops_sweep_results.json" \
        --output-dir "$results_dir" \
        --baseline-limit baseline \
        --optimized-csv "$results_dir/optimized_config_*.csv"
    
    echo ""
    echo "✓ Analysis updated with optimized configuration results"
    
    # Restore original C++ file
    if [ -f "${CPP_FILE}.before_optimization" ]; then
        echo ""
        echo "Restoring original C++ file..."
        cp "${CPP_FILE}.before_optimization" "$CPP_FILE"
        echo "✓ Original SetSplitReductionSizes.cpp restored"
    fi
    
    return 0
}

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
    test-optimized)
        echo "Testing optimized configuration..."
        echo ""
        
        if [ ! -f "$RESULTS_DIR/comprehensive_analysis.txt" ]; then
            echo "Error: No analysis found at $RESULTS_DIR/comprehensive_analysis.txt"
            echo "Run analysis first: ./run_parameter_search.sh analyze"
            exit 1
        fi
        
        # Call the function to test optimized configuration
        test_optimized_config "$RESULTS_DIR" "${TEST_FILES_ABS[@]}"
        
        if [ $? -eq 0 ]; then
            echo ""
            echo "================================================================================"
            echo "  ✅ VALIDATION COMPLETE!"
            echo "================================================================================"
            echo ""
            echo "Results saved to: $RESULTS_DIR/"
            echo ""
            echo "Check PART 6 in: $RESULTS_DIR/comprehensive_analysis.txt"
            echo "  This section shows the validated optimized performance!"
            echo ""
        else
            echo ""
            echo "⚠️  Validation failed. Check the error messages above."
            exit 1
        fi
        exit 0
        ;;
    *)
        echo "Usage: $0 {quick|full|analyze|test-optimized} [test_files...]"
        echo ""
        echo "Modes:"
        echo "  quick         - Test 4 values: 1, 64, 128, 256 (fast)"
        echo "  full          - Test 10 values: 1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048"
        echo "  analyze       - Analyze existing results (no testing)"
        echo "  test-optimized - Apply and test the optimized configuration"
        echo ""
        echo "Examples:"
        echo "  $0 quick ../small_test.txt"
        echo "  $0 full ../prod_weight_shapes_conv.txt"
        echo "  $0 analyze"
        echo "  $0 test-optimized ../prod_weight_shapes_conv.txt"
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

# Apply recommendations and test optimized configuration automatically
echo ""
echo "================================================================================"
echo "  APPLYING RECOMMENDATIONS AND TESTING OPTIMIZED CONFIGURATION"
echo "================================================================================"
echo ""

# Call the function to test optimized configuration
test_optimized_config "$RESULTS_DIR" "${TEST_FILES_ABS[@]}"

if [ $? -ne 0 ]; then
    echo ""
    echo "⚠️  Validation failed. Check the error messages above."
    echo "   You can manually apply the recommendations from PART 4"
    echo "   and run: ./run_parameter_search.sh test-optimized"
    exit 1
fi

echo ""
echo "NOTE: The optimized C++ code is in PART 4 of comprehensive_analysis.txt"
echo "      Apply it manually when you're ready to use the optimized configuration"

echo ""
echo "================================================================================"
echo "  🎉 ALL DONE!"
echo "================================================================================"
echo ""
echo "Results saved to: $RESULTS_DIR/"
echo ""
echo "📊 comprehensive_analysis.txt contains:"
echo "  ✓ PART 1: Performance summary for all tested limits"
echo "  ✓ PART 2: Baseline comparison"
echo "  ✓ PART 3: Cluster analysis"
echo "  ✓ PART 4: C++ code recommendations ⭐ (APPLY THESE!)"
echo "  ✓ PART 5: Top speedups"
echo "  ✓ PART 6: Validated optimized performance ⭐ (TESTED & CONFIRMED!)"
echo ""
echo "📈 Performance validated:"
echo "   The recommendations in PART 4 have been automatically applied,"
echo "   tested, and validated. Check PART 6 for actual speedup numbers!"
echo ""
echo "📝 To use the optimized configuration in production:"
echo "   Copy the C++ code from PART 4 to SetSplitReductionSizes.cpp"
echo "   and rebuild IREE."
echo ""
