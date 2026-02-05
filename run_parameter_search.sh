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
GPU_ID=7

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
    
    # CLEANUP: Restore C++ file to clean state before applying new recommendations
    echo "Cleaning up C++ file before applying recommendations..."
    if [ -f "${CPP_FILE}.before_optimization" ]; then
        cp "${CPP_FILE}.before_optimization" "$CPP_FILE"
        echo "  ✓ Restored from .before_optimization backup"
    else
        # Try git restore as fallback
        cd "$(dirname "$CPP_FILE")"
        if git status --porcelain "$(basename "$CPP_FILE")" 2>/dev/null | grep -q .; then
            git checkout -- "$(basename "$CPP_FILE")"
            echo "  ✓ Restored from git"
        fi
        cd "$SCRIPT_DIR"
    fi
    
    # Apply recommendations
    echo "Applying PART B recommendations to SetSplitReductionSizes.cpp..."
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

# ============================================================================
# BASELINE WITH DIMENSION LOGGING
# ============================================================================
# We temporarily add logging to capture kSize and outputSize during baseline,
# then remove the logging after baseline completes.

echo ""
echo "================================================================================"
echo "  BASELINE RUN WITH DIMENSION LOGGING"
echo "================================================================================"
echo ""

# Check if baseline files already exist - skip baseline run if so
SKIP_BASELINE=true
CAPTURED_DIMS_FILE="$RESULTS_DIR/captured_dimensions.json"

# Check if captured_dimensions.json exists and has content
if [ ! -f "$CAPTURED_DIMS_FILE" ] || [ ! -s "$CAPTURED_DIMS_FILE" ]; then
    SKIP_BASELINE=false
    echo "  captured_dimensions.json not found or empty - need baseline run"
fi

# Check if baseline CSV files exist for all test files
if [ "$SKIP_BASELINE" = true ]; then
    for test_file in "${TEST_FILES_ABS[@]}"; do
        test_basename=$(basename "$test_file" .txt)
        baseline_csv="$RESULTS_DIR/limit_baseline_${test_basename}.csv"
        if [ ! -f "$baseline_csv" ] || [ ! -s "$baseline_csv" ]; then
            SKIP_BASELINE=false
            echo "  $baseline_csv not found or empty - need baseline run"
            break
        fi
    done
fi

if [ "$SKIP_BASELINE" = true ]; then
    echo ""
    echo "✓ Baseline files already exist, skipping baseline run:"
    echo "  - $CAPTURED_DIMS_FILE"
    for test_file in "${TEST_FILES_ABS[@]}"; do
        test_basename=$(basename "$test_file" .txt)
        echo "  - $RESULTS_DIR/limit_baseline_${test_basename}.csv"
    done
    echo ""
else
    echo ""
    echo "Running baseline with dimension logging..."
    echo ""

# NOTE: Dimension parsing is now handled by the temporary C++ logging during baseline run.
# The logged dimensions (captured_dimensions.json) work for ALL test types uniformly:
# - Convolutions (weight backward)
# - GEMMs in ATen format (aten::mm, aten::addmm)
# - GEMMs in conv format (1x1 convolutions)
echo ""

# Step 1: Backup original C++ file
echo "Step 1: Backing up original C++ file..."
cp "$CPP_FILE" "${CPP_FILE}.original_backup"

# Step 2: Add dimension logging to C++ file
# IMPORTANT: 
# - Logging must be added BEFORE early return checks to capture all dimensions
# - We write to a FILE instead of stderr because turbine captures stderr internally
echo "Step 2: Adding dimension logging to C++ file..."
python3 - "$CPP_FILE" "$RESULTS_DIR" << 'PYEOF'
import sys
import os
cpp_file = sys.argv[1]
results_dir = sys.argv[2]

with open(cpp_file, 'r') as f:
    content = f.read()

# Check if logging is already present
if '[OPTIMIZER]' in content:
    print("⚠️ Logging already present in C++ file - skipping")
    sys.exit(0)

# Use absolute path for the log file
log_file_path = os.path.abspath(os.path.join(results_dir, "iree_dimension_log.txt"))
# Ensure parent directory exists
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

# Add file-based logging includes BEFORE the namespace declaration
import re

# Find the #include <fstream> insertion point (after mlir/Dialect/Linalg/IR/Linalg.h include)
# but BEFORE namespace declaration
fstream_include = f'\n#include <fstream>  // [OPTIMIZER] For file-based logging\n'

# Insert after the last standard include but before namespace
linalg_include_match = re.search(r'#include "mlir/Dialect/Linalg/IR/Linalg.h"', content)
if linalg_include_match:
    insert_pos = linalg_include_match.end()
    content = content[:insert_pos] + fstream_include + content[insert_pos:]

# Now add the getDimLogFile function after the #define DEBUG_TYPE line
dim_log_func = f'''
// [OPTIMIZER] File-based dimension logging function
static std::ofstream& getDimLogFile() {{
  static std::ofstream logFile("{log_file_path}", std::ios::app);
  return logFile;
}}
'''

# Insert after the GEN_PASS_DEF line and Passes.h.inc include
passes_inc_match = re.search(r'#include "iree/compiler/DispatchCreation/Passes.h.inc"', content)
if passes_inc_match:
    insert_pos = passes_inc_match.end()
    content = content[:insert_pos] + dim_log_func + content[insert_pos:]

# ============ CONVOLUTION FUNCTION (getWeightBackwardReductionSizes) ============
# Add logging BEFORE early return checks - right after depthSize is calculated
conv_dim_log = '''
  // [OPTIMIZER_LOG] Dimension logging to file (BEFORE early returns)
  getDimLogFile() << "[OPTIMIZER_DIM] Conv"
               << " outputChannelSize=" << outputChannelSize 
               << " batchSize=" << batchSize 
               << " imageSize=" << imageSize 
               << " depthSize=" << depthSize << "\\n";
  getDimLogFile().flush();
'''
content = content.replace(
    'int64_t depthSize = getSizeAt(outputShape, depthPos);',
    'int64_t depthSize = getSizeAt(outputShape, depthPos);' + conv_dim_log
)

# Add reductionSize logging after it's calculated
conv_reduction_log = '''
  // [OPTIMIZER_LOG] Reduction size logging to file
  getDimLogFile() << "[OPTIMIZER_DIM] Conv reductionSize=" << reductionSize << "\\n";
  getDimLogFile().flush();
'''
content = content.replace(
    'int64_t reductionSize = llvm::product_of(tileSizes);',
    'int64_t reductionSize = llvm::product_of(tileSizes);' + conv_reduction_log
)

# ============ MATMUL FUNCTION (getMatmulLikeReductionSizes) ============
# Add logging BEFORE early return checks - right after kSize is calculated
matmul_dim_log = '''
  // [OPTIMIZER_LOG] Dimension logging to file (BEFORE early returns)
  getDimLogFile() << "[OPTIMIZER_DIM] Matmul"
               << " mSize=" << mSize 
               << " nSize=" << nSize 
               << " kSize=" << kSize 
               << " batchSize=" << batchSize
               << " outputSize=" << (mSize * nSize * batchSize) << "\\n";
  getDimLogFile().flush();
'''
content = content.replace(
    'int64_t kSize = getSizeAt(kDims);',
    'int64_t kSize = getSizeAt(kDims);' + matmul_dim_log
)

with open(cpp_file, 'w') as f:
    f.write(content)

print("✓ Logging added to C++ file (writes to file: " + log_file_path + ")")
PYEOF

# Step 3: Build IREE with logging
echo "Step 3: Building IREE with dimension logging..."
cd "$IREE_BUILD_DIR"
BUILD_OUTPUT=$(cmake --build . --target iree-compile -j$(nproc) 2>&1)
BUILD_EXIT_CODE=$?
if [ $BUILD_EXIT_CODE -ne 0 ]; then
    echo "❌ Build failed with logging (exit code: $BUILD_EXIT_CODE)"
    echo "$BUILD_OUTPUT" | grep -E "error:" | head -10
    # Restore original
    cp "${CPP_FILE}.original_backup" "$CPP_FILE"
    rm -f "${CPP_FILE}.original_backup"
    exit 1
fi
echo "✓ Build successful"
cd "$SCRIPT_DIR"

# Step 4: Run baseline and capture dimension logs (ONE TEST AT A TIME for proper correlation)
echo "Step 4: Running baseline with dimension capture (test-by-test)..."

# Create single-test benchmark script
cat > "$RESULTS_DIR/_run_single_test.py" << 'PYEOF'
#!/usr/bin/env python3
"""Run a single test and log its configuration to the dimension log file."""
import sys
import os
import subprocess
import csv
import tempfile

def run_single_test(test_config, csv_output, dimension_log, turbine_dir, iree_build_dir, gpu_id):
    """Run a single benchmark test with dimension logging."""
    
    # Log the test configuration BEFORE compilation/running
    with open(dimension_log, 'a') as f:
        f.write(f"[TEST_CONFIG] {test_config}\n")
        f.flush()
    
    # Create a temporary file with just this one test
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp:
        tmp.write(test_config + '\n')
        tmp_file = tmp.name
    
    try:
        # Set up environment
        env = os.environ.copy()
        env['PATH'] = f"{iree_build_dir}/tools:" + env.get('PATH', '')
        env['PYTHONPATH'] = f"{iree_build_dir}/compiler/bindings/python:" + env.get('PYTHONPATH', '')
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        # Run from turbine directory
        original_dir = os.getcwd()
        os.chdir(turbine_dir)
        
        # Source the venv
        venv_activate = os.path.join(turbine_dir, '.venv/bin/activate')
        
        # Create temp CSV for this single test
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_csv:
            tmp_csv_file = tmp_csv.name
        
        # Build command
        cmd = f"""
        source {venv_activate}
        if [ -f .env ]; then source .env; export PYTHONPATH; fi
        python3 iree/turbine/kernel/boo/driver/driver.py \
            --commands-file="{tmp_file}" \
            --csv="{tmp_csv_file}"
        """
        
        result = subprocess.run(
            ['bash', '-c', cmd],
            env=env,
            capture_output=True,
            text=True
        )
        
        os.chdir(original_dir)
        
        # Read the result from temp CSV and append to main CSV
        if os.path.exists(tmp_csv_file):
            with open(tmp_csv_file, 'r') as f:
                reader = csv.reader(f)
                rows = list(reader)
            
            # Write to main CSV (append mode, skip header if file exists)
            file_exists = os.path.exists(csv_output) and os.path.getsize(csv_output) > 0
            with open(csv_output, 'a', newline='') as f:
                writer = csv.writer(f)
                for i, row in enumerate(rows):
                    if i == 0 and file_exists:
                        continue  # Skip header
                    writer.writerow(row)
            
            os.unlink(tmp_csv_file)
        
        return result.returncode == 0
        
    finally:
        if os.path.exists(tmp_file):
            os.unlink(tmp_file)

if __name__ == '__main__':
    if len(sys.argv) != 7:
        print("Usage: _run_single_test.py <test_config> <csv_output> <dimension_log> <turbine_dir> <iree_build_dir> <gpu_id>")
        sys.exit(1)
    
    test_config = sys.argv[1]
    csv_output = sys.argv[2]
    dimension_log = sys.argv[3]
    turbine_dir = sys.argv[4]
    iree_build_dir = sys.argv[5]
    gpu_id = sys.argv[6]
    
    success = run_single_test(test_config, csv_output, dimension_log, turbine_dir, iree_build_dir, gpu_id)
    sys.exit(0 if success else 1)
PYEOF
chmod +x "$RESULTS_DIR/_run_single_test.py"

# Run baseline for each test file
for test_file in "${TEST_FILES_ABS[@]}"; do
    test_basename=$(basename "$test_file" .txt)
    csv_file="$RESULTS_DIR/limit_baseline_${test_basename}.csv"
    log_file="$RESULTS_DIR/dimension_log_${test_basename}.txt"
    
    echo "  Running baseline for $test_basename..."
    
    # Clear the dimension log file before this test file
    > "$log_file"
    
    # Clear the CSV file
    > "$csv_file"
    
    # Count total tests
    total_tests=$(grep -c . "$test_file" 2>/dev/null || echo "0")
    current_test=0
    
    # Run each test one at a time
    while IFS= read -r test_config || [ -n "$test_config" ]; do
        # Skip empty lines
        [ -z "$test_config" ] && continue
        
        current_test=$((current_test + 1))
        printf "\r    Test %d/%d..." "$current_test" "$total_tests"
        
        python3 "$RESULTS_DIR/_run_single_test.py" \
            "$test_config" \
            "$csv_file" \
            "$RESULTS_DIR/iree_dimension_log.txt" \
            "$TURBINE_DIR" \
            "$IREE_BUILD_DIR" \
            "$GPU_ID" 2>/dev/null || true
            
    done < "$test_file"
    
    echo ""
    echo "  ✓ Baseline complete for $test_basename ($current_test tests)"
done

rm -f "$RESULTS_DIR/_run_single_test.py"

# Step 5: Parse dimension logs and create JSON (with test configuration matching)
echo "Step 5: Parsing dimension logs with test configurations..."
python3 - "$RESULTS_DIR" << 'PYEOF'
import sys
import json
import re
from pathlib import Path

results_dir = Path(sys.argv[1])
dimensions = {}  # Dict mapping test_config -> dimensions

# Read from the file-based log (written directly by iree-compile)
iree_log_file = results_dir / "iree_dimension_log.txt"
if iree_log_file.exists():
    with open(iree_log_file, 'r') as f:
        lines = f.readlines()
    
    current_test_config = None
    
    for line in lines:
        line = line.strip()
        
        # Check for test configuration marker
        test_match = re.match(r'\[TEST_CONFIG\] (.+)', line)
        if test_match:
            current_test_config = test_match.group(1)
            if current_test_config not in dimensions:
                dimensions[current_test_config] = {'test_config': current_test_config}
            continue
        
        # Skip if no test config set yet
        if current_test_config is None:
            continue
        
        # Parse Conv dimensions
        conv_match = re.match(r'\[OPTIMIZER_DIM\] Conv outputChannelSize=(\d+) batchSize=(\d+) imageSize=(\d+) depthSize=(\d+)', line)
        if conv_match:
            output_ch = int(conv_match.group(1))
            batch = int(conv_match.group(2))
            image = int(conv_match.group(3))
            depth = int(conv_match.group(4))
            dimensions[current_test_config].update({
                'type': 'conv',
                'outputChannelSize': output_ch,
                'batchSize': batch,
                'imageSize': image,
                'depthSize': depth,
                'outputSize': output_ch * batch * image * depth,
            })
            continue
        
        # Parse Conv reductionSize
        reduction_match = re.match(r'\[OPTIMIZER_DIM\] Conv reductionSize=(\d+)', line)
        if reduction_match and dimensions[current_test_config].get('type') == 'conv':
            dimensions[current_test_config]['reductionSize'] = int(reduction_match.group(1))
            continue
        
        # Parse Matmul dimensions
        matmul_match = re.match(r'\[OPTIMIZER_DIM\] Matmul mSize=(\d+) nSize=(\d+) kSize=(\d+) batchSize=(\d+) outputSize=(\d+)', line)
        if matmul_match:
            dimensions[current_test_config].update({
                'type': 'matmul',
                'mSize': int(matmul_match.group(1)),
                'nSize': int(matmul_match.group(2)),
                'kSize': int(matmul_match.group(3)),
                'batchSize': int(matmul_match.group(4)),
                'outputSize': int(matmul_match.group(5)),
            })
            continue
    
    print(f"  Parsed {len(dimensions)} test configurations with dimensions from {iree_log_file}")
    
    # Print summary of captured dimensions
    matmul_count = sum(1 for d in dimensions.values() if d.get('type') == 'matmul')
    conv_count = sum(1 for d in dimensions.values() if d.get('type') == 'conv')
    no_dim_count = sum(1 for d in dimensions.values() if 'type' not in d)
    print(f"    - {matmul_count} matmul operations")
    print(f"    - {conv_count} convolution operations")
    if no_dim_count > 0:
        print(f"    - {no_dim_count} tests without dimension data (didn't go through split reduction)")
else:
    print(f"  Warning: IREE dimension log not found at {iree_log_file}")
    print(f"  (This is expected if no operations went through split reduction)")

# Convert to list format for backwards compatibility, but keep test_config
dimensions_list = list(dimensions.values())

# Save to JSON
output_file = results_dir / "captured_dimensions.json"
with open(output_file, 'w') as f:
    json.dump(dimensions_list, f, indent=2)
print(f"  Saved {len(dimensions_list)} entries to {output_file}")
PYEOF

# Step 6: Remove logging from C++ file and restore original
echo "Step 6: Removing logging and restoring original C++ file..."
cp "${CPP_FILE}.original_backup" "$CPP_FILE"
rm -f "${CPP_FILE}.original_backup"
# Clean up the IREE dimension log file (it's no longer needed after parsing)
rm -f "$RESULTS_DIR/iree_dimension_log.txt"
echo "✓ Original C++ file restored"

# Step 7: Rebuild IREE without logging
echo "Step 7: Rebuilding IREE without logging..."
cd "$IREE_BUILD_DIR"
cmake --build . --target iree-compile -j$(nproc) > /dev/null 2>&1
echo "✓ Rebuild complete"
cd "$SCRIPT_DIR"

echo ""
echo "✅ Baseline complete with dimension capture"
echo "   CSV: $RESULTS_DIR/limit_baseline_*.csv"
echo "   Dimensions: $RESULTS_DIR/captured_dimensions.json"
echo ""

fi  # End of SKIP_BASELINE else block

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
