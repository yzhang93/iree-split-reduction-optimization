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
# GPU configuration — distribute benchmark work across these devices.
# Use HIP_VISIBLE_DEVICES (not CUDA_VISIBLE_DEVICES) for ROCm/HIP builds of PyTorch.
GPU_IDS=(2 3 4 5 6 7)
GPU_ID=${GPU_IDS[0]}  # Default single GPU for baseline dimension capture

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
    
    local num_gpus=${#GPU_IDS[@]}
    
    # Run benchmarks for each test file using multi-GPU
    for test_file in "${test_files_abs[@]}"; do
        if [ ! -f "$test_file" ]; then
            echo "⚠️  Test file not found: $test_file"
            continue
        fi
        
        test_basename=$(basename "$test_file" .txt)
        csv_file="$results_dir/optimized_config_${test_basename}.csv"
        local total_shapes=$(wc -l < "$test_file")
        
        echo "  Testing $(basename $test_file) ($total_shapes shapes across $num_gpus GPUs: ${GPU_IDS[*]})..."
        
        # Clear BOO cache
        rm -rf /home/vivizhan/.cache/turbine_kernels/boo/
        
        # Split shapes into per-GPU chunks
        local chunk_prefix="$results_dir/_opt_chunk"
        local base_size=$(( total_shapes / num_gpus ))
        local chunk_remainder=$(( total_shapes % num_gpus ))
        local offset=1
        for (( gi = 0; gi < num_gpus; gi++ )); do
            local chunk_size=$base_size
            if (( gi < chunk_remainder )); then
                (( chunk_size++ ))
            fi
            if (( chunk_size > 0 )); then
                sed -n "${offset},$(( offset + chunk_size - 1 ))p" "$test_file" > "${chunk_prefix}_${gi}"
            else
                > "${chunk_prefix}_${gi}"
            fi
            (( offset += chunk_size ))
        done
        
        # Set up environment and launch parallel benchmarks
        export PATH="$IREE_BUILD_DIR/tools:$PATH"
        cd "$TURBINE_DIR"
        source .venv/bin/activate
        if [ -f .env ]; then source .env; export PYTHONPATH; fi
        
        local pids=()
        local part_csvs=()
        for (( gi = 0; gi < num_gpus; gi++ )); do
            local chunk_file="${chunk_prefix}_${gi}"
            if [ ! -s "$chunk_file" ]; then
                continue
            fi
            local part_csv="$results_dir/_opt_part_${gi}.csv"
            part_csvs+=("$part_csv")
            
            CUDA_VISIBLE_DEVICES="${GPU_IDS[$gi]}" \
            python "$TURBINE_DIR/iree/turbine/kernel/boo/driver/driver.py" \
                --commands-file "$chunk_file" \
                --csv "$part_csv" > /dev/null 2>&1 &
            pids+=($!)
        done
        
        # Monitor progress
        local any_running=true
        while $any_running; do
            any_running=false
            for pid in "${pids[@]}"; do
                if kill -0 "$pid" 2>/dev/null; then
                    any_running=true
                    break
                fi
            done
            local completed=0
            for part_csv in "${part_csvs[@]}"; do
                if [[ -f "$part_csv" ]]; then
                    local lines
                    lines=$(wc -l < "$part_csv")
                    (( lines > 1 )) && (( completed += lines - 1 ))
                fi
            done
            printf "\r    Progress: %d/%d ..." "$completed" "$total_shapes"
            sleep 2
        done
        
        # Wait for all
        for pid in "${pids[@]}"; do
            wait "$pid" 2>/dev/null || true
        done
        
        # Merge per-GPU CSVs
        local first=true
        for part_csv in "${part_csvs[@]}"; do
            if [[ -f "$part_csv" ]]; then
                if $first; then
                    cat "$part_csv" > "$csv_file"
                    first=false
                else
                    tail -n +2 "$part_csv" >> "$csv_file"
                fi
            fi
        done
        
        # Clean up
        for (( gi = 0; gi < num_gpus; gi++ )); do
            rm -f "${chunk_prefix}_${gi}" "$results_dir/_opt_part_${gi}.csv"
        done
        
        if [[ -f "$csv_file" ]]; then
            printf "\r  ✓ Completed $(basename $test_file)                    \n"
        else
            echo ""
            echo "  ❌ Benchmark failed for $(basename $test_file)"
        fi
        
        cd "$SCRIPT_DIR"
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
        LIMITS=(1 8 16 32 64 128 256 1024 2048)
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
echo "GPU IDs: ${GPU_IDS[*]}"
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

# Step 4: Run baseline with dimension capture across multiple GPUs.
# Each GPU gets a chunk of shapes and its own dimension log file.
# The per-GPU logs are merged afterwards for dimension parsing.
echo "Step 4: Running baseline with dimension capture (multi-GPU)..."

NUM_GPUS=${#GPU_IDS[@]}

# Clear BOO cache
rm -rf /home/vivizhan/.cache/turbine_kernels/boo/

for test_file in "${TEST_FILES_ABS[@]}"; do
    test_basename=$(basename "$test_file" .txt)
    csv_file="$RESULTS_DIR/limit_baseline_${test_basename}.csv"
    total_tests=$(wc -l < "$test_file")
    
    echo "  Running baseline for $test_basename ($total_tests shapes across $NUM_GPUS GPUs: ${GPU_IDS[*]})..."
    
    # Split shapes into per-GPU chunks
    local_base_size=$(( total_tests / NUM_GPUS ))
    local_remainder=$(( total_tests % NUM_GPUS ))
    local_offset=1
    
    # Create per-GPU worker scripts
    for (( gi = 0; gi < NUM_GPUS; gi++ )); do
        local_chunk_size=$local_base_size
        if [ "$gi" -lt "$local_remainder" ]; then
            local_chunk_size=$(( local_chunk_size + 1 ))
        fi
        if [ "$local_chunk_size" -le 0 ]; then
            continue
        fi
        
        # Extract chunk
        chunk_file="$RESULTS_DIR/_baseline_chunk_${gi}.txt"
        sed -n "${local_offset},$(( local_offset + local_chunk_size - 1 ))p" "$test_file" > "$chunk_file"
        local_offset=$(( local_offset + local_chunk_size ))
        
        # Create per-GPU worker script that runs shapes one-by-one with dim logging
        cat > "$RESULTS_DIR/_baseline_worker_${gi}.sh" << WORKEREOF
#!/bin/bash
export PATH="$IREE_BUILD_DIR/tools:\$PATH"
export CUDA_VISIBLE_DEVICES=${GPU_IDS[$gi]}
cd "$TURBINE_DIR"
source .venv/bin/activate
if [ -f .env ]; then source .env; export PYTHONPATH; fi

DIM_LOG="$RESULTS_DIR/_baseline_dimlog_${gi}.txt"
PART_CSV="$RESULTS_DIR/_baseline_part_${gi}.csv"

> "\$DIM_LOG"

while IFS= read -r test_config || [ -n "\$test_config" ]; do
    [ -z "\$test_config" ] && continue
    
    # Write test config marker to dimension log
    echo "[TEST_CONFIG] \$test_config" >> "\$DIM_LOG"
    
    # Create temp file for single test
    TMP_FILE=\$(mktemp /tmp/baseline_test_${gi}_XXXXXX.txt)
    echo "\$test_config" > "\$TMP_FILE"
    TMP_CSV=\$(mktemp /tmp/baseline_csv_${gi}_XXXXXX.csv)
    
    python3 iree/turbine/kernel/boo/driver/driver.py \\
        --commands-file="\$TMP_FILE" \\
        --csv="\$TMP_CSV" > /dev/null 2>&1 || true
    
    # Append to part CSV
    if [ -f "\$TMP_CSV" ] && [ -s "\$TMP_CSV" ]; then
        if [ ! -s "\$PART_CSV" ]; then
            cat "\$TMP_CSV" > "\$PART_CSV"
        else
            tail -n +2 "\$TMP_CSV" >> "\$PART_CSV"
        fi
    fi
    
    rm -f "\$TMP_FILE" "\$TMP_CSV"
done < "$RESULTS_DIR/_baseline_chunk_${gi}.txt"
WORKEREOF
        chmod +x "$RESULTS_DIR/_baseline_worker_${gi}.sh"
    done
    
    # Launch all workers in parallel
    baseline_pids=()
    for (( gi = 0; gi < NUM_GPUS; gi++ )); do
        if [ -f "$RESULTS_DIR/_baseline_worker_${gi}.sh" ]; then
            "$RESULTS_DIR/_baseline_worker_${gi}.sh" &
            baseline_pids+=($!)
        fi
    done
    
    # Monitor progress
    while true; do
        any_running=false
        for pid in "${baseline_pids[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                any_running=true
                break
            fi
        done
        $any_running || break
        
        completed=0
        for (( gi = 0; gi < NUM_GPUS; gi++ )); do
            if [ -f "$RESULTS_DIR/_baseline_part_${gi}.csv" ]; then
                lines=$(wc -l < "$RESULTS_DIR/_baseline_part_${gi}.csv")
                (( lines > 1 )) && (( completed += lines - 1 ))
            fi
        done
        printf "\r    Progress: %d/%d ..." "$completed" "$total_tests"
        sleep 2
    done
    
    # Wait for all
    for pid in "${baseline_pids[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
    
    # Merge per-GPU CSVs
    first_csv=true
    for (( gi = 0; gi < NUM_GPUS; gi++ )); do
        part="$RESULTS_DIR/_baseline_part_${gi}.csv"
        if [ -f "$part" ]; then
            if $first_csv; then
                cat "$part" > "$csv_file"
                first_csv=false
            else
                tail -n +2 "$part" >> "$csv_file"
            fi
        fi
    done
    
    # Merge per-GPU dimension logs into single file
    > "$RESULTS_DIR/iree_dimension_log.txt"
    for (( gi = 0; gi < NUM_GPUS; gi++ )); do
        dimlog="$RESULTS_DIR/_baseline_dimlog_${gi}.txt"
        if [ -f "$dimlog" ]; then
            cat "$dimlog" >> "$RESULTS_DIR/iree_dimension_log.txt"
        fi
    done
    
    # Clean up
    for (( gi = 0; gi < NUM_GPUS; gi++ )); do
        rm -f "$RESULTS_DIR/_baseline_chunk_${gi}.txt" \
              "$RESULTS_DIR/_baseline_worker_${gi}.sh" \
              "$RESULTS_DIR/_baseline_part_${gi}.csv" \
              "$RESULTS_DIR/_baseline_dimlog_${gi}.txt"
    done
    
    completed=$(( $(wc -l < "$csv_file") - 1 ))
    printf "\r  ✓ Baseline complete for $test_basename ($completed tests)          \n"
done

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
    GPU_IDS_STR=$(IFS=,; echo "${GPU_IDS[*]}")
    python3 optimize_single_limit.py \
        --cpp-file "$CPP_FILE" \
        --iree-build-dir "$IREE_BUILD_DIR" \
        --turbine-dir "$TURBINE_DIR" \
        --results-dir "$RESULTS_DIR" \
        --test-files "${TEST_FILES_ABS[@]}" \
        --gpu-id "$GPU_ID" \
        --gpu-ids "$GPU_IDS_STR" \
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
