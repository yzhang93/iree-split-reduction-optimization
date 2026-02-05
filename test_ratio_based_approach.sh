#!/bin/bash
#
# Test the ratio-based split reduction approach
#
# This script:
# 1. Generates ratio-based C++ code (or uses provided code file)
# 2. Applies it to SetSplitReductionSizes.cpp
# 3. Builds IREE
# 4. Runs benchmarks
# 5. Compares against baseline

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Configuration
CPP_FILE="../iree/compiler/src/iree/compiler/DispatchCreation/SetSplitReductionSizes.cpp"
IREE_BUILD_DIR="../iree-build"
TURBINE_DIR="../iree-turbine"
GPU_ID=5

# Parse arguments
USE_PROVIDED_CODE=false
PROVIDED_CODE_FILE=""

show_usage() {
    echo "Usage:"
    echo "  Mode 1: Generate code from sweep results"
    echo "    $0 <results_dir> <test_file>"
    echo ""
    echo "  Mode 2: Use handwritten/provided code file"
    echo "    $0 --code <code_file> <results_dir> <test_file>"
    echo ""
    echo "Examples:"
    echo "  $0 ../all_weight_shapes_results ../all_weight_shapes_conv.txt"
    echo "  $0 --code ../all_weight_shapes_matmul_results/ratio_based_code_merged.cpp ../gemm_shapes_results ../gemm_shapes.txt"
}

# Check for --code flag
if [ "$1" = "--code" ]; then
    if [ $# -lt 4 ]; then
        echo "Error: --code requires <code_file> <results_dir> <test_file>"
        echo ""
        show_usage
        exit 1
    fi
    USE_PROVIDED_CODE=true
    PROVIDED_CODE_FILE="$2"
    RESULTS_DIR="$3"
    TEST_FILE="$4"
    
    if [ ! -f "$PROVIDED_CODE_FILE" ]; then
        echo "Error: Code file not found: $PROVIDED_CODE_FILE"
        exit 1
    fi
else
    if [ $# -lt 2 ]; then
        show_usage
        exit 1
    fi
    RESULTS_DIR="$1"
    TEST_FILE="$2"
fi

# Detect test type based on file content or name
TEST_TYPE="conv"
if grep -q "aten::mm\|aten::addmm\|aten::bmm" "$TEST_FILE" 2>/dev/null; then
    TEST_TYPE="gemm"
elif [[ "$TEST_FILE" == *"gemm"* ]] || [[ "$TEST_FILE" == *"matmul"* ]]; then
    TEST_TYPE="gemm"
fi

echo "================================================================================"
echo "  TESTING RATIO-BASED SPLIT REDUCTION APPROACH"
echo "================================================================================"
echo ""
echo "Results directory: $RESULTS_DIR"
echo "Test file: $TEST_FILE"
echo "Detected test type: $TEST_TYPE"
if $USE_PROVIDED_CODE; then
    echo "Using provided code: $PROVIDED_CODE_FILE"
fi
echo ""

# Step 1: Generate or use provided ratio-based C++ code
if $USE_PROVIDED_CODE; then
    echo "Step 1: Using provided ratio-based C++ code..."
    # Copy the provided code to the results directory for consistency
    RATIO_CODE_FILE="$RESULTS_DIR/ratio_based_code_provided.cpp"
    cp "$PROVIDED_CODE_FILE" "$RATIO_CODE_FILE"
    echo "✓ Using: $PROVIDED_CODE_FILE"
    echo "  (Copied to: $RATIO_CODE_FILE)"
else
    echo "Step 1: Generating ratio-based C++ code..."
    RATIO_CODE_FILE="$RESULTS_DIR/ratio_based_code.cpp"
    python3 generate_ratio_based_cpp.py \
        --results-file "$RESULTS_DIR/limitParallelLoops_sweep_results.json" \
        --output "$RATIO_CODE_FILE" \
        --test-type "$TEST_TYPE"

    if [ $? -ne 0 ]; then
        echo "❌ Failed to generate ratio-based C++ code"
        exit 1
    fi

    echo "✓ Generated: $RATIO_CODE_FILE"
fi
echo ""

# Step 2: Backup original C++ file
echo "Step 2: Creating backup of original C++ file..."
if [ ! -f "${CPP_FILE}.backup_before_ratio" ]; then
    cp "$CPP_FILE" "${CPP_FILE}.backup_before_ratio"
    echo "✓ Backup created: ${CPP_FILE}.backup_before_ratio"
else
    echo "✓ Backup already exists"
fi
echo ""

# Step 3: Apply ratio-based code to C++ file
echo "Step 3: Applying ratio-based code to SetSplitReductionSizes.cpp..."
echo "   This will replace the getWeightBackwardReductionSizes function logic"
echo ""

# Create a Python script to do the replacement
cat > "$RESULTS_DIR/_apply_ratio_code.py" << 'EOFPYTHON'
import sys
import re

cpp_file = sys.argv[1]
ratio_code_file = sys.argv[2]
test_type = sys.argv[3] if len(sys.argv) > 3 else 'conv'

# Read the ratio-based code
with open(ratio_code_file, 'r') as f:
    ratio_code = f.read()

# Read the original C++ file
with open(cpp_file, 'r') as f:
    lines = f.readlines()

# Process line by line to modify functions based on test type:
# - conv: Apply ratio-based code to getWeightBackwardReductionSizes, comment out early returns in both
# - gemm: Apply ratio-based code to getMatmulLikeReductionSizes, comment out early returns in both

output = []
in_wb_function = False  # Weight backward (conv)
in_mm_function = False  # Matmul-like
function_brace_depth = 0
skip_old_logic = False
wb_inserted = False
mm_inserted = False
found_getsizeat_wb = False
found_getsizeat_mm = False
in_early_return_block = False

i = 0
while i < len(lines):
    line = lines[i]
    
    # Detect function starts
    if 'getWeightBackwardReductionSizes' in line and '(' in line:
        in_wb_function = True
        in_mm_function = False
        function_brace_depth = 0
        skip_old_logic = False
        output.append(line)
        i += 1
        continue
    
    if 'getMatmulLikeReductionSizes' in line and '(' in line:
        in_mm_function = True
        in_wb_function = False
        function_brace_depth = 0
        skip_old_logic = False
        output.append(line)
        i += 1
        continue
    
    # Track braces within functions
    if in_wb_function or in_mm_function:
        function_brace_depth += line.count('{') - line.count('}')
        
        # Check if we've exited the function
        if function_brace_depth <= 0 and '{' not in line:
            if in_wb_function:
                in_wb_function = False
            if in_mm_function:
                in_mm_function = False
    
    # ==================== WEIGHT BACKWARD FUNCTION (conv) ====================
    if in_wb_function:
        # Track when we see the getSizeAt lambda
        if 'auto getSizeAt = [' in line:
            found_getsizeat_wb = True
        
        # Comment out constant declarations (they become unused when we replace the logic)
        if 'const int64_t largeParallelSize' in line or \
           'const int64_t largeReductionSize' in line or \
           'const int64_t ratioThreshold' in line:
            output.append('  // ' + line.strip() + '  // OPTIMIZER: Disabled (unused)\n')
            i += 1
            continue
        
        # Comment out reductionSize and ratio calculations (become unused when we replace the logic)
        if 'int64_t reductionSize = llvm::product_of(tileSizes)' in line:
            output.append('  // ' + line.strip() + '  // OPTIMIZER: Disabled (unused)\n')
            i += 1
            continue
        if 'int64_t ratio = reductionSize /' in line and 'std::sqrt' in line:
            output.append('  // ' + line.strip() + '  // OPTIMIZER: Disabled (unused)\n')
            i += 1
            continue
        
        # Comment out early return checks (always, for both test types)
        if 'if (outputChannelSize >= largeParallelSize' in line:
            output.append('#if 0  // OPTIMIZER: Disabled early return\n')
            in_early_return_block = True
        elif 'if (ratio <= ratioThreshold' in line and 'reductionSize' in ''.join(lines[max(0,i-5):i+1]) and not in_early_return_block:
            output.append('#if 0  // OPTIMIZER: Disabled early return\n')
            in_early_return_block = True
        
        if in_early_return_block:
            output.append(line)
            if line.strip() == '}':
                output.append('#endif  // OPTIMIZER\n')
                in_early_return_block = False
            i += 1
            continue
        
        # For conv tests: Apply ratio-based code after depthSize calculation
        if test_type == 'conv' and not wb_inserted and found_getsizeat_wb and 'depthSize = getSizeAt(outputShape, depthPos)' in line:
            output.append(line)
            # Skip all old logic and insert new ratio-based code
            skip_old_logic = True
            wb_inserted = True
            
            # Insert ratio-based code with proper indentation
            output.append('\n')
            output.append('  // ========== RATIO-BASED SPLIT REDUCTION ==========\n')
            for code_line in ratio_code.split('\n'):
                if code_line.strip():
                    output.append('  ' + code_line + '\n')
                else:
                    output.append('\n')
            output.append('\n')
            i += 1
            continue
        
        # If skipping old logic, look for the continuation point
        if skip_old_logic:
            if '// Based on the limitParallelLoops' in line:
                skip_old_logic = False
                output.append(line)
                i += 1
                continue
            else:
                i += 1
                continue
        
        output.append(line)
        i += 1
        continue
    
    # ==================== MATMUL-LIKE FUNCTION (gemm) ====================
    if in_mm_function:
        # Track when we see the getSizeAt lambda
        if 'auto getSizeAt = [&shapes]' in line:
            found_getsizeat_mm = True
        
        # Comment out constant declarations (they become unused when we replace the logic)
        if 'const int64_t ratioThreshold' in line or \
           'const int64_t largeKSize' in line or \
           'const int64_t largeMNSize' in line:
            output.append('  // ' + line.strip() + '  // OPTIMIZER: Disabled (unused)\n')
            i += 1
            continue
        
        # Comment out the ratio calculation (becomes unused when we replace the logic)
        if 'int64_t ratio = kSize /' in line and 'std::sqrt' in line:
            output.append('  // ' + line.strip() + '  // OPTIMIZER: Disabled (unused)\n')
            i += 1
            continue
        
        # Comment out early return checks (always, for both test types)
        if 'if (mSize > largeMNSize' in line:
            output.append('#if 0  // OPTIMIZER: Disabled early return\n')
            in_early_return_block = True
        elif 'if (ratio <= ratioThreshold' in line and 'kSize' in ''.join(lines[max(0,i-5):i+1]) and not in_early_return_block:
            output.append('#if 0  // OPTIMIZER: Disabled early return\n')
            in_early_return_block = True
        
        if in_early_return_block:
            output.append(line)
            if line.strip() == '}':
                output.append('#endif  // OPTIMIZER\n')
                in_early_return_block = False
            i += 1
            continue
        
        # For gemm tests: Apply ratio-based code after kSize calculation
        if test_type == 'gemm' and not mm_inserted and found_getsizeat_mm and 'int64_t kSize = getSizeAt(kDims)' in line:
            output.append(line)
            # Skip all old logic and insert new ratio-based code
            skip_old_logic = True
            mm_inserted = True
            
            # Insert ratio-based code with proper indentation
            output.append('\n')
            output.append('  // ========== RATIO-BASED SPLIT REDUCTION ==========\n')
            for code_line in ratio_code.split('\n'):
                if code_line.strip():
                    output.append('  ' + code_line + '\n')
                else:
                    output.append('\n')
            output.append('\n')
            i += 1
            continue
        
        # If skipping old logic, look for the continuation point
        if skip_old_logic:
            if '// Based on the limitParallelLoops' in line:
                skip_old_logic = False
                output.append(line)
                i += 1
                continue
            else:
                i += 1
                continue
        
        output.append(line)
        i += 1
        continue
    
    output.append(line)
    i += 1

# Validate insertion
if test_type == 'conv' and not wb_inserted:
    print("❌ Error: Could not find insertion point in getWeightBackwardReductionSizes")
    sys.exit(1)
if test_type == 'gemm' and not mm_inserted:
    print("❌ Error: Could not find insertion point in getMatmulLikeReductionSizes")
    sys.exit(1)

# Write modified file
with open(cpp_file, 'w') as f:
    f.writelines(output)

if test_type == 'conv':
    print("✓ Applied ratio-based code to getWeightBackwardReductionSizes")
else:
    print("✓ Applied ratio-based code to getMatmulLikeReductionSizes")
print("✓ Commented out early return checks in both functions")
EOFPYTHON

python3 "$RESULTS_DIR/_apply_ratio_code.py" "$CPP_FILE" "$RATIO_CODE_FILE" "$TEST_TYPE"

if [ $? -ne 0 ]; then
    echo "❌ Failed to apply ratio-based code"
    cp "${CPP_FILE}.backup_before_ratio" "$CPP_FILE"
    exit 1
fi

echo "✓ Ratio-based code applied"
echo ""

# Step 4: Build IREE
echo "Step 4: Building IREE with ratio-based approach..."
cd "$IREE_BUILD_DIR"
cmake --build . --target iree-compile 2>&1 | grep -E "(Building|error|warning)" || true

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "❌ Build failed!"
    cp "${CPP_FILE}.backup_before_ratio" "$CPP_FILE"
    exit 1
fi

echo "✓ Build successful"
echo ""

# Step 5: Run benchmark
echo "Step 5: Running benchmark with ratio-based approach..."
cd "$SCRIPT_DIR"

test_basename=$(basename "$TEST_FILE" .txt)
csv_file="$RESULTS_DIR/ratio_based_${test_basename}.csv"

# Create benchmark script
cat > "$RESULTS_DIR/_run_ratio_bench.sh" << EOFSCRIPT
#!/bin/bash
set -e

export PATH="$IREE_BUILD_DIR/tools:\$PATH"
export PYTHONPATH="$IREE_BUILD_DIR/compiler/bindings/python:\$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=$GPU_ID

cd "$TURBINE_DIR"
source .venv/bin/activate
if [ -f .env ]; then
    source .env
    export PYTHONPATH
fi

python3 iree/turbine/kernel/boo/driver/driver.py \\
    --commands-file="$TEST_FILE" \\
    --csv="$csv_file"
EOFSCRIPT

chmod +x "$RESULTS_DIR/_run_ratio_bench.sh"

if "$RESULTS_DIR/_run_ratio_bench.sh" > "$RESULTS_DIR/_ratio_bench_output.log" 2>&1; then
    echo "✓ Benchmark complete"
else
    echo "❌ Benchmark failed"
    echo "Last 20 lines of output:"
    tail -20 "$RESULTS_DIR/_ratio_bench_output.log" | sed "s/^/  /"
    cp "${CPP_FILE}.backup_before_ratio" "$CPP_FILE"
    exit 1
fi

# Step 6: Compare results
echo ""
echo "Step 6: Comparing ratio-based vs baseline..."
echo ""

# Save comparison to file and display
COMPARISON_FILE="$RESULTS_DIR/ratio_based_analysis.txt"

python3 - "$RESULTS_DIR" "$test_basename" << 'EOFCOMPARE' | tee "$COMPARISON_FILE"
import csv
import sys
from datetime import datetime

results_dir = sys.argv[1]
test_basename = sys.argv[2]

# Print header with metadata
print("="*80)
print("RATIO-BASED SPLIT REDUCTION - PERFORMANCE ANALYSIS")
print("="*80)
print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Test file: {test_basename}")
print(f"Results directory: {results_dir}")
print("="*80)
print()

# Read baseline
baseline_file = f"{results_dir}/limit_baseline_{test_basename}.csv"
baseline = {}
try:
    with open(baseline_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            test = row['arguments']
            runtime = float(row['iree_boo_experimental mean'])
            baseline[test] = runtime
except FileNotFoundError:
    print(f"Warning: Baseline file not found: {baseline_file}")
    sys.exit(0)

# Read ratio-based
ratio_file = f"{results_dir}/ratio_based_{test_basename}.csv"
ratio = {}
try:
    with open(ratio_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            test = row['arguments']
            runtime = float(row['iree_boo_experimental mean'])
            ratio[test] = runtime
except FileNotFoundError:
    print(f"Error: Ratio-based results not found: {ratio_file}")
    sys.exit(1)

# Compare
common = set(baseline.keys()) & set(ratio.keys())
if not common:
    print("Error: No common tests found")
    sys.exit(1)

total_baseline = sum(baseline[t] for t in common)
total_ratio = sum(ratio[t] for t in common)
speedup = total_baseline / total_ratio

improved = sum(1 for t in common if ratio[t] < baseline[t] * 0.99)
neutral = sum(1 for t in common if 0.99 <= ratio[t] / baseline[t] <= 1.01)
regressed = sum(1 for t in common if ratio[t] > baseline[t] * 1.01)

print("="*80)
print("RATIO-BASED vs BASELINE COMPARISON")
print("="*80)
print(f"Total tests: {len(common)}")
print(f"\nBaseline total:     {total_baseline:.2f} ms")
print(f"Ratio-based total:  {total_ratio:.2f} ms")
print(f"Overall speedup:    {speedup:.2f}x")
print(f"Overall improvement: {(1 - total_ratio/total_baseline)*100:+.2f}%")
print(f"\nTests improved:  {improved}/{len(common)} ({improved/len(common)*100:.1f}%)")
print(f"Tests neutral:   {neutral}/{len(common)} ({neutral/len(common)*100:.1f}%)")
print(f"Tests regressed: {regressed}/{len(common)} ({regressed/len(common)*100:.1f}%)")
print("="*80)

# Top improvements
improvements = []
for t in common:
    if ratio[t] < baseline[t]:
        speedup = baseline[t] / ratio[t]
        improvements.append((t, speedup, baseline[t], ratio[t]))

if improvements:
    improvements.sort(key=lambda x: x[1], reverse=True)
    print("\nTop 10 improvements:")
    for i, (test, spd, base, rat) in enumerate(improvements[:10], 1):
        print(f"  {i:2d}. {spd:6.2f}x - {test[:60]}...")

# Top regressions
regressions = []
for t in common:
    if ratio[t] > baseline[t]:
        speedup = baseline[t] / ratio[t]
        regressions.append((t, speedup, baseline[t], ratio[t]))

if regressions:
    regressions.sort(key=lambda x: x[1])
    print("\nTop 10 regressions:")
    for i, (test, spd, base, rat) in enumerate(regressions[:10], 1):
        print(f"  {i:2d}. {spd:6.2f}x - {test[:60]}...")

EOFCOMPARE

# Step 7: Automatic restore & cleanup
echo ""
echo "Step 7: Restoring original code and cleaning up..."
echo ""

echo "1. Restoring original C++ code..."
if [ -f "${CPP_FILE}.backup_before_ratio" ]; then
    cp "${CPP_FILE}.backup_before_ratio" "$CPP_FILE"
    echo "   ✓ Original code restored"
else
    echo "   ❌ Error: Backup file not found"
    exit 1
fi

echo ""
echo "2. Verifying restoration..."
if grep -q "int64_t outputSize = outputChannelSize \* batchSize \* imageSize \* depthSize;" "$CPP_FILE"; then
    echo "   ✓ Original outputSize-based code confirmed"
else
    echo "   ⚠️  Warning: Could not verify original code pattern"
fi

echo ""
echo "3. Deleting backup files..."
BACKUP_DIR="$(dirname "$CPP_FILE")"
BACKUP_COUNT=$(ls -1 "$BACKUP_DIR"/SetSplitReductionSizes.cpp.backup* 2>/dev/null | wc -l)

if [ "$BACKUP_COUNT" -gt 0 ]; then
    echo "   Found $BACKUP_COUNT backup file(s)"
    rm -f "$BACKUP_DIR"/SetSplitReductionSizes.cpp.backup*
    echo "   ✓ All backup files deleted"
else
    echo "   No backup files to delete"
fi

echo ""
echo "4. Rebuilding compiler with original code..."
cd "$IREE_BUILD_DIR"
if cmake --build . --target iree-compile -j$(nproc) 2>&1 | grep -E "(Building|error|\[.*%\])" | tail -10; then
    echo ""
    echo "   ✓ Rebuild successful"
else
    echo ""
    echo "   ❌ Rebuild failed!"
fi

echo ""
echo "═══════════════════════════════════════════════════════════════════════════════"
echo "  ✅ CLEANUP COMPLETE"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""
echo "  ✓ Original C++ code:  RESTORED"
echo "  ✓ Backup files:       DELETED"
echo "  ✓ Compiler:           REBUILT"
echo ""

echo ""
echo "================================================================================"
echo "  ✅ RATIO-BASED APPROACH TEST COMPLETE"
echo "================================================================================"
echo ""
echo "Results saved to:"
if $USE_PROVIDED_CODE; then
    echo "  - Provided code:      $PROVIDED_CODE_FILE"
    echo "  - Code copy:          $RATIO_CODE_FILE"
else
    echo "  - Generated code:     $RATIO_CODE_FILE"
fi
echo "  - Benchmark results:  $RESULTS_DIR/ratio_based_${test_basename}.csv"
echo "  - Analysis report:    $RESULTS_DIR/ratio_based_analysis.txt"
echo ""
