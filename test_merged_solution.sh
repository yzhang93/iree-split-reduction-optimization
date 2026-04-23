#!/bin/bash
set -e

# ============================================================================
# TEST MERGED HARDWARE SOLUTION
# ============================================================================
# This script:
# 1. Takes comprehensive_analysis.txt from multiple hardware systems
# 2. Calls merge_hardware_results.py to generate optimized C++ code
# 3. Backs up and applies the code to SetSplitReductionSizes.cpp
# 4. Builds the compiler
# 5. Runs benchmarks
# 6. Compares results with baseline
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IREE_DIR="$SCRIPT_DIR/../iree"
IREE_BUILD_DIR="$SCRIPT_DIR/../iree-build"
TURBINE_DIR="$SCRIPT_DIR/../iree-turbine"
GPU_ID=5

# Default values
OUTPUT_DIR=""
SYSTEM1_ANALYSIS=""
SYSTEM2_ANALYSIS=""
TEST_FILES=()
MERGE_PAIRS="32,64 128,256 512,1024"
KEEP_LIMITS="1 8 16 2048"
NO_DYNAMIC_LIMIT=""
CPP_OUTPUT_NAME="merged_ratio_based_code.cpp"

# Parse arguments
usage() {
    cat << EOF
Usage: $0 [OPTIONS] --system1 <analysis1> --system2 <analysis2> --output-dir <dir> --test-files <file1> [file2...]

Required arguments:
  --system1 <path>        Path to comprehensive_analysis.txt from system 1
  --system2 <path>        Path to comprehensive_analysis.txt from system 2
  --output-dir <path>     Directory for results and generated code
  --test-files <files>    One or more test files (e.g., all_weight_shapes.txt)

Optional arguments:
  --merge-pairs <pairs>   Pairs to merge (default: "32,64 128,256 512,1024")
  --keep-limits <limits>  Limits to keep (default: "1 8 16 2048")
  --no-dynamic-limit      Disable dynamic limit for low-ratio cases
  --gpu <id>              GPU ID to use (default: 5)
  --cpp-output <name>     C++ output filename (default: merged_ratio_based_code.cpp)
  -h, --help              Show this help message
  
Note: 
  • Code is automatically applied to SetSplitReductionSizes.cpp (backup created)
  • Original code is ALWAYS restored at the end with backup cleanup
  • Results are preserved in the output directory

Examples:
  # Basic usage
  $0 \\
    --system1 ../all_weight_shapes_results/comprehensive_analysis.txt \\
    --system2 ../all_weight_shapes_results/comprehensive_analysis_mi300.txt \\
    --output-dir ../merged_results \\
    --test-files ../all_weight_shapes.txt

  # Custom configuration
  $0 \\
    --system1 ../all_weight_shapes_results/comprehensive_analysis.txt \\
    --system2 ../all_weight_shapes_results/comprehensive_analysis_mi300.txt \\
    --output-dir ../merged_results \\
    --test-files ../all_weight_shapes.txt \\
    --merge-pairs "64,128 256,512" \\
    --keep-limits "1 16 32 1024"
EOF
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --system1)
            SYSTEM1_ANALYSIS="$2"
            shift 2
            ;;
        --system2)
            SYSTEM2_ANALYSIS="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --test-files)
            shift
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; do
                TEST_FILES+=("$1")
                shift
            done
            ;;
        --merge-pairs)
            MERGE_PAIRS="$2"
            shift 2
            ;;
        --keep-limits)
            KEEP_LIMITS="$2"
            shift 2
            ;;
        --no-dynamic-limit)
            NO_DYNAMIC_LIMIT="--no-dynamic-limit"
            shift
            ;;
        --gpu)
            GPU_ID="$2"
            shift 2
            ;;
        --cpp-output)
            CPP_OUTPUT_NAME="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "❌ Unknown option: $1"
            usage
            ;;
    esac
done

# Validate required arguments
if [ -z "$SYSTEM1_ANALYSIS" ] || [ -z "$SYSTEM2_ANALYSIS" ] || [ -z "$OUTPUT_DIR" ] || [ ${#TEST_FILES[@]} -eq 0 ]; then
    echo "❌ Error: Missing required arguments"
    echo ""
    usage
fi

# Convert to absolute paths
SYSTEM1_ANALYSIS_ABS="$(realpath "$SYSTEM1_ANALYSIS")"
SYSTEM2_ANALYSIS_ABS="$(realpath "$SYSTEM2_ANALYSIS")"
OUTPUT_DIR_ABS="$(realpath "$OUTPUT_DIR")"
TEST_FILES_ABS=()
for file in "${TEST_FILES[@]}"; do
    TEST_FILES_ABS+=("$(realpath "$file")")
done

# Validate files exist
if [ ! -f "$SYSTEM1_ANALYSIS_ABS" ]; then
    echo "❌ Error: System 1 analysis file not found: $SYSTEM1_ANALYSIS_ABS"
    exit 1
fi

if [ ! -f "$SYSTEM2_ANALYSIS_ABS" ]; then
    echo "❌ Error: System 2 analysis file not found: $SYSTEM2_ANALYSIS_ABS"
    exit 1
fi

for test_file in "${TEST_FILES_ABS[@]}"; do
    if [ ! -f "$test_file" ]; then
        echo "❌ Error: Test file not found: $test_file"
        exit 1
    fi
done

mkdir -p "$OUTPUT_DIR_ABS"

echo "================================================================================"
echo "  TESTING MERGED HARDWARE SOLUTION"
echo "================================================================================"
echo ""
echo "System 1 analysis: $SYSTEM1_ANALYSIS_ABS"
echo "System 2 analysis: $SYSTEM2_ANALYSIS_ABS"
echo "Output directory:  $OUTPUT_DIR_ABS"
echo "Test files:        ${TEST_FILES_ABS[@]}"
echo "Merge pairs:       $MERGE_PAIRS"
echo "Keep limits:       $KEEP_LIMITS"
echo "Dynamic limit:     $([ -z "$NO_DYNAMIC_LIMIT" ] && echo "enabled" || echo "disabled")"
echo ""

# ============================================================================
# STEP 1: Merge hardware results and generate C++ code
# ============================================================================
echo "────────────────────────────────────────────────────────────────────────────────"
echo "STEP 1: Merging hardware results"
echo "────────────────────────────────────────────────────────────────────────────────"
echo ""

cd "$SCRIPT_DIR"

# Build merge_hardware_results.py command
MERGE_CMD="python3 merge_hardware_results.py \
    --system1 \"$SYSTEM1_ANALYSIS_ABS\" \
    --system2 \"$SYSTEM2_ANALYSIS_ABS\" \
    --output-dir \"$OUTPUT_DIR_ABS\" \
    --output-file \"$CPP_OUTPUT_NAME\""

# Add merge pairs
for pair in $MERGE_PAIRS; do
    MERGE_CMD="$MERGE_CMD --merge-pairs \"$pair\""
done

# Add keep limits
MERGE_CMD="$MERGE_CMD --keep-limits $KEEP_LIMITS"

# Add no-dynamic-limit flag if set
if [ -n "$NO_DYNAMIC_LIMIT" ]; then
    MERGE_CMD="$MERGE_CMD $NO_DYNAMIC_LIMIT"
fi

# Execute merge command
eval $MERGE_CMD

if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to merge hardware results"
    exit 1
fi

CPP_FILE="$OUTPUT_DIR_ABS/$CPP_OUTPUT_NAME"
if [ ! -f "$CPP_FILE" ]; then
    echo "❌ Error: Generated C++ file not found: $CPP_FILE"
    exit 1
fi

echo ""
echo "✓ C++ code generated: $CPP_FILE"
echo ""

# ============================================================================
# STEP 2: Backup and apply code
# ============================================================================
echo "────────────────────────────────────────────────────────────────────────────────"
echo "STEP 2: Applying code to SetSplitReductionSizes.cpp"
echo "────────────────────────────────────────────────────────────────────────────────"
echo ""

TARGET_FILE="$IREE_DIR/compiler/src/iree/compiler/DispatchCreation/SetSplitReductionSizes.cpp"
BACKUP_FILE="$TARGET_FILE.backup_$(date +%Y%m%d_%H%M%S)"

if [ ! -f "$TARGET_FILE" ]; then
    echo "❌ Error: SetSplitReductionSizes.cpp not found at: $TARGET_FILE"
    exit 1
fi

# Create backup
cp "$TARGET_FILE" "$BACKUP_FILE"
echo "✓ Backup created: $BACKUP_FILE"
echo ""

# Automatically apply the generated code
echo "Applying generated C++ code automatically..."
echo ""

# Detect test type based on first test file content
TEST_TYPE="conv"
for test_file in "${TEST_FILES_ABS[@]}"; do
    if grep -q "aten::mm\|aten::addmm\|aten::bmm" "$test_file" 2>/dev/null; then
        TEST_TYPE="gemm"
        break
    elif [[ "$test_file" == *"gemm"* ]] || [[ "$test_file" == *"matmul"* ]]; then
        TEST_TYPE="gemm"
        break
    fi
done
echo "Detected test type: $TEST_TYPE"
echo ""

# Create a Python script to apply the code (using line-by-line approach)
cat > "$OUTPUT_DIR_ABS/_apply_code.py" << 'PYEOF'
import sys
import re

def apply_code(generated_file, target_file, test_type='conv'):
    # Read the generated code to extract the ratio-based logic
    with open(generated_file, 'r') as f:
        generated = f.read()
    
    # Extract the ratio calculation and decision tree from generated code
    # Look for the comment starting with "Extract tile sizes" and extract everything
    # from there until the closing brace of the decision tree
    ratio_match = re.search(
        r'(// Extract tile sizes from maybeSizes[\s\S]*?\n\} else \{[\s\S]*?\n\})',
        generated,
        re.DOTALL | re.MULTILINE
    )
    
    if not ratio_match:
        print("❌ Error: Could not extract ratio-based code from generated file")
        return False
    
    ratio_code = ratio_match.group(1).strip()
    print(f"✓ Extracted ratio-based code from generated file")
    print(f"  Code length: {len(ratio_code)} chars, {len(ratio_code.splitlines())} lines")
    
    # Read the target C++ file
    with open(target_file, 'r') as f:
        lines = f.readlines()
    
    # Process line by line to modify functions based on test type
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
            
            # Comment out early return checks (always)
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
            
            # For conv tests: Apply ratio-based code after batchSize calculation
            if test_type == 'conv' and not wb_inserted and found_getsizeat_wb and 'batchSize = getSizeAt(outputShape, batchPos)' in line:
                output.append(line)
                skip_old_logic = True
                wb_inserted = True
                
                output.append('\n')
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
            
            # Comment out early return checks (always)
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
                skip_old_logic = True
                mm_inserted = True
                
                output.append('\n')
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
        return False
    if test_type == 'gemm' and not mm_inserted:
        print("❌ Error: Could not find insertion point in getMatmulLikeReductionSizes")
        return False
    
    # Write the modified file
    with open(target_file, 'w') as f:
        f.writelines(output)
    
    if test_type == 'conv':
        print("✓ Applied ratio-based code to getWeightBackwardReductionSizes")
    else:
        print("✓ Applied ratio-based code to getMatmulLikeReductionSizes")
    print("✓ Commented out early return checks in both functions")
    return True

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python3 apply_code.py <generated_file> <target_file> [test_type]")
        sys.exit(1)
    
    test_type = sys.argv[3] if len(sys.argv) > 3 else 'conv'
    if apply_code(sys.argv[1], sys.argv[2], test_type):
        print("✓ Code applied successfully")
        sys.exit(0)
    else:
        sys.exit(1)
PYEOF

# Run the Python script
python3 "$OUTPUT_DIR_ABS/_apply_code.py" "$CPP_FILE" "$TARGET_FILE" "$TEST_TYPE"

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Automatic application failed!"
    echo "   Backup is available at: $BACKUP_FILE"
    echo ""
    echo "Manual edit required:"
    echo "  1. Open: $TARGET_FILE"
    echo "  2. Find getWeightBackwardReductionSizes() function"
    echo "  3. Replace limitParallelLoops calculation with code from: $CPP_FILE"
    exit 1
fi

rm -f "$OUTPUT_DIR_ABS/_apply_code.py"
echo ""

# ============================================================================
# STEP 3: Build compiler
# ============================================================================
echo "────────────────────────────────────────────────────────────────────────────────"
echo "STEP 3: Building IREE compiler"
echo "────────────────────────────────────────────────────────────────────────────────"
echo ""

cd "$IREE_BUILD_DIR"

echo "Building iree-compile..."
if cmake --build . --target iree-compile -j$(nproc) 2>&1 | grep -E "(Building|error|warning|\[.*%\])" | tail -20; then
    echo ""
    echo "✓ Build successful"
else
    echo "❌ Build failed!"
    echo ""
    echo "To restore original file:"
    echo "  cp $BACKUP_FILE $TARGET_FILE"
    exit 1
fi

# Verify new binary
COMPILER_BINARY="$IREE_BUILD_DIR/tools/iree-compile"
if [ -f "$COMPILER_BINARY" ]; then
    echo "✓ Compiler binary updated: $(ls -lh $COMPILER_BINARY | awk '{print $5, $6, $7, $8}')"
else
    echo "⚠️  Warning: Could not verify compiler binary"
fi

echo ""

# ============================================================================
# STEP 4: Run benchmarks
# ============================================================================
echo "────────────────────────────────────────────────────────────────────────────────"
echo "STEP 4: Running benchmarks"
echo "────────────────────────────────────────────────────────────────────────────────"
echo ""

for test_file in "${TEST_FILES_ABS[@]}"; do
    test_basename=$(basename "$test_file" .txt)
    csv_file="$OUTPUT_DIR_ABS/merged_${test_basename}.csv"
    
    echo "Testing $(basename $test_file)..."
    
    # Create temporary benchmark script
    cat > "$OUTPUT_DIR_ABS/_run_merged_bench.sh" << EOFSCRIPT
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
    
    chmod +x "$OUTPUT_DIR_ABS/_run_merged_bench.sh"
    
    if "$OUTPUT_DIR_ABS/_run_merged_bench.sh" 2>&1 | tee "$OUTPUT_DIR_ABS/_bench_output.log" | grep -E "(Testing|✓|✗|Error)" || true; then
        if [ -f "$csv_file" ]; then
            echo "  ✓ Completed $(basename $test_file)"
        else
            echo "  ❌ Benchmark completed but no CSV file created"
        fi
    else
        echo "  ❌ Benchmark failed for $(basename $test_file)"
        echo "     Check log: $OUTPUT_DIR_ABS/_bench_output.log"
    fi
    
    rm -f "$OUTPUT_DIR_ABS/_run_merged_bench.sh"
done

echo ""
echo "✓ Benchmarks complete"
echo ""

# ============================================================================
# STEP 5: Compare with baseline
# ============================================================================
echo "────────────────────────────────────────────────────────────────────────────────"
echo "STEP 5: Comparing with baseline"
echo "────────────────────────────────────────────────────────────────────────────────"
echo ""

# Find baseline directory (look for limit_baseline CSV files)
BASELINE_DIR=""
for dir in "$OUTPUT_DIR_ABS/../all_weight_shapes_results" "$OUTPUT_DIR_ABS/../prod_weight_shapes_results" "$OUTPUT_DIR_ABS"; do
    if [ -d "$dir" ] && ls "$dir"/limit_baseline_*.csv >/dev/null 2>&1; then
        BASELINE_DIR="$dir"
        break
    fi
done

if [ -z "$BASELINE_DIR" ]; then
    echo "⚠️  Warning: Could not find baseline results"
    echo "   Searched in:"
    echo "     - $OUTPUT_DIR_ABS/../all_weight_shapes_results"
    echo "     - $OUTPUT_DIR_ABS/../prod_weight_shapes_results"
    echo "     - $OUTPUT_DIR_ABS"
    echo ""
    echo "   Skipping comparison..."
else
    echo "Found baseline in: $BASELINE_DIR"
    echo ""
    
    # Run comparison
    python3 << PYEOF
import csv
import sys
from pathlib import Path
import glob

merged_dir = Path("$OUTPUT_DIR_ABS")
baseline_dir = Path("$BASELINE_DIR")

def read_results(filepath):
    results = {}
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_name = row['arguments']
            runtime = float(row['iree_boo_experimental mean'])
            results[test_name] = runtime
    return results

try:
    # Read merged results
    merged = {}
    for merged_file in glob.glob(str(merged_dir / "merged_*.csv")):
        merged_results = read_results(Path(merged_file))
        merged.update(merged_results)
    
    # Read baseline results
    baseline = {}
    for base_file in glob.glob(str(baseline_dir / "limit_baseline_*.csv")):
        base_results = read_results(Path(base_file))
        baseline.update(base_results)
    
    if not merged:
        print("❌ No merged results found")
        sys.exit(1)
    
    if not baseline:
        print("⚠️  No baseline results found for comparison")
        sys.exit(0)
    
    print("="*80)
    print("PERFORMANCE COMPARISON: MERGED vs BASELINE")
    print("="*80)
    print(f"\nTotal test cases: {len(merged)}")
    
    # Calculate metrics
    total_merged = sum(merged.values())
    total_baseline = sum(baseline.values())
    
    improvements = []
    regressions = []
    neutral = 0
    
    for test_name, merged_time in merged.items():
        if test_name in baseline:
            base_time = baseline[test_name]
            speedup = base_time / merged_time
            improvement_pct = ((base_time - merged_time) / base_time) * 100
            
            if speedup > 1.05:  # More than 5% improvement
                improvements.append((test_name, speedup, improvement_pct))
            elif speedup < 0.95:  # More than 5% regression
                regressions.append((test_name, speedup, improvement_pct))
            else:
                neutral += 1
    
    # Sort by speedup
    improvements.sort(key=lambda x: x[1], reverse=True)
    regressions.sort(key=lambda x: x[1])
    
    # Overall metrics
    overall_speedup = total_baseline / total_merged
    overall_improvement = ((total_baseline - total_merged) / total_baseline) * 100
    
    print(f"\n--- OVERALL METRICS ---")
    print(f"Baseline Total Runtime: {total_baseline:>10.2f} ms")
    print(f"Merged Total Runtime:   {total_merged:>10.2f} ms")
    print(f"Overall Speedup:        {overall_speedup:>10.2f}x")
    print(f"Overall Improvement:    {overall_improvement:>+10.2f}%")
    print(f"\nTests Improved:  {len(improvements):>3}/{len(merged)} ({len(improvements)/len(merged)*100:.1f}%)")
    print(f"Tests Neutral:   {neutral:>3}/{len(merged)} ({neutral/len(merged)*100:.1f}%)")
    print(f"Tests Regressed: {len(regressions):>3}/{len(merged)} ({len(regressions)/len(merged)*100:.1f}%)")
    
    # Top improvements
    if improvements:
        print(f"\n--- TOP 10 IMPROVEMENTS ---")
        for i, (test, speedup, improvement) in enumerate(improvements[:10], 1):
            print(f"{i:2d}. {speedup:6.2f}x ({improvement:+6.2f}%) - {test[:60]}")
    
    # Top regressions
    if regressions:
        print(f"\n--- TOP 10 REGRESSIONS ---")
        for i, (test, speedup, improvement) in enumerate(regressions[:10], 1):
            print(f"{i:2d}. {speedup:6.2f}x ({improvement:+6.2f}%) - {test[:60]}")
    
    print("\n" + "="*80)
    
    # Save comparison to file
    comparison_file = merged_dir / "comparison_vs_baseline.txt"
    with open(comparison_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("PERFORMANCE COMPARISON: MERGED vs BASELINE\n")
        f.write("="*80 + "\n\n")
        f.write(f"Baseline Total Runtime: {total_baseline:.2f} ms\n")
        f.write(f"Merged Total Runtime:   {total_merged:.2f} ms\n")
        f.write(f"Overall Speedup:        {overall_speedup:.2f}x\n")
        f.write(f"Overall Improvement:    {overall_improvement:+.2f}%\n\n")
        f.write(f"Tests Improved:  {len(improvements)}/{len(merged)} ({len(improvements)/len(merged)*100:.1f}%)\n")
        f.write(f"Tests Neutral:   {neutral}/{len(merged)} ({neutral/len(merged)*100:.1f}%)\n")
        f.write(f"Tests Regressed: {len(regressions)}/{len(merged)} ({len(regressions)/len(merged)*100:.1f}%)\n\n")
        
        if improvements:
            f.write("TOP 10 IMPROVEMENTS\n")
            f.write("-" * 80 + "\n")
            for i, (test, speedup, improvement) in enumerate(improvements[:10], 1):
                f.write(f"{i:2d}. {speedup:6.2f}x ({improvement:+6.2f}%) - {test}\n")
            f.write("\n")
        
        if regressions:
            f.write("TOP 10 REGRESSIONS\n")
            f.write("-" * 80 + "\n")
            for i, (test, speedup, improvement) in enumerate(regressions[:10], 1):
                f.write(f"{i:2d}. {speedup:6.2f}x ({improvement:+6.2f}%) - {test}\n")
    
    print(f"\n✓ Comparison saved to: {comparison_file}")

except FileNotFoundError as e:
    print(f"❌ Error: Could not find file - {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error analyzing results: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

PYEOF
fi

echo ""

# ============================================================================
# STEP 6: Summary
# ============================================================================
echo "================================================================================"
echo "  ✅ TESTING COMPLETE!"
echo "================================================================================"
echo ""
echo "Results directory: $OUTPUT_DIR_ABS"
echo ""
echo "Files generated:"
echo "  - $CPP_OUTPUT_NAME          (C++ code)"
echo "  - merge_summary.txt                (merge details)"
echo "  - merged_*.csv                     (benchmark results)"
echo "  - comparison_vs_baseline.txt       (performance comparison)"
echo ""

# ============================================================================
# AUTOMATIC RESTORE & CLEANUP
# ============================================================================
echo "────────────────────────────────────────────────────────────────────────────────"
echo "Restoring original code and cleaning up"
echo "────────────────────────────────────────────────────────────────────────────────"
echo ""

echo "1. Restoring original C++ code..."
if [ -f "$BACKUP_FILE" ]; then
    cp "$BACKUP_FILE" "$TARGET_FILE"
    echo "   ✓ Original code restored from: $(basename $BACKUP_FILE)"
else
    echo "   ❌ Error: Backup file not found: $BACKUP_FILE"
    exit 1
fi

echo ""
echo "2. Verifying restoration..."
if grep -q "int64_t outputSize = outputChannelSize \* batchSize \* imageSize \* depthSize;" "$TARGET_FILE"; then
    echo "   ✓ Original outputSize-based code confirmed"
else
    echo "   ⚠️  Warning: Could not verify original code pattern"
fi

echo ""
echo "3. Deleting all backup files..."
BACKUP_DIR="$(dirname "$TARGET_FILE")"
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
echo "Results are saved in: $OUTPUT_DIR_ABS"
echo ""
