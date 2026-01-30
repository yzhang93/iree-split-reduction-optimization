#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IREE_DIR="$SCRIPT_DIR/../iree"
IREE_BUILD_DIR="$SCRIPT_DIR/../iree-build"
TURBINE_DIR="$SCRIPT_DIR/../iree-turbine"
GPU_ID=5

# Parse arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <results_dir> <test_file1> [test_file2 ...]"
    echo ""
    echo "Example:"
    echo "  $0 ../prod_weight_shapes_results ../prod_weight_shapes_conv.txt"
    exit 1
fi

RESULTS_DIR="$1"
shift
TEST_FILES=("$@")

# Convert to absolute paths
RESULTS_DIR_ABS="$(realpath "$RESULTS_DIR")"
TEST_FILES_ABS=()
for file in "${TEST_FILES[@]}"; do
    TEST_FILES_ABS+=("$(realpath "$file")")
done

mkdir -p "$RESULTS_DIR_ABS"

echo "================================================================================"
echo "  TESTING OPTIMIZED C++ CONFIGURATION"
echo "================================================================================"
echo ""
echo "Results directory: $RESULTS_DIR_ABS"
echo "Test files: ${TEST_FILES_ABS[@]}"
echo ""
echo "Building IREE with optimized settings..."
echo ""

cd "$IREE_BUILD_DIR"
cmake --build . --target iree-compile 2>&1 | grep -E "(Building|error|warning)" || true

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "❌ Build failed!"
    exit 1
fi

echo "✓ Build successful"
echo ""
echo "Running benchmarks with optimized configuration..."
echo ""

# Run benchmarks for each test file
for test_file in "${TEST_FILES_ABS[@]}"; do
    if [ ! -f "$test_file" ]; then
        echo "⚠️  Test file not found: $test_file"
        continue
    fi
    
    test_basename=$(basename "$test_file" .txt)
    csv_file="$RESULTS_DIR_ABS/optimized_config_${test_basename}.csv"
    
    echo "  Testing $(basename $test_file)..."
    
    # Create temporary benchmark script for complete environment isolation
    cat > "$RESULTS_DIR_ABS/_run_opt_bench.sh" << EOFSCRIPT
#!/bin/bash
set -e

# Set up IREE environment
export PATH="$IREE_BUILD_DIR/tools:\\\$PATH"
export PYTHONPATH="$IREE_BUILD_DIR/compiler/bindings/python:\\\$PYTHONPATH"
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
    
    chmod +x "$RESULTS_DIR_ABS/_run_opt_bench.sh"
    
    
    # Run benchmark and show errors if any
    if "$RESULTS_DIR_ABS/_run_opt_bench.sh" 2>&1 | tee "$RESULTS_DIR_ABS/_bench_output.log"; then
        if [ -f "$csv_file" ]; then
            echo "  ✓ Completed $(basename $test_file)"
        else
            echo "  ❌ Benchmark completed but no CSV file created"
        fi
    else
        echo "  ❌ Benchmark failed for $(basename $test_file)"
        echo "     Last 20 lines of output:"
        if [ -f "$RESULTS_DIR_ABS/_bench_output.log" ]; then
            tail -20 "$RESULTS_DIR_ABS/_bench_output.log" | sed "s/^/     /"
        fi
    fi
    
    rm -f "$RESULTS_DIR_ABS/_run_opt_bench.sh"
done

echo ""
echo "✓ Optimized configuration benchmarks complete"
echo ""

# Update comprehensive analysis with optimized results
echo "Updating comprehensive analysis with validated results..."
echo ""

cd "$SCRIPT_DIR"
python3 analyze_results.py \
    --results-file "$RESULTS_DIR_ABS/limitParallelLoops_sweep_results.json" \
    --output-dir "$RESULTS_DIR_ABS" \
    --baseline-limit baseline \
    --optimized-csv "$RESULTS_DIR_ABS/optimized_config_*.csv"

echo ""
echo "✓ Analysis updated with optimized configuration results"
echo ""

# Display summary comparison
python3 << PYEOF
import csv
import sys
from pathlib import Path
import glob

results_dir = Path("$RESULTS_DIR_ABS")

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
    # Read all optimized results
    optimized = {}
    for opt_file in glob.glob(str(results_dir / "optimized_config_*.csv")):
        opt_results = read_results(Path(opt_file))
        optimized.update(opt_results)
    
    # Read all baseline results
    baseline = {}
    for base_file in glob.glob(str(results_dir / "limit_baseline_*.csv")):
        base_results = read_results(Path(base_file))
        baseline.update(base_results)
    
    print("="*80)
    print("PERFORMANCE COMPARISON: OPTIMIZED vs BASELINE")
    print("="*80)
    print(f"\nTotal test cases: {len(optimized)}")
    
    # Calculate metrics
    total_optimized = sum(optimized.values())
    total_baseline = sum(baseline.values())
    
    improvements = []
    regressions = []
    
    for test_name, opt_time in optimized.items():
        if test_name in baseline:
            base_time = baseline[test_name]
            speedup = base_time / opt_time
            improvement_pct = ((base_time - opt_time) / base_time) * 100
            
            if speedup > 1.05:  # More than 5% improvement
                improvements.append((test_name, speedup, improvement_pct))
            elif speedup < 0.95:  # More than 5% regression
                regressions.append((test_name, speedup, improvement_pct))
    
    # Sort by speedup
    improvements.sort(key=lambda x: x[1], reverse=True)
    regressions.sort(key=lambda x: x[1])
    
    # Overall metrics
    overall_speedup = total_baseline / total_optimized
    overall_improvement = ((total_baseline - total_optimized) / total_baseline) * 100
    
    print(f"\n--- OVERALL METRICS ---")
    print(f"Baseline Total Runtime:  {total_baseline:.2f} ms")
    print(f"Optimized Total Runtime: {total_optimized:.2f} ms")
    print(f"Overall Speedup:         {overall_speedup:.2f}x")
    print(f"Overall Improvement:     {overall_improvement:+.2f}%")
    print(f"\nTests Improved:  {len(improvements)}/{len(optimized)} ({len(improvements)/len(optimized)*100:.1f}%)")
    print(f"Tests Regressed: {len(regressions)}/{len(optimized)} ({len(regressions)/len(optimized)*100:.1f}%)")
    
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
    
except FileNotFoundError as e:
    print(f"Error: Could not find file - {e}")
    sys.exit(1)
except Exception as e:
    print(f"Error analyzing results: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

PYEOF

echo ""
echo "================================================================================"
echo "  ✅ VALIDATION COMPLETE!"
echo "================================================================================"
echo ""
echo "Results saved to: $RESULTS_DIR_ABS/"
echo ""
echo "Check PART 6 in: $RESULTS_DIR_ABS/comprehensive_analysis.txt"
echo "  This section now shows the validated optimized performance!"
echo ""