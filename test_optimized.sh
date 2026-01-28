#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IREE_DIR="$SCRIPT_DIR/../iree"
IREE_BUILD_DIR="$SCRIPT_DIR/../iree-build"
TURBINE_DIR="$SCRIPT_DIR/../iree-turbine"
RESULTS_DIR="$SCRIPT_DIR/../optimized_results"
TEST_FILE="/home/vivizhan/all_weight_shapes_conv.txt"

mkdir -p "$RESULTS_DIR"

echo "================================================================================"
echo "  TESTING OPTIMIZED C++ CONFIGURATION"
echo "================================================================================"
echo ""
echo "Building IREE with optimized settings..."
echo ""

cd "$IREE_BUILD_DIR"
cmake --build . --target iree-compile 2>&1 | grep -E "(Building|error|warning)" || true

if [ $? -ne 0 ]; then
    echo "❌ Build failed!"
    exit 1
fi

echo "✓ Build successful"
echo ""
echo "Running benchmarks..."
echo ""

# Set up environment and run benchmark
export PATH="$IREE_BUILD_DIR/tools:$PATH"
export CUDA_VISIBLE_DEVICES=5

cd "$TURBINE_DIR"
source .venv/bin/activate
source .env

# Run benchmark
RESULT_FILE="$RESULTS_DIR/optimized_config.csv"
python3 iree/turbine/kernel/boo/driver/driver.py \
    --commands-file="$TEST_FILE" \
    --csv="$RESULT_FILE"

echo "✓ Benchmark complete"
echo ""
echo "Results saved to: $RESULT_FILE"
echo ""

# Parse results and compare
python3 << 'PYEOF'
import csv
import sys
from pathlib import Path

# Read optimized results
optimized_file = Path("/home/vivizhan/optimized_results/optimized_config.csv")
baseline_file = Path("/home/vivizhan/limitParallelLoops_sweep_results/limit_baseline_all_weight_shapes_conv.csv")

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
    optimized = read_results(optimized_file)
    baseline = read_results(baseline_file)
    
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
echo "  TESTING COMPLETE"
echo "================================================================================"
