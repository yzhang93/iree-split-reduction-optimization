# Split Reduction Optimization - Quick Start Guide

## Overview

This toolkit optimizes IREE's `limitParallelLoops` constants for split reduction operations. It automatically tests different configurations, analyzes performance, generates C++ code, and validates the results.

**Proven Results:** 20.3x overall speedup with zero regressions.

---

## Prerequisites

### Required Components

1. **IREE** compiler source and build directory
2. **iree-turbine** repository with virtual environment
3. **PyTorch** installed in turbine's venv
4. **GPU** (tested on MI300)

### Verify Setup

```bash
cd /home/vivizhan/split_reduction_optimization
./check_setup.sh
```

Expected output:
```
✓ C++ file found
✓ IREE build directory found
✓ iree-turbine directory found
✓ Virtual environment found
✓ PyTorch installed in venv
✓ Setup verification passed!
```

---

## Running Optimization

### Command Syntax

```bash
./run_parameter_search.sh <mode> <test_file>
```

**Modes:**
- `quick` - Tests 4 limits (1, 64, 128, 256) + baseline - ~5 minutes
- `full` - Tests 10 limits (1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048) + baseline - ~1 hour
- `analyze` - Re-analyze existing results without re-running tests
- `test-optimized` - Apply and test the recommended configuration

### Examples

```bash
# Quick test for fast iteration
./run_parameter_search.sh quick ~/small_test.txt

# Full sweep for production (includes automatic validation)
./run_parameter_search.sh full ~/prod_weight_shapes_conv.txt

# Re-analyze without re-running benchmarks
./run_parameter_search.sh analyze

# Test only the optimized configuration (after analysis)
./run_parameter_search.sh test-optimized ~/prod_weight_shapes_conv.txt
```

### What Happens

**`quick` and `full` modes:**
1. ✓ Run baseline benchmark (original C++ code)
2. ✓ Test each limit value in a separate process (complete isolation)
3. ✓ Generate JSON summary of all results
4. ✓ Analyze results and create C++ recommendations
5. ✓ **Automatically apply recommendations and test optimized configuration**
6. ✓ **Validate performance vs baseline**
7. ✓ Generate comprehensive analysis report with Part 6 validation

**`analyze` mode:**
- Re-analyze existing results without re-running tests
- Update comprehensive_analysis.txt with latest analysis logic

**`test-optimized` mode:**
- Apply Part 4 recommendations from comprehensive_analysis.txt
- Build IREE with optimized configuration
- Run benchmarks and validate performance
- Update comprehensive_analysis.txt with Part 6 results

**All results saved to:** `../<test_name>_results/` (derived from input file name)

Example: If testing with `prod_weight_shapes_conv.txt`, results go to `../prod_weight_shapes_results/`

---

## Understanding Results

### Main Output File

**File:** `../<test_name>_results/comprehensive_analysis.txt`

Example: `../prod_weight_shapes_results/comprehensive_analysis.txt`

This single file contains everything:

### Part 1: Performance Summary by Limit

Shows statistics for each tested limit:
```
limitParallelLoops = 8:
  Geometric Mean:   157.50 ms
  Arithmetic Mean:  895.97 ms
  Median:           220.32 ms
  P95:              2989.33 ms
  Total Runtime:    138876 ms
```

**Look for:** Limit with lowest Geometric Mean

### Part 2: Baseline Comparison

Win rates and improvements vs baseline:
```
limit=8 vs baseline:
  Win Rate:         92.3% (144/155 tests faster)
  Avg Improvement:  87.5%
  Max Improvement:  99.8%
```

**Look for:** High win rate (>80%)

### Part 3: Cluster Analysis

Which tests perform best with each limit:
```
Cluster for limitParallelLoops = 512:
  Count: 7 tests
  OutputSize range: 256 - 1,024
  Avg speedup: 262.57x
```

**Look for:** Clear separation by output size

### Part 4: C++ Code Recommendations ⭐

**THIS IS WHAT YOU NEED!**

#### Part A: Early Return Thresholds

Separate recommendations for convolution and matmul:

```cpp
FOR CONVOLUTION (getWeightBackwardReductionSizes):
  const int64_t largeParallelSize = 512;
  const int64_t largeReductionSize = 368;
  const int64_t ratioThreshold = 28;
```

#### Part B: limitParallelLoops Logic

Monotonically decreasing thresholds:

```cpp
int64_t limitParallelLoops;
if (outputSize < 32 * 32) {           // Small outputs
  limitParallelLoops = 512;
} else if (outputSize < 128 * 128) {  // Medium outputs
  limitParallelLoops = 64;
} else if (outputSize < 1024 * 1024) { // Large outputs
  limitParallelLoops = 32;
} else if (outputSize < 4096 * 4096) { // Very large outputs
  limitParallelLoops = 8;
} else {                               // Largest outputs
  limitParallelLoops = std::min<int64_t>(16, tileSizes[0]);
}
```

### Part 5: Top 20 Highest Speedups

```
#1: convbfp16 -n 16 -c 2048 ...
    Best limit: 1, Runtime: 168.95ms, Speedup: 3248.50x
```

**Insight:** See which operations benefit most

### Part 6: Optimized Configuration - Validated Performance ⭐

**NEW!** Actual validation of the recommended configuration:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    BASELINE vs OPTIMIZED                                 │
├─────────────────────────────────────────────────────────────────────────┤
│ Total Tests:                 155                                       │
│ Baseline Runtime:        1012435 ms  (~16.9 min)                  │
│ Optimized Runtime:         49834 ms  (~0.8 min)                   │
│ Overall Speedup:           20.32x                                      │
│ Overall Improvement:       95.08%                                      │
│ Geometric Mean:             5.07x                                      │
├─────────────────────────────────────────────────────────────────────────┤
│ Tests Improved:              144 ( 92.9%)                            │
│ Tests Neutral (±5%):          11 (  7.1%)                            │
│ Tests Regressed:               0 (  0.0%)                            │
└─────────────────────────────────────────────────────────────────────────┘

✅ EXCELLENT RESULTS - READY FOR PRODUCTION
```

**This proves the recommendations work!**

---

## Applying Recommendations

### Step 1: Copy C++ Code

From Part 4 of `comprehensive_analysis.txt`, copy:
1. Early return thresholds (Part A)
2. limitParallelLoops logic (Part B)

### Step 2: Edit C++ File

**File:** `iree/compiler/src/iree/compiler/DispatchCreation/SetSplitReductionSizes.cpp`

**For Convolution** (`getWeightBackwardReductionSizes` function):
- Lines ~231-233: Early return constants
- Lines ~259-271: limitParallelLoops logic

**For Matmul** (`getMatmulLikeReductionSizes` function):
- Lines ~358-360: Early return constants
- Lines ~384-396: limitParallelLoops logic

### Step 3: Rebuild IREE

```bash
cd /path/to/iree-build
cmake --build . --target iree-compile
```

### Step 4: Verify

Run your benchmarks and compare against Part 6 of the analysis to confirm the improvements.

---

## Advanced Usage

### Create Test Subset

```bash
# Create small test file (first N lines)
./create_small_test.sh ~/prod_weight_shapes_conv.txt 10 ~/small_test.txt
```

### Split by Operation Type

```bash
# Separate convolution and matmul operations
./split_test_files.sh ~/mixed_operations.txt
# Creates: *_conv.txt and *_matmul.txt
```

### Re-analyze Existing Results

```bash
# Update analysis without re-running benchmarks
./run_parameter_search.sh analyze
```

### Test Specific Configuration

```bash
# Only test the optimized configuration (requires existing analysis)
./run_parameter_search.sh test-optimized ~/test.txt
```

### Specify GPU

```bash
# Edit GPU_ID in run_parameter_search.sh
GPU_ID=5  # Default
```

### Smart Caching

By default, if result files already exist, the script skips re-running them:
- Speeds up interrupted runs
- Allows testing different test files incrementally
- To force re-run a specific limit, delete its CSV file manually

```bash
# Example: Force re-run baseline only
rm ../prod_weight_shapes_results/limit_baseline_*.csv
./run_parameter_search.sh full ~/prod_weight_shapes_conv.txt
```

---

## Interpreting Metrics

### Geometric Mean
- **Best for:** Overall performance comparison
- **Why:** Not skewed by outliers, reflects consistent gains
- **Use:** Compare different limits, lower is better

### Win Rate
- **Best for:** Reliability assessment
- **Why:** Shows how often a configuration wins
- **Use:** Choose configurations with >80% win rate

### Overall Speedup
- **Best for:** Total time savings
- **Why:** Direct measure of efficiency gain
- **Use:** Justify the optimization effort

### Regressions
- **Best for:** Risk assessment
- **Why:** Shows if any tests get worse
- **Use:** 0 regressions = safe to deploy

---

## Troubleshooting

### "Build failed" Error

**Cause:** C++ code modification error

**Fix:**
```bash
# Restore original C++ file
cd /path/to/iree/compiler/src/iree/compiler/DispatchCreation
cp SetSplitReductionSizes.cpp.backup SetSplitReductionSizes.cpp

# Rebuild
cd /path/to/iree-build
cmake --build . --target iree-compile
```

### "ModuleNotFoundError: torch" Error

**Cause:** Virtual environment not activated

**Fix:**
```bash
cd /path/to/iree-turbine
source .venv/bin/activate
# Verify: python -c "import torch; print(torch.__version__)"
```

### Results Look Identical

**Cause:** Python module caching (should not happen with process isolation)

**Fix:**
```bash
# Force clean re-run
./run_parameter_search.sh full ~/test.txt --force-rerun
```

### Analysis Shows inf/nan Values

**Cause:** Some tests failed or had zero runtime

**Fix:** Check CSV files in results directory for errors, exclude problematic tests

---

## Best Practices

### 1. Start with Quick Mode
Test with `quick` mode first to verify everything works before committing to a full sweep.

### 2. Use Representative Workloads
Include diverse operation sizes in your test file for robust optimization.

### 3. Separate Conv and Matmul
For best results, optimize convolution and matmul operations separately:
```bash
./split_test_files.sh ~/all_ops.txt
./run_parameter_search.sh full ~/all_ops_conv.txt
./run_parameter_search.sh full ~/all_ops_matmul.txt
```

### 4. Validate Before Deploying
Always check Part 6 of the analysis to confirm actual improvements:
- Look for 0 regressions
- Verify geometric mean improvement
- Check top improvements align with expectations

### 5. Archive Results
```bash
# Save results with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
cp -r ../prod_weight_shapes_results ../results_archive_$TIMESTAMP
```

---

## Performance Expectations

### Quick Mode (~5 minutes)
- Tests: 4 limits (1, 64, 128, 256) + baseline
- Good for: Fast iteration, verification
- Use when: Developing, debugging, testing changes

### Full Mode (~1 hour)
- Tests: 10 limits (1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048) + baseline + optimized validation
- Good for: Production optimization
- Use when: Final optimization, benchmarking for deployment

### Analyze Mode (~10 seconds)
- No new tests: Re-analyzes existing results only
- Good for: Testing analysis improvements, quick insights
- Use when: Results exist but want updated analysis

### Test-Optimized Mode (~10 minutes)
- Tests: Only the recommended optimized configuration
- Good for: Validating recommendations, quick verification
- Use when: Want to validate specific recommendations without full sweep

### Expected Improvements
Based on validated results:
- **Typical:** 5-10x geometric mean speedup
- **Good:** 15-20x overall speedup
- **Excellent:** >20x overall with 0 regressions
- **Best cases:** Individual operations 100-600x faster

---

## Next Steps

1. **Review** Part 6 of `comprehensive_analysis.txt` for validated performance
2. **Copy** C++ code from Part 4
3. **Apply** to SetSplitReductionSizes.cpp
4. **Rebuild** IREE
5. **Test** in production
6. **Monitor** performance

For technical details, see [IMPLEMENTATION.md](IMPLEMENTATION.md).

---

## Summary

**One command:**
```bash
./run_parameter_search.sh full ~/test.txt
```

**One file to read:**
```bash
cat ../<test_name>_results/comprehensive_analysis.txt
# Example: cat ../prod_weight_shapes_results/comprehensive_analysis.txt
```

**One section to apply:** Part 4 (C++ Code)

**One metric to check:** Part 6 (Validation)

**Result:** Production-ready optimization with verified 20x speedup! 🚀
