#!/usr/bin/env python3
"""
Single limitParallelLoops Test Runner

This script tests ONE limitParallelLoops value and exits.
This ensures complete process isolation between different values.

To test multiple values, call this script multiple times (see run_parameter_search.sh).
"""

import argparse
import sys
import shutil
import subprocess
import csv
from pathlib import Path
from typing import Optional, List, Dict

class LimitParallelLoopsOptimizer:
    """Optimizer for testing different limitParallelLoops values"""
    
    def __init__(self, cpp_file: str, iree_build_dir: str, turbine_dir: str, results_dir: str):
        self.cpp_file = Path(cpp_file)
        self.iree_build_dir = Path(iree_build_dir)
        self.turbine_dir = Path(turbine_dir)
        self.results_dir = Path(results_dir)
        self.backup_file = self.cpp_file.with_suffix('.cpp.backup')
        self.gpu_id = 5
        
        # Create backup if doesn't exist
        if not self.backup_file.exists():
            shutil.copy2(self.cpp_file, self.backup_file)
        
        # Create results directory
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def set_fixed_limitParallelLoops(self, limit_value: int, mode: str = 'both'):
        """Modify C++ file to use a fixed limitParallelLoops value
        
        This also comments out early return checks so we can test all limitParallelLoops
        values without being skipped by the heuristics.
        """
        with open(self.cpp_file, 'r') as f:
            lines = f.readlines()
        
        modified_lines = []
        in_wb_function = False
        in_mm_function = False
        function_brace_depth = 0
        function_started = False  # True once we see the first { after function declaration
        skip_until_closing_brace = False
        skip_brace_depth = 0
        in_early_return_block = False  # Track if we're in an early return if statement
        constants_declared = {'wb': False, 'mm': False}  # Track if we've added suppression after constants
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Track which function we're in based on line content
            if 'getWeightBackwardReductionSizes' in line and '(' in line:
                in_wb_function = True
                in_mm_function = False
                function_brace_depth = 0
                function_started = False
            elif 'getMatmulLikeReductionSizes' in line and '(' in line:
                in_mm_function = True
                in_wb_function = False
                function_brace_depth = 0
                function_started = False
            
            # Track braces to know when we exit the function
            if (in_wb_function or in_mm_function) and not skip_until_closing_brace:
                # Once we see a {, the function body has started
                if '{' in line:
                    function_started = True
                
                if function_started:
                    function_brace_depth += line.count('{') - line.count('}')
                    if function_brace_depth <= 0:
                        in_wb_function = False
                        in_mm_function = False
                        function_brace_depth = 0
                        function_started = False
            
            # Handle skipping original if-else logic
            if skip_until_closing_brace:
                # Track braces to find the end of the if-else block
                skip_brace_depth += line.count('{') - line.count('}')
                modified_lines.append(line)
                
                # When we're back to depth 0, we've exited the if-else block
                if skip_brace_depth <= 0:
                    modified_lines.append('#endif  // OPTIMIZER\n')
                    skip_until_closing_brace = False
                    skip_brace_depth = 0
                i += 1
                continue
            
            # Comment out early return checks (so we can test all limitParallelLoops values)
            # Detect start of early return blocks by looking for the if statement
            # Weight backward: "if (outputChannelSize >= largeParallelSize"
            if in_wb_function and function_started and 'if (outputChannelSize >= largeParallelSize' in line:
                modified_lines.append('#if 0  // OPTIMIZER: Disabled early return to test all limits\n')
                in_early_return_block = True
            
            # Weight backward: "if (ratio <= ratioThreshold"
            elif in_wb_function and function_started and 'if (ratio <= ratioThreshold' in line:
                modified_lines.append('#if 0  // OPTIMIZER: Disabled early return to test all limits\n')
                in_early_return_block = True
            
            # Matmul-like: "if (mSize > largeMNSize"
            elif in_mm_function and function_started and 'if (mSize > largeMNSize' in line:
                modified_lines.append('#if 0  // OPTIMIZER: Disabled early return to test all limits\n')
                in_early_return_block = True
            
            # Matmul-like: "if (ratio <= ratioThreshold"
            elif in_mm_function and function_started and 'if (ratio <= ratioThreshold' in line:
                modified_lines.append('#if 0  // OPTIMIZER: Disabled early return to test all limits\n')
                in_early_return_block = True
            
            # If in early return block, append lines and look for the closing brace
            if in_early_return_block:
                modified_lines.append(line)
                # Look for the closing brace of the if block (single '}' line or '  }')
                if line.strip() == '}':
                    modified_lines.append('#endif  // OPTIMIZER\n')
                    in_early_return_block = False
                i += 1
                continue
            
            # Add unused variable suppressions after constant declarations
            # Weight backward function constants
            if in_wb_function and not constants_declared['wb'] and 'const int64_t ratioThreshold = 64;' in line:
                modified_lines.append(line)
                # Add blank line and suppressions
                modified_lines.append('\n')
                modified_lines.append('  // OPTIMIZER: Suppress unused variable warnings\n')
                modified_lines.append('  (void)largeParallelSize;\n')
                modified_lines.append('  (void)largeReductionSize;\n')
                modified_lines.append('  (void)ratioThreshold;\n')
                constants_declared['wb'] = True
                i += 1
                continue
            
            # Matmul-like function constants (add suppressions after the last one)
            if in_mm_function and not constants_declared['mm'] and 'const int64_t largeMNSize = 1024;' in line:
                modified_lines.append(line)
                # Add blank line and suppressions
                modified_lines.append('\n')
                modified_lines.append('  // OPTIMIZER: Suppress unused variable warnings\n')
                modified_lines.append('  (void)ratioThreshold;\n')
                modified_lines.append('  (void)largeKSize;\n')
                modified_lines.append('  (void)largeMNSize;\n')
                constants_declared['mm'] = True
                i += 1
                continue
            
            # Suppress ratio variable in weight backward function (used only in early return)
            if in_wb_function and 'int64_t ratio = reductionSize / std::sqrt(outputChannelSize * batchSize);' in line:
                modified_lines.append(line)
                modified_lines.append('  (void)reductionSize;  // OPTIMIZER: Suppress unused warning\n')
                modified_lines.append('  (void)ratio;  // OPTIMIZER: Suppress unused warning\n')
                i += 1
                continue
            
            # Suppress ratio variable in matmul-like function (used only in early return)
            if in_mm_function and 'int64_t ratio = kSize / std::sqrt(mSize * nSize) / batchSize;' in line:
                modified_lines.append(line)
                modified_lines.append('  (void)ratio;  // OPTIMIZER: Suppress unused warning\n')
                i += 1
                continue
            
            # Modify weight backward function
            if in_wb_function and function_started and 'int64_t limitParallelLoops;' in line:
                if mode == 'conv' or mode == 'mixed' or mode == 'both':
                    modified_lines.append(line)
                    modified_lines.append(f'  // OPTIMIZER: Fixed value for testing\n')
                    modified_lines.append(f'  limitParallelLoops = {limit_value};\n')
                    modified_lines.append(f'  (void)outputSize;  // Suppress unused warning\n')
                    modified_lines.append(f'#if 0  // Original code disabled\n')
                    skip_until_closing_brace = True
                    skip_brace_depth = 0
                    i += 1
                    continue
                else:
                    modified_lines.append(line)
                    i += 1
                    continue
            
            # Modify matmul-like function
            if in_mm_function and function_started and 'int64_t limitParallelLoops;' in line:
                if mode == 'matmul' or mode == 'mixed' or mode == 'both' or mode == 'conv':
                    modified_lines.append(line)
                    modified_lines.append(f'  // OPTIMIZER: Fixed value for testing\n')
                    modified_lines.append(f'  limitParallelLoops = {limit_value};\n')
                    modified_lines.append(f'  (void)outputSize;  // Suppress unused warning\n')
                    modified_lines.append(f'  (void)kSize;       // Suppress unused warning\n')
                    modified_lines.append(f'#if 0  // Original code disabled\n')
                    skip_until_closing_brace = True
                    skip_brace_depth = 0
                    i += 1
                    continue
                else:
                    modified_lines.append(line)
                    i += 1
                    continue
            
            # Regular line
            modified_lines.append(line)
            i += 1
        
        # Write modified file
        with open(self.cpp_file, 'w') as f:
            f.writelines(modified_lines)
    
    def restore_original(self):
        """Restore original C++ file"""
        if self.backup_file.exists():
            shutil.copy2(self.backup_file, self.cpp_file)
    
    def build_iree(self) -> bool:
        """Build IREE with current C++ modifications"""
        try:
            result = subprocess.run(
                ['cmake', '--build', str(self.iree_build_dir), '--target', 'iree-compile'],
                capture_output=True,
                text=True,
                timeout=600
            )
            if result.returncode == 0:
                print("Build successful")
                return True
            else:
                print(f"Build failed: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            print("Build timed out")
            return False
    
    def check_results_exist(self, test_files: List[Path], limit_value: int) -> bool:
        """Check if all result files already exist"""
        all_exist = True
        for test_file in test_files:
            test_basename = test_file.stem
            csv_file = self.results_dir / f"limit_{limit_value}_{test_basename}.csv"
            if not csv_file.exists():
                all_exist = False
                break
        return all_exist
    
    def run_benchmark(self, test_file: Path, limit_value: int, force_rerun: bool = False) -> Optional[Path]:
        """Run benchmark and return path to CSV results"""
        test_basename = test_file.stem
        csv_file = self.results_dir / f"limit_{limit_value}_{test_basename}.csv"
        
        # Skip if results already exist (unless force_rerun is True)
        # Note: This shouldn't happen since we check/delete at the top level
        if csv_file.exists() and not force_rerun:
            return csv_file
        
        # Show progress
        print(f"  Running benchmarks for {test_file.name}...")
        
        # Create a bash script with proper environment setup
        setup_script = f"""#!/bin/bash
set -e

# Get absolute paths
IREE_BUILD_DIR="{self.iree_build_dir.resolve()}"
TURBINE_DIR="{self.turbine_dir.resolve()}"
TEST_FILE="{test_file.resolve()}"
CSV_FILE="{csv_file.resolve()}"

# Setup IREE environment
export PATH="$IREE_BUILD_DIR/tools:$PATH"
export PYTHONPATH="$IREE_BUILD_DIR/compiler/bindings/python:$PYTHONPATH"

# Set GPU
export CUDA_VISIBLE_DEVICES={self.gpu_id}

# Activate turbine venv and setup Python environment
cd "$TURBINE_DIR"
source .venv/bin/activate

# Source .env if it exists and export PYTHONPATH
if [ -f .env ]; then
    source .env
    export PYTHONPATH
fi

# Run the benchmark
python3 iree/turbine/kernel/boo/driver/driver.py \\
    --commands-file="$TEST_FILE" \\
    --csv="$CSV_FILE"
"""
        
        try:
            # Write temporary script
            script_path = self.results_dir / f"_run_benchmark_{limit_value}.sh"
            script_path.write_text(setup_script)
            script_path.chmod(0o755)
            
            # Execute the script
            result = subprocess.run(
                ["/bin/bash", str(script_path)],
                capture_output=True,
                text=True,
                timeout=3600
            )
            
            # Clean up script
            script_path.unlink()
            
            # Check if CSV file was created and has valid data
            # Note: ROCTracer warnings may cause non-zero exit codes, but benchmarks still succeed
            if csv_file.exists():
                try:
                    # Verify CSV has header + at least one data line
                    lines = csv_file.read_text().strip().split('\n')
                    if len(lines) >= 2:  # header + at least 1 result
                        print(f"  ✓ Benchmark complete ({len(lines)-1} tests)")
                        if result.returncode != 0:
                            print(f"  ⚠️  Warning: Non-zero exit code ({result.returncode}), but CSV is valid")
                        return csv_file
                except Exception as e:
                    print(f"  ✗ CSV file exists but is invalid: {e}")
                    return None
            
            print(f"  ✗ Benchmark failed: {result.stderr[:200]}")
            return None
        except subprocess.TimeoutExpired:
            print("Benchmark timed out")
            return None
    
    def parse_csv_results(self, csv_file: Path) -> List[Dict]:
        """Parse CSV results, skipping N.A. entries"""
        results = []
        skipped = 0
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                mean_val = row['iree_boo_experimental mean']
                if mean_val == 'N.A.' or mean_val == '':
                    skipped += 1
                    continue
                try:
                    results.append({
                        'arguments': row['arguments'],
                        'mean': float(mean_val)
                    })
                except ValueError:
                    skipped += 1
                    continue
        if skipped > 0:
            print(f"  ℹ Skipped {skipped} tests with N.A. results")
        return results

def main():
    parser = argparse.ArgumentParser(
        description='Test a single limitParallelLoops value'
    )
    parser.add_argument('--cpp-file', required=True, help='Path to SetSplitReductionSizes.cpp')
    parser.add_argument('--iree-build-dir', required=True, help='Path to IREE build directory')
    parser.add_argument('--turbine-dir', required=True, help='Path to iree-turbine directory')
    parser.add_argument('--results-dir', required=True, help='Directory to save results')
    parser.add_argument('--test-files', nargs='+', required=True, help='Test files to benchmark')
    parser.add_argument('--gpu-id', type=int, default=5, help='GPU ID to use')
    parser.add_argument('--limit', type=int, help='Single limitParallelLoops value to test (omit for baseline)')
    parser.add_argument('--baseline', action='store_true',
                       help='Run baseline test without modifying C++ code')
    parser.add_argument('--function', choices=['conv', 'matmul', 'both'], default='both',
                       help='Which function to optimize')
    parser.add_argument('--force-rerun', action='store_true',
                       help='Force re-run even if results exist')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.baseline and args.limit is None:
        print("Error: Must specify either --limit <value> or --baseline")
        return 1
    
    if args.baseline and args.limit is not None:
        print("Error: Cannot specify both --baseline and --limit")
        return 1
    
    limit_name = "baseline" if args.baseline else str(args.limit)
    
    print(f"\n{'='*80}")
    if args.baseline:
        print(f"TESTING BASELINE (original C++ code)")
    else:
        print(f"TESTING SINGLE LIMIT: {args.limit}")
    print(f"{'='*80}\n")
    
    # Create optimizer
    optimizer = LimitParallelLoopsOptimizer(
        args.cpp_file,
        args.iree_build_dir,
        args.turbine_dir,
        args.results_dir
    )
    optimizer.gpu_id = args.gpu_id
    
    try:
        # Test files
        test_files = [Path(f) for f in args.test_files]
        
        # Check if all results already exist
        limit_check = "baseline" if args.baseline else args.limit
        if optimizer.check_results_exist(test_files, limit_check) and not args.force_rerun:
            print(f"✓ Results already exist for {limit_name}")
            for test_file in test_files:
                csv_file = optimizer.results_dir / f"limit_{limit_check}_{test_file.stem}.csv"
                print(f"  Found: {csv_file.name}")
            print(f"\n⏭️  Skipping: C++ modification, IREE build, and benchmarks")
            print(f"   Use --force-rerun to regenerate\n")
            return 0
        
        # If force-rerun, delete existing results
        if args.force_rerun and optimizer.check_results_exist(test_files, limit_check):
            print(f"♻️  Force rerun enabled - regenerating results for {limit_name}")
            for test_file in test_files:
                csv_file = optimizer.results_dir / f"limit_{limit_check}_{test_file.stem}.csv"
                if csv_file.exists():
                    csv_file.unlink()
                    print(f"  Deleted: {csv_file.name}")
            print()
        
        # Restore original C++ first
        optimizer.restore_original()
        
        if not args.baseline:
            # Modify C++ file with fixed limit
            print(f"Modifying C++ to use limitParallelLoops = {args.limit}")
            optimizer.set_fixed_limitParallelLoops(args.limit, args.function)
        else:
            print("Using original C++ code (no modifications)")
        
        # Build IREE
        print("Building IREE...")
        if not optimizer.build_iree():
            print(f"❌ Build failed for {limit_name}")
            optimizer.restore_original()
            return 1
        
        # Run benchmarks
        print("Run benchmarks...")
        limit_value = "baseline" if args.baseline else args.limit
        for test_file in test_files:
            csv_file = optimizer.run_benchmark(test_file, limit_value, args.force_rerun)
            if csv_file:
                results = optimizer.parse_csv_results(csv_file)
                print(f"✓ Completed {len(results)} test cases from {test_file.name}")
            else:
                print(f"✗ Benchmark failed for {test_file.name}")
                optimizer.restore_original()
                return 1
        
        # Restore original
        optimizer.restore_original()
        
        print(f"\n✅ Successfully tested {limit_name}")
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        optimizer.restore_original()
        return 1

if __name__ == '__main__':
    sys.exit(main())
