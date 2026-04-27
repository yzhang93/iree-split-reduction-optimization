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
        self.gpu_id = 0
        
        # Create backup if doesn't exist
        if not self.backup_file.exists():
            shutil.copy2(self.cpp_file, self.backup_file)
        
        # Create results directory
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def set_fixed_limitParallelLoops(self, limit_value: int, mode: str = 'both'):
        """Modify C++ file to use a fixed limitParallelLoops value.

        Also wraps the two "heuristic" early-return `if` blocks in each of
        `getConvolutionReductionSizes` and `getMatmulLikeReductionSizes` in
        `#if 0 / #endif` so that no op is skipped by the stale heuristics
        during the limitParallelLoops sweep. The other early returns (dynamic
        shape checks, failed inference, etc.) are left intact.

        Pattern matching is tied to the current shape of
        SetSplitReductionSizes.cpp. If upstream changes the heuristic
        conditions, update the substrings in HEURISTIC_EARLY_RETURNS and
        CONSTANTS_END_MARKER below.
        """
        # Heuristic early-return `if` statements we want to disable.
        # Key: function tag ('conv' or 'matmul'); Value: list of unique
        # substrings that identify the start of each heuristic-early-return.
        HEURISTIC_EARLY_RETURNS = {
            'conv': [
                'if (parallelSize >= largeParallelSize',
                'if (ratio <= ratioThreshold',  # compound w/ reductionSize
            ],
            'matmul': [
                'if (outputSize > largeOutputSize',
                'if (kSize < largeKSize',       # compound w/ ratio
            ],
        }
        # The LAST constant declaration in the heuristic-constants block of
        # each function -- used as the anchor for (void) suppressions so the
        # constants don't become "unused variable" build errors once the
        # early returns are disabled.
        CONSTANTS_END_MARKER = {
            'conv': 'const int64_t ratioThreshold = 64;',
            'matmul': 'const int64_t ratioThreshold = 48;',
        }
        CONSTANTS_TO_SUPPRESS = {
            'conv': ['largeParallelSize', 'largeReductionSize', 'ratioThreshold'],
            'matmul': ['largeOutputSize', 'largeKSize', 'ratioThreshold'],
        }
        # Where the `ratio` (and any companion) variables get declared.
        # After disabling the early returns these become unused.
        RATIO_DECL_MARKER = {
            'conv': 'int64_t ratio = reductionSize / std::sqrt(parallelSize);',
            'matmul': 'int64_t ratio = kSize / std::sqrt(mSize * nSize) / batchSize;',
        }
        RATIO_COMPANION_SUPPRESS = {
            'conv': ['reductionSize', 'ratio'],
            'matmul': ['ratio'],
        }
        # Extra (void) casts for locals that become unused once the
        # original `limitParallelLoops` if-else chain is `#if 0`'d out.
        EXTRA_LIMIT_SUPPRESS = {
            'conv': ['outputSize', 'startTileSize'],
            'matmul': ['outputSize', 'kSize'],
        }
        # Which function tag corresponds to which --function mode.
        MODE_ENABLES = {
            'conv':   {'conv'},
            'matmul': {'matmul'},
            'both':   {'conv', 'matmul'},
            'mixed':  {'conv', 'matmul'},
        }
        enabled = MODE_ENABLES.get(mode, {'conv', 'matmul'})

        with open(self.cpp_file, 'r') as f:
            lines = f.readlines()

        modified_lines = []
        current_func = None          # 'conv', 'matmul', or None
        function_brace_depth = 0
        function_started = False     # True once we see the first { after function declaration
        skip_until_closing_brace = False
        skip_brace_depth = 0
        in_early_return_block = False
        constants_declared = {'conv': False, 'matmul': False}

        i = 0
        while i < len(lines):
            line = lines[i]

            # Track which function we're in based on line content.
            if 'getConvolutionReductionSizes' in line and '(' in line:
                current_func = 'conv'
                function_brace_depth = 0
                function_started = False
            elif 'getMatmulLikeReductionSizes' in line and '(' in line:
                current_func = 'matmul'
                function_brace_depth = 0
                function_started = False

            # Track braces to know when we exit the function.
            if current_func is not None and not skip_until_closing_brace:
                if '{' in line:
                    function_started = True
                if function_started:
                    function_brace_depth += line.count('{') - line.count('}')
                    if function_brace_depth <= 0:
                        current_func = None
                        function_brace_depth = 0
                        function_started = False

            # Handle skipping original limitParallelLoops if-else chain.
            if skip_until_closing_brace:
                skip_brace_depth += line.count('{') - line.count('}')
                modified_lines.append(line)
                if skip_brace_depth <= 0:
                    modified_lines.append('#endif  // OPTIMIZER\n')
                    skip_until_closing_brace = False
                    skip_brace_depth = 0
                i += 1
                continue

            # Disable heuristic early returns inside enabled functions.
            if (current_func in enabled
                    and function_started
                    and not in_early_return_block
                    and any(pat in line for pat in HEURISTIC_EARLY_RETURNS[current_func])):
                modified_lines.append(
                    '#if 0  // OPTIMIZER: Disabled early return to test all limits\n')
                in_early_return_block = True

            if in_early_return_block:
                modified_lines.append(line)
                if line.strip() == '}':
                    modified_lines.append('#endif  // OPTIMIZER\n')
                    in_early_return_block = False
                i += 1
                continue

            # (void)-suppress the heuristic constants once we hit the last one
            # so they don't trigger unused-variable warnings.
            if (current_func in enabled
                    and not constants_declared[current_func]
                    and CONSTANTS_END_MARKER[current_func] in line):
                modified_lines.append(line)
                modified_lines.append('\n')
                modified_lines.append('  // OPTIMIZER: Suppress unused variable warnings\n')
                for name in CONSTANTS_TO_SUPPRESS[current_func]:
                    modified_lines.append(f'  (void){name};\n')
                constants_declared[current_func] = True
                i += 1
                continue

            # (void)-suppress ratio (and companions) once declared, since the
            # heuristic early returns that used them are now disabled.
            if current_func in enabled and RATIO_DECL_MARKER[current_func] in line:
                modified_lines.append(line)
                for name in RATIO_COMPANION_SUPPRESS[current_func]:
                    modified_lines.append(
                        f'  (void){name};  // OPTIMIZER: Suppress unused warning\n')
                i += 1
                continue

            # Replace the `limitParallelLoops` if-else cascade with a fixed value.
            if (current_func in enabled
                    and function_started
                    and 'int64_t limitParallelLoops;' in line):
                modified_lines.append(line)
                modified_lines.append('  // OPTIMIZER: Fixed value for testing\n')
                modified_lines.append(f'  limitParallelLoops = {limit_value};\n')
                for name in EXTRA_LIMIT_SUPPRESS[current_func]:
                    modified_lines.append(
                        f'  (void){name};  // Suppress unused warning\n')
                modified_lines.append('#if 0  // Original code disabled\n')
                skip_until_closing_brace = True
                skip_brace_depth = 0
                i += 1
                continue

            modified_lines.append(line)
            i += 1

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
        """Run benchmark across multiple GPUs and return path to merged CSV results"""
        test_basename = test_file.stem
        csv_file = self.results_dir / f"limit_{limit_value}_{test_basename}.csv"
        
        # Skip if results already exist (unless force_rerun is True)
        if csv_file.exists() and not force_rerun:
            return csv_file
        
        gpu_ids = self.gpu_ids if hasattr(self, 'gpu_ids') else [self.gpu_id]
        num_gpus = len(gpu_ids)
        
        # Read all test lines
        with open(test_file) as f:
            all_lines = [l for l in f.read().strip().split('\n') if l.strip()]
        total = len(all_lines)
        
        if num_gpus <= 1 or total <= 1:
            # Single GPU path
            return self._run_benchmark_single_gpu(test_file, csv_file, limit_value, gpu_ids[0])
        
        print(f"  Running benchmarks for {test_file.name} ({total} shapes across {num_gpus} GPUs: {gpu_ids}) ...")
        
        import shutil as _shutil
        import time
        
        # Clear shared BOO cache to ensure fresh compilations
        boo_cache = Path.home() / ".cache" / "turbine_kernels" / "boo"
        if boo_cache.exists():
            _shutil.rmtree(boo_cache, ignore_errors=True)
        
        # Split shapes into per-GPU chunk files
        chunk_size = total // num_gpus
        remainder = total % num_gpus
        chunk_files = []
        part_csvs = []
        offset = 0
        for i in range(num_gpus):
            sz = chunk_size + (1 if i < remainder else 0)
            chunk = all_lines[offset:offset+sz]
            offset += sz
            if not chunk:
                continue
            chunk_file = self.results_dir / f"_chunk_{limit_value}_{i}.txt"
            chunk_file.write_text('\n'.join(chunk) + '\n')
            chunk_files.append((i, chunk_file))
            part_csvs.append(self.results_dir / f"_part_{limit_value}_{i}.csv")
        
        # Write a single bash script that sets up env, then launches
        # parallel driver.py processes (exactly like run_benchmark.sh)
        turbine_dir = self.turbine_dir.resolve()
        iree_build_dir = self.iree_build_dir.resolve()
        
        launch_lines = []
        launch_lines.append("PIDS=()")
        for idx, ((i, chunk_file), part_csv) in enumerate(zip(chunk_files, part_csvs)):
            gpu_id = gpu_ids[i]
            launch_lines.append(
                f'CUDA_VISIBLE_DEVICES="{gpu_id}" '
                f'python "{turbine_dir}/iree/turbine/kernel/boo/driver/driver.py" '
                f'--commands-file "{chunk_file.resolve()}" '
                f'--csv "{part_csv.resolve()}" > /dev/null 2>&1 &'
            )
            launch_lines.append("PIDS+=($!)")
        launch_lines.append("wait ${PIDS[@]}")
        
        parallel_script = f"""#!/bin/bash
set -e
export PATH="{iree_build_dir}/tools:$PATH"
cd "{turbine_dir}"
source .venv/bin/activate
if [ -f .env ]; then source .env; export PYTHONPATH; fi

{chr(10).join(launch_lines)}
"""
        script_path = self.results_dir / f"_run_parallel_{limit_value}.sh"
        script_path.write_text(parallel_script)
        script_path.chmod(0o755)
        
        # Launch the script and monitor progress
        proc = subprocess.Popen(
            ["/bin/bash", str(script_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        all_part_csvs = [pc for pc in part_csvs]
        while proc.poll() is None:
            completed = 0
            for pc in all_part_csvs:
                if pc.exists():
                    try:
                        lines = pc.read_text().strip().split('\n')
                        completed += max(0, len(lines) - 1)
                    except:
                        pass
            print(f"\r  Running benchmark {completed}/{total} ...", end='', flush=True)
            time.sleep(2)
        
        proc.wait()
        script_path.unlink(missing_ok=True)
        
        # Merge per-GPU CSVs (keep header from first, strip from rest)
        first = True
        with open(csv_file, 'w') as out:
            for pc in all_part_csvs:
                if pc.exists():
                    lines = pc.read_text().strip().split('\n')
                    if first:
                        out.write('\n'.join(lines) + '\n')
                        first = False
                    elif len(lines) > 1:
                        out.write('\n'.join(lines[1:]) + '\n')
        
        # Clean up
        for _, cf in chunk_files:
            cf.unlink(missing_ok=True)
        for pc in all_part_csvs:
            pc.unlink(missing_ok=True)
        
        if csv_file.exists():
            lines = csv_file.read_text().strip().split('\n')
            if len(lines) >= 2:
                print(f"\r  ✓ Benchmark complete ({len(lines)-1} tests across {num_gpus} GPUs)")
                return csv_file
        
        print(f"\n  ✗ Benchmark failed")
        return None
    
    def _run_benchmark_single_gpu(self, test_file: Path, csv_file: Path, limit_value: int, gpu_id: int) -> Optional[Path]:
        """Run benchmark on a single GPU"""
        print(f"  Running benchmarks for {test_file.name}...")
        
        setup_script = f"""#!/bin/bash
set -e
IREE_BUILD_DIR="{self.iree_build_dir.resolve()}"
TURBINE_DIR="{self.turbine_dir.resolve()}"
TEST_FILE="{test_file.resolve()}"
CSV_FILE="{csv_file.resolve()}"
export PATH="$IREE_BUILD_DIR/tools:$PATH"
export PYTHONPATH="$IREE_BUILD_DIR/compiler/bindings/python:$PYTHONPATH"
export HIP_VISIBLE_DEVICES={gpu_id}
export CUDA_VISIBLE_DEVICES={gpu_id}
cd "$TURBINE_DIR"
source .venv/bin/activate
if [ -f .env ]; then
    source .env
    export PYTHONPATH
fi
python3 iree/turbine/kernel/boo/driver/driver.py \\
    --commands-file="$TEST_FILE" \\
    --csv="$CSV_FILE"
"""
        
        try:
            script_path = self.results_dir / f"_run_benchmark_{limit_value}.sh"
            script_path.write_text(setup_script)
            script_path.chmod(0o755)
            
            result = subprocess.run(
                ["/bin/bash", str(script_path)],
                capture_output=True,
                text=True,
                timeout=3600
            )
            
            script_path.unlink()
            
            if csv_file.exists():
                try:
                    lines = csv_file.read_text().strip().split('\n')
                    if len(lines) >= 2:
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
    parser.add_argument('--gpu-id', type=int, default=5, help='GPU ID to use (single GPU)')
    parser.add_argument('--gpu-ids', type=str, default=None,
                       help='Comma-separated GPU IDs for multi-GPU parallelism (e.g. "2,3,4,5,6,7")')
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
    if args.gpu_ids:
        optimizer.gpu_ids = [int(x) for x in args.gpu_ids.split(',')]
    else:
        optimizer.gpu_ids = [args.gpu_id]
    
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
