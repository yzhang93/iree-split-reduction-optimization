#!/usr/bin/env python3
"""
Create JSON summary from CSV results

This script reads all limit_*.csv files and creates the JSON file
that analyze_limitParallelLoops.py expects.
"""

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

def parse_csv(csv_file: Path) -> List[Dict]:
    """Parse a CSV file and return test results"""
    results = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                results.append({
                    'arguments': row['arguments'],
                    'mean': float(row['iree_boo_experimental mean'])
                })
            except (KeyError, ValueError) as e:
                print(f"Warning: Skipping row in {csv_file.name}: {e}", file=sys.stderr)
    return results

def create_json_from_csvs(results_dir: Path, output_file: Path):
    """Create JSON summary from all CSV files"""
    
    # Find all CSV files
    csv_files = sorted(results_dir.glob('limit_*_*.csv'))
    
    if not csv_files:
        print(f"Error: No CSV files found in {results_dir}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Found {len(csv_files)} CSV files")
    
    # Extract limit values
    limits = set()
    for csv_file in csv_files:
        # Extract limit from filename: limit_64_small_test.csv -> 64
        # or limit_baseline_small_test.csv -> "baseline"
        parts = csv_file.stem.split('_')
        if len(parts) >= 2:
            limit_str = parts[1]
            if limit_str == "baseline":
                limits.add("baseline")
            else:
                try:
                    limit = int(limit_str)
                    limits.add(limit)
                except ValueError:
                    pass
    
    # Sort with baseline first
    limits_list = []
    if "baseline" in limits:
        limits_list.append("baseline")
        limits.remove("baseline")
    limits_list.extend(sorted(limits))
    
    print(f"Limits found: {limits_list}")
    
    # Build test_name -> {limit -> runtime} mapping
    test_runtimes = {}  # test_name -> {limit: runtime}
    
    for csv_file in csv_files:
        # Extract limit from filename
        parts = csv_file.stem.split('_')
        if len(parts) < 2:
            continue
        
        limit_str = parts[1]
        if limit_str == "baseline":
            limit = "baseline"
        else:
            try:
                limit = int(limit_str)
            except ValueError:
                continue
        
        # Parse CSV
        test_cases = parse_csv(csv_file)
        
        if not test_cases:
            print(f"Warning: No test cases in {csv_file.name}", file=sys.stderr)
            continue
        
        # Add runtimes for each test case
        for tc in test_cases:
            test_name = tc['arguments']
            if test_name not in test_runtimes:
                test_runtimes[test_name] = {}
            test_runtimes[test_name][limit] = tc['mean']
        
        print(f"  Loaded {len(test_cases)} test cases for limit={limit}")
    
    # Build analyses structure: test_name -> analysis
    analyses = {}
    
    for test_name, runtimes in test_runtimes.items():
        # Find best limit (minimum runtime), excluding baseline from "best"
        non_baseline_runtimes = {k: v for k, v in runtimes.items() if k != "baseline"}
        if non_baseline_runtimes:
            best_limit = min(non_baseline_runtimes.keys(), key=lambda l: non_baseline_runtimes[l])
            best_runtime = non_baseline_runtimes[best_limit]
        else:
            # Fallback if only baseline exists
            best_limit = min(runtimes.keys(), key=lambda l: runtimes[l])
            best_runtime = runtimes[best_limit]
        
        worst_runtime = max(runtimes.values())
        speedup = worst_runtime / best_runtime if best_runtime > 0 else 1.0
        
        analyses[test_name] = {
            'csv_data': {
                'arguments': test_name,
                'runtimes': runtimes
            },
            'best_limit': best_limit,
            'best_runtime': best_runtime,
            'worst_runtime': worst_runtime,
            'speedup_vs_worst': speedup,
            'all_runtimes': runtimes
        }
    
    # Create final JSON structure
    json_data = {
        'candidate_limits': limits_list,
        'analyses': analyses
    }
    
    # Write JSON
    with open(output_file, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"\n✓ Created {output_file}")
    print(f"  Candidate limits: {limits}")
    print(f"  Total test cases: {len(analyses)}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Create JSON summary from CSV results')
    parser.add_argument('--results-dir', type=Path, required=True,
                       help='Directory containing limit_*.csv files')
    parser.add_argument('--output', type=Path,
                       help='Output JSON file (default: results_dir/limitParallelLoops_sweep_results.json)')
    
    args = parser.parse_args()
    
    if not args.output:
        args.output = args.results_dir / 'limitParallelLoops_sweep_results.json'
    
    create_json_from_csvs(args.results_dir, args.output)
