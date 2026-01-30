#!/usr/bin/env python3
"""
Analyze which factors best predict optimal limitParallelLoops.

This script examines the relationship between:
- outputSize (parallel dimensions)
- kSize (reduction dimension)
- ratio (kSize / sqrt(outputSize))
- Individual dimensions (outputChannelSize, batchSize, etc.)

And determines the best decision criteria for choosing limitParallelLoops.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import argparse


def analyze_decision_factors(results_file: Path):
    """Analyze which factors best separate different limitParallelLoops clusters"""
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    # Extract all test cases with their characteristics
    test_data = []
    for test_name, analysis in data['analyses'].items():
        best_limit = analysis.get('best_limit')
        if isinstance(best_limit, str):  # Skip baseline
            continue
        
        # Get dimensions
        dims = analysis.get('dimensions', {})
        m = dims.get('m', 0)
        n = dims.get('n', 0)
        k = dims.get('k', 0)
        
        if not (m and n and k):
            continue
        
        output_size = m * n
        k_size = k
        ratio = k_size / (output_size ** 0.5) if output_size > 0 else 0
        
        test_data.append({
            'test_name': test_name,
            'best_limit': best_limit,
            'output_size': output_size,
            'k_size': k_size,
            'ratio': ratio,
            'm': m,
            'n': n,
            'k': k
        })
    
    if not test_data:
        print("No test data found")
        return
    
    # Group by best_limit
    limits = {}
    for test in test_data:
        limit = test['best_limit']
        if limit not in limits:
            limits[limit] = []
        limits[limit].append(test)
    
    print("="*100)
    print("DECISION FACTOR ANALYSIS")
    print("="*100)
    print()
    
    # Analyze each limit's characteristics
    for limit in sorted(limits.keys()):
        tests = limits[limit]
        output_sizes = [t['output_size'] for t in tests]
        k_sizes = [t['k_size'] for t in tests]
        ratios = [t['ratio'] for t in tests]
        
        print(f"limitParallelLoops = {limit}:")
        print(f"  Count: {len(tests)}")
        print(f"  Output size range: {min(output_sizes):,} - {max(output_sizes):,}")
        print(f"  K size range: {min(k_sizes):,} - {max(k_sizes):,}")
        print(f"  Ratio range: {min(ratios):.2f} - {max(ratios):.2f}")
        print()
    
    # Check for overlaps
    print("="*100)
    print("OVERLAP ANALYSIS")
    print("="*100)
    print()
    
    sorted_limits = sorted(limits.keys())
    for i, limit1 in enumerate(sorted_limits):
        for limit2 in sorted_limits[i+1:]:
            tests1 = limits[limit1]
            tests2 = limits[limit2]
            
            # Check outputSize overlap
            out1 = [t['output_size'] for t in tests1]
            out2 = [t['output_size'] for t in tests2]
            out_overlap = min(max(out1), max(out2)) - max(min(out1), min(out2))
            
            # Check kSize overlap
            k1 = [t['k_size'] for t in tests1]
            k2 = [t['k_size'] for t in tests2]
            k_overlap = min(max(k1), max(k2)) - max(min(k1), min(k2))
            
            # Check ratio overlap
            r1 = [t['ratio'] for t in tests1]
            r2 = [t['ratio'] for t in tests2]
            r_overlap = min(max(r1), max(r2)) - max(min(r1), min(r2))
            
            if out_overlap > 0:
                print(f"limit={limit1} vs limit={limit2}:")
                print(f"  OutputSize overlap: {out_overlap > 0} (range: {out_overlap:,})")
                print(f"  KSize overlap: {k_overlap > 0} (range: {k_overlap:,})")
                print(f"  Ratio overlap: {r_overlap > 0} (range: {r_overlap:.2f})")
                print()
    
    # Suggest 2D decision boundaries
    print("="*100)
    print("SUGGESTED 2D DECISION CRITERIA")
    print("="*100)
    print()
    
    print("Option 1: outputSize + kSize thresholds")
    print("-" * 100)
    for limit in sorted(limits.keys()):
        tests = limits[limit]
        output_sizes = [t['output_size'] for t in tests]
        k_sizes = [t['k_size'] for t in tests]
        
        med_out = sorted(output_sizes)[len(output_sizes)//2]
        med_k = sorted(k_sizes)[len(k_sizes)//2]
        
        print(f"limit={limit}: median outputSize={med_out:,}, median kSize={med_k:,}")
    print()
    
    print("Option 2: outputSize + ratio thresholds")
    print("-" * 100)
    for limit in sorted(limits.keys()):
        tests = limits[limit]
        output_sizes = [t['output_size'] for t in tests]
        ratios = [t['ratio'] for t in tests]
        
        med_out = sorted(output_sizes)[len(output_sizes)//2]
        med_ratio = sorted(ratios)[len(ratios)//2]
        
        print(f"limit={limit}: median outputSize={med_out:,}, median ratio={med_ratio:.2f}")
    print()
    
    # Analyze which factor separates best
    print("="*100)
    print("SEPARATION QUALITY")
    print("="*100)
    print()
    
    def separation_score(factor_name: str):
        """Calculate how well a factor separates different limits"""
        # For each pair of limits, check if their ranges don't overlap
        separations = 0
        total_pairs = 0
        
        for i, limit1 in enumerate(sorted_limits):
            for limit2 in sorted_limits[i+1:]:
                total_pairs += 1
                
                if factor_name == 'output_size':
                    vals1 = [t['output_size'] for t in limits[limit1]]
                    vals2 = [t['output_size'] for t in limits[limit2]]
                elif factor_name == 'k_size':
                    vals1 = [t['k_size'] for t in limits[limit1]]
                    vals2 = [t['k_size'] for t in limits[limit2]]
                elif factor_name == 'ratio':
                    vals1 = [t['ratio'] for t in limits[limit1]]
                    vals2 = [t['ratio'] for t in limits[limit2]]
                
                # Check if ranges don't overlap
                if max(vals1) < min(vals2) or max(vals2) < min(vals1):
                    separations += 1
        
        return separations, total_pairs
    
    for factor in ['output_size', 'k_size', 'ratio']:
        seps, total = separation_score(factor)
        print(f"{factor}: {seps}/{total} pairs cleanly separated ({seps/total*100:.1f}%)")
    print()
    
    print("RECOMMENDATION:")
    print("-" * 100)
    print("Since outputSize alone has significant overlaps, consider:")
    print("1. Use 2D decision tree: (outputSize, kSize) or (outputSize, ratio)")
    print("2. Use decision tree algorithm to find optimal split points")
    print("3. Priority: kSize and ratio seem to have better separation")
    print()


def main():
    parser = argparse.ArgumentParser(description='Analyze decision factors')
    parser.add_argument('--results-file', required=True, type=Path,
                       help='Path to limitParallelLoops_sweep_results.json')
    
    args = parser.parse_args()
    
    if not args.results_file.exists():
        print(f"Error: Results file not found: {args.results_file}")
        return 1
    
    analyze_decision_factors(args.results_file)
    return 0


if __name__ == '__main__':
    sys.exit(main())
