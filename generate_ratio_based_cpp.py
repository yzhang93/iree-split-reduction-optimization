#!/usr/bin/env python3
"""
Generate ratio-based C++ code for limitParallelLoops.

This script analyzes sweep results and generates a UNIFIED decision tree
based on ratio = kSize / sqrt(outputSize), which naturally combines both
parallel and reduction dimensions.

Benefits over outputSize-only approach:
- Better separation of clusters (less overlap)
- Single decision mechanism (no separate early return)
- limit=1 explicitly means "no split"
- Simpler, more maintainable code
"""

import json
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass
import math


@dataclass
class TestData:
    """Test case with computed ratio"""
    test_name: str
    best_limit: int
    best_runtime: float
    output_size: int
    k_size: int
    ratio: float  # kSize / sqrt(outputSize)
    test_type: str  # 'conv' or 'gemm'


def parse_args_for_dimensions(args_str: str) -> Tuple[int, int, int, str]:
    """Extract m, n, k from argument string.
    
    Returns (output_size, n, k_size, test_type) where test_type is 'conv' or 'gemm'
    """
    import re
    import ast
    
    # ==================== ATen GEMM Format ====================
    # aten::mm "[[M, K], [K, N]]" ...
    # aten::addmm "[[bias], [M, K], [K, N], ...]" ...
    
    if args_str.startswith('aten::mm') or args_str.startswith("aten::mm"):
        # Extract the matrix dimensions
        try:
            # Find the first quoted bracket section
            match = re.search(r'\[\[(\d+),\s*(\d+)\],\s*\[(\d+),\s*(\d+)\]\]', args_str)
            if match:
                m = int(match.group(1))
                k1 = int(match.group(2))
                k2 = int(match.group(3))
                n = int(match.group(4))
                # k1 and k2 should match (inner dimension)
                k = k1  
                output_size = m * n
                return output_size, n, k, 'gemm'
        except:
            pass
    
    if args_str.startswith('aten::addmm') or args_str.startswith("aten::addmm"):
        # addmm format: "[[bias], [M, K], [K, N], ...]"
        try:
            # Find all bracket pairs with dimensions
            matches = re.findall(r'\[(\d+),\s*(\d+)\]', args_str)
            if len(matches) >= 2:
                # First pair is [M, K], second is [K, N]
                m = int(matches[0][0])
                k1 = int(matches[0][1])
                k2 = int(matches[1][0])
                n = int(matches[1][1])
                k = k1
                output_size = m * n
                return output_size, n, k, 'gemm'
        except:
            pass
    
    # ==================== Weight Backward Convolution Pattern ====================
    wb_match = re.search(r'-n (\d+).*-c (\d+).*-k (\d+).*-y (\d+).*-x (\d+)', args_str)
    if wb_match and '-y 1 -x 1' not in args_str:
        # Weight backward: output = k * c * y * x, reduction = n * H * W
        n, c, k, y, x = [int(wb_match.group(i)) for i in range(1, 6)]
        
        # Get spatial dimensions
        h_match = re.search(r'-H (\d+)', args_str)
        w_match = re.search(r'-W (\d+)', args_str)
        H = int(h_match.group(1)) if h_match else 1
        W = int(w_match.group(1)) if w_match else 1
        
        # Get depth if 3D
        d_match = re.search(r'--in_d (\d+)', args_str)
        D = int(d_match.group(1)) if d_match else 1
        
        filter_d = 1
        fd_match = re.search(r'--fil_d (\d+)', args_str)
        if fd_match:
            filter_d = int(fd_match.group(1))
        
        output_size = k * c * y * x * filter_d
        k_size = n * H * W * D
        
        return output_size, 1, k_size, 'conv'
    
    # ==================== 1x1 Convolution (Matmul-like) Pattern ====================
    mm_match = re.search(r'-n (\d+).*-c (\d+).*-k (\d+).*-y 1 -x 1', args_str)
    if mm_match:
        n, c, k = [int(mm_match.group(i)) for i in range(1, 4)]
        # Matmul: m=n*c, n=k, k=c (batch dimension)
        return n * c, k, c, 'conv'
    
    return 0, 0, 0, 'unknown'


def load_captured_dimensions(results_dir: Path) -> Dict[str, Dict]:
    """Load captured dimensions JSON and build lookup by test_config."""
    dims_file = results_dir / "captured_dimensions.json"
    if not dims_file.exists():
        return {}
    
    with open(dims_file, 'r') as f:
        dims_list = json.load(f)
    
    # Build lookup by test_config
    lookup = {}
    for entry in dims_list:
        test_config = entry.get('test_config')
        if test_config:
            lookup[test_config] = entry
    
    return lookup


def load_and_analyze(results_file: Path, captured_dims: Dict[str, Dict] = None) -> Tuple[List[TestData], str]:
    """Load results and compute ratio for each test.
    
    Args:
        results_file: Path to sweep results JSON
        captured_dims: Optional dict mapping test_config -> captured dimensions
    
    Returns (test_data, test_type) where test_type is 'conv', 'gemm', or 'mixed'
    """
    if captured_dims is None:
        captured_dims = {}
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    test_data = []
    test_types = set()
    
    matched_from_captured = 0
    parsed_from_name = 0
    
    for test_name, analysis in data['analyses'].items():
        best_limit = analysis.get('best_limit')
        
        # Skip baseline
        if isinstance(best_limit, str):
            continue
        
        # Try to get dimensions from captured_dimensions.json first
        output_size = 0
        k_size = 0
        test_type = 'unknown'
        
        if test_name in captured_dims:
            entry = captured_dims[test_name]
            if entry.get('type') == 'matmul':
                output_size = entry.get('outputSize', 0)
                k_size = entry.get('kSize', 0)
                test_type = 'gemm'
                matched_from_captured += 1
            elif entry.get('type') == 'conv':
                output_size = entry.get('outputSize', 0)
                k_size = entry.get('reductionSize', 0)
                test_type = 'conv'
                if k_size:  # Only count if reductionSize exists
                    matched_from_captured += 1
        
        # Fall back to parsing from test name if no captured dimensions
        if not (output_size and k_size):
            output_size, n, k_size, test_type = parse_args_for_dimensions(test_name)
            if output_size and k_size:
                parsed_from_name += 1
        
        if not (output_size and k_size):
            print(f"Warning: Could not get dimensions for: {test_name[:60]}...", file=sys.stderr)
            continue
        
        test_types.add(test_type)
        
        # Compute ratio = kSize / sqrt(outputSize)
        ratio = k_size / math.sqrt(output_size) if output_size > 0 else 0
        
        test_data.append(TestData(
            test_name=test_name,
            best_limit=best_limit,
            best_runtime=analysis['best_runtime'],
            output_size=output_size,
            k_size=k_size,
            ratio=ratio,
            test_type=test_type
        ))
    
    # Print matching statistics
    total = matched_from_captured + parsed_from_name
    if total > 0:
        print(f"  Dimension sources: {matched_from_captured} from captured_dimensions.json, {parsed_from_name} parsed from test names", file=sys.stderr)
    
    # Determine overall test type
    if len(test_types) == 1:
        overall_type = test_types.pop()
    elif 'gemm' in test_types and 'conv' in test_types:
        overall_type = 'mixed'
    else:
        overall_type = 'unknown'
    
    return test_data, overall_type


def cluster_by_ratio(test_data: List[TestData]) -> Dict[int, List[TestData]]:
    """Group tests by their optimal limitParallelLoops"""
    clusters = {}
    for test in test_data:
        limit = test.best_limit
        if limit not in clusters:
            clusters[limit] = []
        clusters[limit].append(test)
    return clusters


def filter_outliers_iqr(ratios: List[float], multiplier: float = 1.5) -> Tuple[List[float], List[float]]:
    """Filter outliers using Interquartile Range (IQR) method.
    
    Args:
        ratios: List of ratio values
        multiplier: IQR multiplier (1.5 for mild outliers, 3.0 for extreme)
    
    Returns:
        (filtered_ratios, outliers)
    """
    if len(ratios) < 4:
        return ratios, []
    
    sorted_ratios = sorted(ratios)
    q1_idx = len(sorted_ratios) // 4
    q3_idx = (3 * len(sorted_ratios)) // 4
    
    q1 = sorted_ratios[q1_idx]
    q3 = sorted_ratios[q3_idx]
    iqr = q3 - q1
    
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    filtered = [r for r in ratios if lower_bound <= r <= upper_bound]
    outliers = [r for r in ratios if r < lower_bound or r > upper_bound]
    
    return filtered, outliers


def derive_ratio_thresholds(clusters: Dict[int, List[TestData]]) -> Tuple[List[Tuple[int, float]], Dict[int, Dict]]:
    """Derive ratio thresholds using statistical analysis with outlier removal.
    
    Strategy:
    1. Remove outliers from each cluster using IQR method
    2. Calculate robust statistics (median, percentiles) for each cluster
    3. For each pair of adjacent clusters (sorted by median), find the optimal
       threshold that maximizes classification accuracy
    
    Returns:
        (thresholds, cluster_stats) where thresholds is list of (limit, threshold)
    """
    # Step 1: Calculate robust statistics with outlier removal
    cluster_stats = {}
    all_test_ratios = []  # (ratio, optimal_limit) for accuracy calculation
    
    for limit, tests in clusters.items():
        ratios = [t.ratio for t in tests]
        all_test_ratios.extend([(t.ratio, limit) for t in tests])
        
        filtered_ratios, outliers = filter_outliers_iqr(ratios, multiplier=1.5)
        
        if len(filtered_ratios) < 2:
            # If too few remain, use original ratios with more aggressive filtering
            filtered_ratios, outliers = filter_outliers_iqr(ratios, multiplier=3.0)
        
        if len(filtered_ratios) == 0:
            filtered_ratios = ratios  # Fallback to all ratios
        
        sorted_filtered = sorted(filtered_ratios)
        n = len(sorted_filtered)
        
        cluster_stats[limit] = {
            'limit': limit,
            'count': len(tests),
            'count_filtered': n,
            'outliers_removed': len(outliers),
            'median': sorted_filtered[n // 2],
            'p25': sorted_filtered[n // 4] if n >= 4 else sorted_filtered[0],
            'p75': sorted_filtered[(3 * n) // 4] if n >= 4 else sorted_filtered[-1],
            'min': min(sorted_filtered),
            'max': max(sorted_filtered),
            'original_min': min(ratios),
            'original_max': max(ratios),
            'filtered_ratios': filtered_ratios,
        }
    
    # Step 2: Sort clusters by median ratio (descending - highest first)
    sorted_limits = sorted(cluster_stats.keys(), key=lambda l: cluster_stats[l]['median'], reverse=True)
    
    print("\n  Cluster statistics (after outlier removal):", file=sys.stderr)
    for limit in sorted_limits:
        stats = cluster_stats[limit]
        print(f"    limit={limit}: {stats['count_filtered']}/{stats['count']} tests, "
              f"median={stats['median']:.2f}, range=[{stats['min']:.2f}, {stats['max']:.2f}]"
              + (f", removed {stats['outliers_removed']} outliers" if stats['outliers_removed'] > 0 else ""),
              file=sys.stderr)
    
    # Step 3: Find optimal thresholds using accuracy optimization
    thresholds = find_optimal_thresholds(sorted_limits, cluster_stats, all_test_ratios)
    
    return thresholds, cluster_stats


def find_optimal_thresholds_2d(sorted_limits: List[int], cluster_stats: Dict[int, Dict], 
                               all_test_data: List[Dict]) -> Dict:
    """Find optimal 2D thresholds using both ratio and outputSize.
    
    Returns a dict with:
      - 'ratio_thresholds': List of (limit, ratio_threshold) for high-ratio cases
      - 'outputsize_rules': List of (limit, ratio_low, ratio_high, output_thresh) for 2D rules
      - 'default_limit': Default limit for unmatched cases
    """
    # Calculate accuracy for a 2D decision tree configuration
    def calculate_accuracy_2d(config):
        correct = 0
        for d in all_test_data:
            predicted = predict_2d(d['ratio'], d['output_size'], config)
            if predicted == d['best_limit']:
                correct += 1
        return correct
    
    def predict_2d(ratio, output_size, config):
        # Check ratio thresholds first (high ratio cases)
        for limit, thresh in config.get('ratio_thresholds', []):
            if ratio > thresh:
                return limit
        
        # Check 2D rules (ratio + outputSize)
        for limit, ratio_low, ratio_high, out_thresh in config.get('outputsize_rules', []):
            if ratio_low < ratio <= ratio_high and output_size < out_thresh:
                return limit
        
        return config.get('default_limit', 1)
    
    # Find limit counts
    limit_counts = {}
    for d in all_test_data:
        limit = d['best_limit']
        limit_counts[limit] = limit_counts.get(limit, 0) + 1
    
    # Default is most common limit
    default_limit = max(limit_counts.keys(), key=lambda l: limit_counts[l])
    
    # Start with basic ratio thresholds for high-ratio limits
    config = {
        'ratio_thresholds': [
            (2048, 393464),   # High ratio (covers both 1024 and 2048 cases)
            (64, 1142),       # Medium-high ratio
            (8, 8),           # Medium ratio
        ],
        'outputsize_rules': [],
        'default_limit': default_limit
    }
    
    base_accuracy = calculate_accuracy_2d(config)
    print(f"\n  Base accuracy (ratio-only): {base_accuracy}/{len(all_test_data)} ({100*base_accuracy/len(all_test_data):.1f}%)", 
          file=sys.stderr)
    
    # Try adding limit=16 rule for low ratio + small outputSize
    best_out_thresh = 50000
    best_improvement = 0
    
    for out_thresh in [30000, 40000, 50000, 60000, 75000]:
        test_config = config.copy()
        test_config['outputsize_rules'] = [(16, 0, 8, out_thresh)]
        test_config['default_limit'] = 1  # Low ratio + large output → no split
        acc = calculate_accuracy_2d(test_config)
        if acc > base_accuracy + best_improvement:
            best_improvement = acc - base_accuracy
            best_out_thresh = out_thresh
    
    config['outputsize_rules'].append((16, 0, 8, best_out_thresh))
    config['default_limit'] = 1
    
    print(f"  Added limit=16 rule (ratio<=8, outputSize<{best_out_thresh}): +{best_improvement} correct", 
          file=sys.stderr)
    
    # Try adding limit=32 rule for medium ratio + smaller outputSize
    best_out_thresh_32 = 50000
    best_ratio_thresh_32 = 150
    best_improvement_32 = 0
    current_accuracy = calculate_accuracy_2d(config)
    
    for ratio_thresh in [100, 150, 200, 250]:
        for out_thresh in [50000, 75000, 100000]:
            test_config = config.copy()
            test_config['outputsize_rules'] = config['outputsize_rules'] + [(32, 8, 1142, out_thresh)]
            # Modify ratio threshold for limit=8 to account for limit=32
            test_config['ratio_thresholds'] = [
                (1024, 1000000),
                (2048, 393464),
                (64, 1142),
                (8, ratio_thresh),  # Only use limit=8 for ratio > ratio_thresh
            ]
            acc = calculate_accuracy_2d(test_config)
            if acc > current_accuracy + best_improvement_32:
                best_improvement_32 = acc - current_accuracy
                best_out_thresh_32 = out_thresh
                best_ratio_thresh_32 = ratio_thresh
    
    if best_improvement_32 > len(all_test_data) * 0.01:  # At least 1% improvement
        # Update config with limit=32 rule
        config['outputsize_rules'].append((32, best_ratio_thresh_32, 1142, best_out_thresh_32))
        config['ratio_thresholds'] = [
            (2048, 393464),
            (64, 1142),
            (8, best_ratio_thresh_32),
        ]
        print(f"  Added limit=32 rule (ratio>{best_ratio_thresh_32}, outputSize<{best_out_thresh_32}): +{best_improvement_32} correct", 
              file=sys.stderr)
    
    # Try adding rule for large outputSize + medium ratio → limit=1 (no split is better)
    # This catches cases where ratio > 8 but outputSize is very large
    current_accuracy = calculate_accuracy_2d(config)
    best_improvement_large_out = 0
    best_large_out_thresh = 500000
    
    for out_thresh in [300000, 500000, 750000, 1000000]:
        # Test: if ratio > 8 but outputSize > out_thresh → assign limit=1 instead of 8
        test_config = config.copy()
        test_config['large_output_rule'] = (1, 8, 1142, out_thresh)  # (limit, ratio_low, ratio_high, out_thresh)
        
        correct = 0
        for d in all_test_data:
            ratio = d['ratio']
            out_size = d['output_size']
            
            # Check large output rule first
            if 8 < ratio <= 1142 and out_size > out_thresh:
                predicted = 1
            else:
                predicted = predict_2d(ratio, out_size, config)
            
            if predicted == d['best_limit']:
                correct += 1
        
        improvement = correct - current_accuracy
        if improvement > best_improvement_large_out:
            best_improvement_large_out = improvement
            best_large_out_thresh = out_thresh
    
    if best_improvement_large_out > 0:
        config['large_output_rule'] = (1, 8, 1142, best_large_out_thresh)
        print(f"  Added large output rule (ratio 8-1142, outputSize>{best_large_out_thresh}): +{best_improvement_large_out} correct", 
              file=sys.stderr)
    
    # Recalculate final accuracy with all rules
    def final_predict(ratio, output_size):
        # Check large output rule first
        if 'large_output_rule' in config:
            limit, r_low, r_high, out_thresh = config['large_output_rule']
            if r_low < ratio <= r_high and output_size > out_thresh:
                return limit
        
        return predict_2d(ratio, output_size, config)
    
    final_correct = sum(1 for d in all_test_data if final_predict(d['ratio'], d['output_size']) == d['best_limit'])
    
    # Final accuracy report
    print(f"\n  Final 2D accuracy: {final_correct}/{len(all_test_data)} ({100*final_correct/len(all_test_data):.1f}%)", 
          file=sys.stderr)
    
    return config


def find_optimal_thresholds(sorted_limits: List[int], cluster_stats: Dict[int, Dict], 
                            all_test_ratios: List[Tuple[float, int]]) -> List[Tuple[int, float]]:
    """Find optimal thresholds using greedy search to maximize accuracy.
    
    Strategy:
    1. Start with single most common limit as default
    2. Iteratively add (threshold, limit) pairs that improve accuracy most
    3. Stop when no significant improvement or max branches reached
    """
    # Collect all unique ratio values as potential thresholds
    all_ratios = sorted(set([r for r, _ in all_test_ratios]))
    
    # Calculate accuracy for a given threshold configuration
    def calculate_accuracy(thresholds: List[Tuple[int, float]]) -> int:
        correct = 0
        sorted_thresh = sorted(thresholds, key=lambda x: x[1], reverse=True)
        
        for ratio, optimal_limit in all_test_ratios:
            predicted = sorted_thresh[-1][0]  # Default
            for limit, threshold in sorted_thresh[:-1]:
                if ratio > threshold:
                    predicted = limit
                    break
            if predicted == optimal_limit:
                correct += 1
        return correct
    
    # Find the best default limit (most common)
    limit_counts = {}
    for _, optimal_limit in all_test_ratios:
        limit_counts[optimal_limit] = limit_counts.get(optimal_limit, 0) + 1
    default_limit = max(limit_counts.keys(), key=lambda l: limit_counts[l])
    
    # Start with just the default
    best_thresholds = [(default_limit, 0)]
    best_accuracy = calculate_accuracy(best_thresholds)
    
    print(f"\n  Starting with default limit={default_limit}: {best_accuracy}/{len(all_test_ratios)} correct", 
          file=sys.stderr)
    
    # Iteratively add thresholds
    max_branches = min(6, len(sorted_limits))  # Limit complexity
    available_limits = [l for l in sorted_limits if l != default_limit]
    
    for iteration in range(max_branches - 1):
        best_improvement = 0
        best_new_thresh = None
        
        # Try adding each possible (threshold, limit) pair
        for limit in available_limits:
            stats = cluster_stats[limit]
            # Try multiple candidate thresholds for this limit
            candidates = [
                stats['min'],
                stats['p25'],
                stats['median'],
                (stats['min'] + stats['median']) / 2,
            ]
            
            for threshold in candidates:
                if threshold <= 0:
                    continue
                # Skip if too close to existing threshold
                if any(abs(threshold - t) < threshold * 0.1 for _, t in best_thresholds if t > 0):
                    continue
                    
                test_thresholds = best_thresholds.copy()
                test_thresholds.append((limit, threshold))
                accuracy = calculate_accuracy(test_thresholds)
                improvement = accuracy - best_accuracy
                
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_new_thresh = (limit, threshold)
        
        # Stop if no significant improvement
        if best_improvement < len(all_test_ratios) * 0.01:  # Less than 1% improvement
            break
            
        best_thresholds.append(best_new_thresh)
        best_accuracy += best_improvement
        available_limits = [l for l in available_limits if l != best_new_thresh[0]]
        
        print(f"  Added limit={best_new_thresh[0]} at threshold={best_new_thresh[1]:.2f}: "
              f"+{best_improvement} correct, total {best_accuracy}/{len(all_test_ratios)}",
              file=sys.stderr)
    
    # Sort by threshold descending
    best_thresholds.sort(key=lambda x: x[1], reverse=True)
    
    # Ensure limit=1 is always included if there are tests for it (represents "no split")
    # limit=1 should typically be the default (else case) for LOW ratios
    limits_in_thresholds = {l for l, _ in best_thresholds}
    if 1 in cluster_stats and 1 not in limits_in_thresholds:
        limit1_stats = cluster_stats[1]
        
        # Get the current default limit (last in sorted thresholds, with threshold=0)
        current_default = best_thresholds[-1][0] if best_thresholds else default_limit
        current_default_stats = cluster_stats.get(current_default, {})
        
        # If limit=1 has lower median ratio than current default, swap them
        if current_default_stats and limit1_stats['median'] < current_default_stats.get('median', float('inf')):
            # limit=1 should be the new default (for lowest ratios)
            # Current default needs a threshold
            threshold_for_old_default = limit1_stats['max']  # Above this ratio, use old default
            
            # Remove the old default (threshold=0) and add it with proper threshold
            best_thresholds = [(l, t) for l, t in best_thresholds if t > 0]
            best_thresholds.append((current_default, threshold_for_old_default))
            best_thresholds.append((1, 0))  # limit=1 becomes the new default (else case)
            best_thresholds.sort(key=lambda x: x[1], reverse=True)
            print(f"  Ensured limit=1 is default for low ratios (ratio <= {threshold_for_old_default:.2f})", 
                  file=sys.stderr)
        else:
            # limit=1 has higher median, add it with a threshold
            limit1_threshold = limit1_stats['min']  # Use min to capture most limit=1 cases
            best_thresholds.append((1, limit1_threshold))
            best_thresholds.sort(key=lambda x: x[1], reverse=True)
            print(f"  Ensured limit=1 is included at threshold={limit1_threshold:.2f} (no split)", 
                  file=sys.stderr)
    
    # Ensure monotonic thresholds and add catch-all
    final_thresholds = []
    prev_threshold = float('inf')
    for limit, threshold in best_thresholds:
        if threshold >= prev_threshold:
            threshold = prev_threshold * 0.9
        if threshold < 0:
            threshold = 0
        final_thresholds.append((limit, threshold))
        prev_threshold = threshold
    
    # Final accuracy report
    final_accuracy = calculate_accuracy(final_thresholds)
    print(f"\n  Final accuracy: {final_accuracy}/{len(all_test_ratios)} ({final_accuracy/len(all_test_ratios)*100:.1f}%)", 
          file=sys.stderr)
    
    return final_thresholds


def generate_cpp_code(thresholds: List[Tuple[int, float]], clusters: Dict[int, List[TestData]], 
                      cluster_stats: Dict[int, Dict], test_type: str = 'conv') -> str:
    """Generate C++ code using ratio-based decision tree.
    
    Args:
        thresholds: List of (limit, threshold) tuples
        clusters: Dictionary mapping limits to test data
        cluster_stats: Statistics for each cluster (from derive_ratio_thresholds)
        test_type: 'conv' for getWeightBackwardReductionSizes, 'gemm' for getMatmulLikeReductionSizes
    """
    
    lines = []
    lines.append("// ============================================================================")
    lines.append("// UNIFIED RATIO-BASED SPLIT REDUCTION")
    lines.append("// ============================================================================")
    lines.append("//")
    lines.append(f"// Target function: {'getWeightBackwardReductionSizes' if test_type == 'conv' else 'getMatmulLikeReductionSizes'}")
    lines.append("//")
    lines.append("// This approach uses a SINGLE decision tree based on:")
    lines.append("//   ratio = kSize / sqrt(outputSize)")
    lines.append("//")
    lines.append("// Benefits:")
    lines.append("//   ✓ Combines both parallel and reduction dimensions naturally")
    lines.append("//   ✓ No separate early return mechanism")
    lines.append("//   ✓ limitParallelLoops=1 explicitly means 'no split'")
    lines.append("//")
    lines.append("// Cluster statistics (after outlier removal):")
    
    # Sort by median descending for consistent ordering
    sorted_limits = sorted(cluster_stats.keys(), key=lambda l: cluster_stats[l]['median'], reverse=True)
    for limit in sorted_limits:
        stats = cluster_stats[limit]
        outlier_info = f" ({stats['outliers_removed']} outliers removed)" if stats['outliers_removed'] > 0 else ""
        lines.append(f"//   limit={limit}: {stats['count_filtered']} tests, "
                    f"median={stats['median']:.2f}, range=[{stats['min']:.2f}-{stats['max']:.2f}]{outlier_info}")
    lines.append("//")
    lines.append("")
    
    # Different variable setup for conv vs gemm
    if test_type == 'conv':
        lines.append("// Calculate problem characteristics (convolution)")
        lines.append("int64_t outputSize = outputChannelSize * batchSize * imageSize * depthSize;")
        lines.append("SmallVector<int64_t> tileSizes = std::move(*maybeSizes);")
        lines.append("int64_t reductionSize = llvm::product_of(tileSizes);")
        lines.append("")
        lines.append("// Compute ratio (reduction vs parallel balance)")
        lines.append("double ratio = (double)reductionSize / std::sqrt((double)outputSize);")
    else:  # gemm
        lines.append("// Calculate problem characteristics (matmul-like)")
        lines.append("SmallVector<int64_t> tileSizes = std::move(*maybeSizes);")
        lines.append("int64_t outputSize = mSize * nSize * batchSize;")
        lines.append("")
        lines.append("// Compute ratio (reduction vs parallel balance)")
        lines.append("double ratio = (double)kSize / std::sqrt((double)outputSize);")
    
    lines.append("")
    lines.append("// Unified decision tree based on ratio")
    lines.append("int64_t limitParallelLoops;")
    
    # Generate if-else chain sorted by threshold (descending)
    sorted_thresholds = sorted(thresholds, key=lambda x: x[1], reverse=True)
    for i, (limit, threshold) in enumerate(sorted_thresholds):
        if i == 0:
            lines.append(f"if (ratio > {threshold:.2f}) {{")
        elif i == len(sorted_thresholds) - 1:
            lines.append("} else {")
        else:
            lines.append(f"}} else if (ratio > {threshold:.2f}) {{")
        
        lines.append(f"  limitParallelLoops = {limit};")
    lines.append("}")
    lines.append("")
    
    lines.append("// Special handling for limit=1 (no split)")
    lines.append("if (limitParallelLoops == 1) {")
    lines.append("  // Return original tile sizes without splitting")
    lines.append("  return std::nullopt;")
    lines.append("}")
    
    return "\n".join(lines)


def generate_cpp_code_2d(config: Dict, clusters: Dict[int, List], cluster_stats: Dict[int, Dict], test_type: str) -> str:
    """Generate C++ code for 2D decision tree (ratio + outputSize)."""
    lines = []
    
    lines.append("// ============================================================================")
    lines.append("// UNIFIED RATIO-BASED SPLIT REDUCTION (2D IMPROVED)")
    lines.append("// ============================================================================")
    lines.append("//")
    lines.append(f"// Target function: {'getWeightBackwardReductionSizes' if test_type == 'conv' else 'getMatmulLikeReductionSizes'}")
    lines.append("//")
    lines.append("// This improved approach uses a 2D decision tree based on:")
    lines.append("//   - ratio = reductionSize / sqrt(outputSize)")
    lines.append("//   - outputSize (as secondary discriminator)")
    lines.append("//")
    lines.append("// Key improvements over ratio-only approach:")
    lines.append("//   ✓ Added limit=16 for small outputSize + low ratio cases")
    lines.append("//   ✓ Added limit=32 for medium outputSize + medium ratio cases")
    lines.append("//   ✓ Better accuracy by using 2D decision boundaries")
    lines.append("//")
    lines.append("// Cluster statistics (after outlier removal):")
    
    # Sort by median descending for consistent ordering
    sorted_limits = sorted(cluster_stats.keys(), key=lambda l: cluster_stats[l]['median'], reverse=True)
    for limit in sorted_limits:
        stats = cluster_stats[limit]
        outlier_info = f" ({stats['outliers_removed']} outliers removed)" if stats['outliers_removed'] > 0 else ""
        lines.append(f"//   limit={limit}: {stats['count_filtered']} tests, "
                    f"median={stats['median']:.2f}, range=[{stats['min']:.2f}-{stats['max']:.2f}]{outlier_info}")
    lines.append("//")
    lines.append("")
    
    # Variable setup based on test type
    if test_type == 'conv':
        lines.append("// Calculate problem characteristics (convolution)")
        lines.append("int64_t outputSize = outputChannelSize * batchSize * imageSize * depthSize;")
        lines.append("SmallVector<int64_t> tileSizes = std::move(*maybeSizes);")
        lines.append("int64_t reductionSize = llvm::product_of(tileSizes);")
        lines.append("")
        lines.append("// Compute ratio (reduction vs parallel balance)")
        lines.append("double ratio = (double)reductionSize / std::sqrt((double)outputSize);")
    else:  # gemm
        lines.append("// Calculate problem characteristics (matmul-like)")
        lines.append("SmallVector<int64_t> tileSizes = std::move(*maybeSizes);")
        lines.append("int64_t outputSize = mSize * nSize * batchSize;")
        lines.append("")
        lines.append("// Compute ratio (reduction vs parallel balance)")
        lines.append("double ratio = (double)kSize / std::sqrt((double)outputSize);")
    
    lines.append("")
    lines.append("// 2D decision tree with ratio and outputSize")
    lines.append("int64_t limitParallelLoops;")
    lines.append("")
    
    # Generate decision tree from config
    ratio_thresholds = config.get('ratio_thresholds', [])
    outputsize_rules = config.get('outputsize_rules', [])
    large_output_rule = config.get('large_output_rule', None)  # (limit, ratio_low, ratio_high, out_thresh)
    default_limit = config.get('default_limit', 1)
    
    # High ratio conditions first
    first_cond = True
    for limit, thresh in ratio_thresholds:
        if first_cond:
            lines.append(f"if (ratio > {thresh:.0f}) {{")
            first_cond = False
        else:
            lines.append(f"}} else if (ratio > {thresh:.0f}) {{")
        
        # Check if this is the range with large_output_rule
        if large_output_rule and large_output_rule[1] == thresh:
            # This is the medium ratio range - add nested condition
            lines.append(f"  if (outputSize > {large_output_rule[3]}) {{")
            lines.append(f"    limitParallelLoops = {large_output_rule[0]};  // Large output → no split")
            lines.append("  } else {")
            lines.append(f"    limitParallelLoops = {limit};")
            lines.append("  }")
        else:
            # Check for nested outputSize conditions from outputsize_rules
            nested_rules = [(l, r_low, r_high, out_t) for l, r_low, r_high, out_t in outputsize_rules 
                           if r_low <= thresh <= r_high or r_low == thresh]
            if nested_rules:
                for i, (rule_limit, r_low, r_high, out_thresh) in enumerate(nested_rules):
                    if i == 0:
                        lines.append(f"  if (outputSize < {out_thresh}) {{")
                        lines.append(f"    limitParallelLoops = {rule_limit};")
                        lines.append("  } else {")
                        lines.append(f"    limitParallelLoops = {limit};")
                        lines.append("  }")
                        break
            else:
                lines.append(f"  limitParallelLoops = {limit};")
    
    # Handle low-ratio case with outputSize discrimination
    low_ratio_rules = [(l, r_low, r_high, out_t) for l, r_low, r_high, out_t in outputsize_rules 
                      if r_low == 0 or r_high <= 8]
    
    lines.append("} else {")
    if low_ratio_rules:
        rule = low_ratio_rules[0]  # Take first low-ratio rule
        lines.append(f"  // Low ratio - use outputSize to distinguish")
        lines.append(f"  if (outputSize < {rule[3]}) {{")
        lines.append(f"    limitParallelLoops = {rule[0]};  // Small output + low ratio")
        lines.append("  } else {")
        lines.append(f"    limitParallelLoops = {default_limit};  // Large output + low ratio → no split")
        lines.append("  }")
    else:
        lines.append(f"  limitParallelLoops = {default_limit};")
    lines.append("}")
    
    lines.append("")
    lines.append("// Special handling for limit=1 (no split)")
    lines.append("if (limitParallelLoops == 1) {")
    lines.append("  return std::nullopt;")
    lines.append("}")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Generate ratio-based C++ code for split reduction'
    )
    parser.add_argument('--results-file', required=True, type=Path,
                       help='Path to limitParallelLoops_sweep_results.json')
    parser.add_argument('--output', type=Path,
                       help='Output file for generated C++ code (default: print to stdout)')
    parser.add_argument('--test-type', choices=['conv', 'gemm', 'auto'], default='auto',
                       help='Type of tests: conv (weight backward), gemm (matmul-like), or auto-detect')
    parser.add_argument('--dimensions-file', type=Path,
                       help='Path to captured_dimensions.json (if not provided, will look in same dir as results)')
    parser.add_argument('--use-2d', action='store_true',
                       help='Use 2D decision tree with both ratio and outputSize (recommended)')
    
    args = parser.parse_args()
    
    if not args.results_file.exists():
        print(f"Error: Results file not found: {args.results_file}")
        return 1
    
    # Load captured dimensions (for test_config matching)
    captured_dims = {}
    dims_file = args.dimensions_file if args.dimensions_file else args.results_file.parent / "captured_dimensions.json"
    if dims_file.exists():
        print(f"Loading captured dimensions from {dims_file}...", file=sys.stderr)
        captured_dims = load_captured_dimensions(dims_file.parent if not args.dimensions_file else dims_file.parent)
        # Reload with correct path if dimensions_file is specified
        if args.dimensions_file:
            with open(dims_file, 'r') as f:
                dims_list = json.load(f)
            captured_dims = {entry.get('test_config'): entry for entry in dims_list if entry.get('test_config')}
        print(f"  Loaded {len(captured_dims)} test configurations with dimensions", file=sys.stderr)
    else:
        print(f"  No captured_dimensions.json found, will parse dimensions from test names", file=sys.stderr)
    
    # Load and analyze data
    print("Loading results and computing ratios...", file=sys.stderr)
    test_data, detected_type = load_and_analyze(args.results_file, captured_dims)
    print(f"Loaded {len(test_data)} test cases", file=sys.stderr)
    print(f"Detected test type: {detected_type}", file=sys.stderr)
    
    if len(test_data) == 0:
        print("Error: No valid test data found", file=sys.stderr)
        return 1
    
    # Determine test type
    if args.test_type == 'auto':
        test_type = detected_type if detected_type in ['conv', 'gemm'] else 'conv'
    else:
        test_type = args.test_type
    
    print(f"Generating code for: {test_type}", file=sys.stderr)
    
    # Cluster by optimal limit
    print("Clustering by optimal limitParallelLoops...", file=sys.stderr)
    clusters = cluster_by_ratio(test_data)
    
    # Derive ratio thresholds with outlier removal
    print("Deriving ratio-based thresholds (with outlier removal)...", file=sys.stderr)
    thresholds, cluster_stats = derive_ratio_thresholds(clusters)
    
    print("\nDerived thresholds:", file=sys.stderr)
    for limit, threshold in sorted(thresholds, key=lambda x: x[1], reverse=True):
        print(f"  limit={limit}: ratio > {threshold:.2f}", file=sys.stderr)
    
    # Use 2D approach only if explicitly requested
    if args.use_2d:
        print("\nUsing 2D decision tree (ratio + outputSize)...", file=sys.stderr)
        
        # Prepare data for 2D threshold finding
        all_test_data = []
        for d in test_data:
            if d.ratio > 0 and d.output_size > 0:
                all_test_data.append({
                    'ratio': d.ratio,
                    'output_size': d.output_size,
                    'best_limit': d.best_limit
                })
        
        # Find optimal 2D thresholds
        sorted_limits = sorted(cluster_stats.keys())
        config_2d = find_optimal_thresholds_2d(sorted_limits, cluster_stats, all_test_data)
        
        # Generate C++ code with 2D decision tree
        print(f"\nGenerating 2D C++ code for {test_type}...\n", file=sys.stderr)
        cpp_code = generate_cpp_code_2d(config_2d, clusters, cluster_stats, test_type)
    else:
        # Generate standard ratio-only C++ code
        print(f"\nGenerating C++ code for {test_type}...\n", file=sys.stderr)
        cpp_code = generate_cpp_code(thresholds, clusters, cluster_stats, test_type)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(cpp_code)
        print(f"✓ C++ code saved to: {args.output}", file=sys.stderr)
        print(f"  Target function: {'getWeightBackwardReductionSizes' if test_type == 'conv' else 'getMatmulLikeReductionSizes'}", file=sys.stderr)
    else:
        print(cpp_code)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
