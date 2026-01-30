#!/usr/bin/env python3
"""
Comprehensive Analysis for limitParallelLoops Results

This script provides complete analysis including:
1. Threshold recommendations (for C++ code generation)
2. Cluster analysis (grouping tests by optimal limit)
3. Comparison metrics (win rates, geometric mean)
4. Statistical summaries
5. Baseline comparisons
"""

import json
import csv
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Union
from dataclasses import dataclass
from collections import defaultdict
import statistics
import re


@dataclass
class TestCharacteristics:
    """Problem characteristics extracted from CSV data"""
    test_name: str
    best_limit: int
    best_runtime: float
    speedup: float
    
    # Problem characteristics
    output_size: int  # M * N for conv/matmul
    k_size: int       # K dimension (reduction)
    ratio: float      # kSize / outputSize
    
    # Raw dimensions
    m: int = 0
    n: int = 0
    k: int = 0
    
    # All runtimes for this test
    all_runtimes: Dict[Union[int, str], float] = None


class ComprehensiveAnalyzer:
    """Comprehensive analyzer combining threshold and comparison analysis"""
    
    def __init__(self, results_file: Path):
        self.results_file = results_file
        with open(results_file, 'r') as f:
            self.data = json.load(f)
        
        self.analyses = self.data['analyses']
        self.candidate_limits = self.data['candidate_limits']
    
    def extract_characteristics(self) -> List[TestCharacteristics]:
        """Extract problem characteristics from all test cases"""
        characteristics = []
        
        for test_name, analysis in self.analyses.items():
            csv_data = analysis['csv_data']
            
            # Parse dimensions from arguments string
            args = csv_data.get('arguments', '')
            dims = self._parse_arguments(args)
            
            m = dims['m']
            n = dims['n']
            k = dims['k']
            
            # Calculate characteristics
            output_size = m * n if (m and n) else 0
            k_size = k if k else 0
            ratio = k_size / output_size if output_size > 0 else 0.0
            
            # Convert all_runtimes keys from strings to ints (except "baseline")
            all_runtimes = analysis.get('all_runtimes', {})
            if all_runtimes:
                converted = {}
                for k, v in all_runtimes.items():
                    # Keep "baseline" as string, convert others to int
                    if k == "baseline" or (isinstance(k, str) and k.lower() == "baseline"):
                        converted["baseline"] = float(v)
                    else:
                        converted[int(k)] = float(v)
                all_runtimes = converted
            
            characteristics.append(TestCharacteristics(
                test_name=test_name,
                best_limit=analysis['best_limit'],
                best_runtime=analysis['best_runtime'],
                speedup=analysis['speedup_vs_worst'],
                output_size=output_size,
                k_size=k_size,
                ratio=ratio,
                m=m,
                n=n,
                k=k,
                all_runtimes=all_runtimes
            ))
        
        return characteristics
    
    def _parse_arguments(self, args_string: str) -> Dict[str, int]:
        """Parse dimensions from convolution/matmul arguments string"""
        def extract_value(flag: str) -> int:
            pattern = rf'{flag}\s+(\d+)'
            match = re.search(pattern, args_string)
            return int(match.group(1)) if match else 0
        
        batch = extract_value('-n')
        height = extract_value('-H')
        width = extract_value('-W')
        in_channels = extract_value('-c')
        out_channels = extract_value('-k')
        depth = extract_value('--in_d') or 1
        
        filter_y = extract_value('-y')
        filter_x = extract_value('-x')
        filter_d = extract_value('--fil_d') or 1
        filter_spatial = filter_y * filter_x * filter_d if (filter_y and filter_x) else 0
        
        # For weight backward convolutions:
        # - Output (parallel): filter weights = k * c * y * x * d
        # - Reduction: input spatial = batch * height * width * depth
        output_size = out_channels * in_channels * filter_spatial if (out_channels and in_channels and filter_spatial) else 0
        
        input_spatial = height * width * depth if (height and width) else 0
        k_size = batch * input_spatial if (batch and input_spatial) else 0
        
        return {
            'm': output_size,
            'n': 1,
            'k': k_size
        }
    
    def compute_metrics(self, runtimes: List[float]) -> Dict:
        """Compute statistical metrics for a list of runtimes"""
        if not runtimes:
            return {}
        
        sorted_runtimes = sorted(runtimes)
        n = len(sorted_runtimes)
        
        # Geometric mean (using log space to avoid overflow)
        import math
        try:
            # Filter out zero or negative runtimes
            valid_runtimes = [r for r in runtimes if r > 0]
            if not valid_runtimes:
                geo_mean = float('inf')
            else:
                log_sum = sum(math.log(r) for r in valid_runtimes)
                geo_mean = math.exp(log_sum / len(valid_runtimes))
        except (ValueError, OverflowError):
            geo_mean = float('inf')
        
        # Percentiles
        p95_idx = int(0.95 * n)
        p99_idx = int(0.99 * n)
        
        return {
            'geometric_mean': geo_mean,
            'arithmetic_mean': statistics.mean(runtimes),
            'median': statistics.median(runtimes),
            'min': min(runtimes),
            'max': max(runtimes),
            'p95': sorted_runtimes[min(p95_idx, n-1)],
            'p99': sorted_runtimes[min(p99_idx, n-1)],
            'total_runtime': sum(runtimes),
            'count': n
        }
    
    def find_viable_limits(self, char: TestCharacteristics, tolerance: float = 0.15) -> List[int]:
        """Find all limits that perform within tolerance of the best for this test.
        
        Args:
            char: Test characteristics
            tolerance: Performance tolerance (default 15% - limits within 15% of best are viable)
        
        Returns:
            List of viable limit values, sorted from largest to smallest
        """
        if not char.all_runtimes:
            return []
        
        # Find the best runtime among numeric limits (exclude baseline)
        numeric_limits = {k: v for k, v in char.all_runtimes.items() 
                         if isinstance(k, int) and k > 0}
        
        if not numeric_limits:
            return []
        
        best_runtime = min(numeric_limits.values())
        
        # Find all limits within tolerance of the best
        viable = []
        for limit, runtime in numeric_limits.items():
            if (runtime - best_runtime) / best_runtime <= tolerance:
                viable.append(limit)
        
        return sorted(viable, reverse=True)  # Prefer larger limits when tied
    
    def cluster_by_limit(self, characteristics: List[TestCharacteristics]) -> Dict[Union[int, str], List[TestCharacteristics]]:
        """Group test cases by their optimal limitParallelLoops value.
        
        Strategy: Sort tests by output_size, then assign each to its best-performing limit.
        When multiple limits perform similarly (within 5%), prefer consistency with 
        neighbors of similar output_size.
        """
        # Filter out baseline-best tests
        valid_tests = []
        for char in characteristics:
            if isinstance(char.best_limit, str) and char.best_limit == "baseline":
                continue
            viable_limits = self.find_viable_limits(char, tolerance=0.15)
            if viable_limits:
                valid_tests.append((char, viable_limits))
        
        if not valid_tests:
            return {}
        
        # Sort by output_size
        valid_tests.sort(key=lambda x: x[0].output_size)
        
        # Assign limits with neighborhood consensus
        window_size = 5  # Look at 5 neighbors on each side
        assigned_limits = []
        
        for i, (char, viable_limits) in enumerate(valid_tests):
            if len(viable_limits) == 1:
                # Only one viable option
                assigned_limits.append((char, viable_limits[0]))
            else:
                # Multiple viable - check what neighbors prefer
                neighbor_votes = defaultdict(int)
                
                # Look at neighbors within window
                start = max(0, i - window_size)
                end = min(len(valid_tests), i + window_size + 1)
                
                for j in range(start, end):
                    if j == i:
                        continue
                    neighbor_char, neighbor_viables = valid_tests[j]
                    # If neighbor has strong preference (only 1-2 viable), weight it more
                    weight = 3 if len(neighbor_viables) <= 2 else 1
                    for limit in neighbor_viables:
                        if limit in viable_limits:
                            neighbor_votes[limit] += weight
                
                # Choose the viable limit with most neighbor support
                if neighbor_votes:
                    chosen = max(viable_limits, key=lambda x: (neighbor_votes.get(x, 0), -x))
                else:
                    # No neighbor consensus - prefer smaller limit (more aggressive splitting)
                    # for small outputs, larger limit (less splitting) for large outputs
                    if char.output_size < 100000:
                        chosen = max(viable_limits)  # Prefer large limitParallelLoops for small outputs
                    else:
                        chosen = min(viable_limits)  # Prefer small limitParallelLoops for large outputs
                
                assigned_limits.append((char, chosen))
        
        # Group into clusters
        clusters = defaultdict(list)
        for char, limit in assigned_limits:
            clusters[limit].append(char)
        
        return dict(clusters)
    
    def analyze_cluster(self, cluster: List[TestCharacteristics]) -> Dict:
        """Analyze characteristics of a cluster"""
        if not cluster:
            return {}
        
        output_sizes = [c.output_size for c in cluster if c.output_size > 0]
        k_sizes = [c.k_size for c in cluster if c.k_size > 0]
        ratios = [c.ratio for c in cluster if c.ratio > 0]
        runtimes = [c.best_runtime for c in cluster]
        speedups = [c.speedup for c in cluster]
        
        return {
            'count': len(cluster),
            'output_size': {
                'min': min(output_sizes) if output_sizes else 0,
                'max': max(output_sizes) if output_sizes else 0,
                'median': sorted(output_sizes)[len(output_sizes)//2] if output_sizes else 0,
            },
            'k_size': {
                'min': min(k_sizes) if k_sizes else 0,
                'max': max(k_sizes) if k_sizes else 0,
                'median': sorted(k_sizes)[len(k_sizes)//2] if k_sizes else 0,
            },
            'ratio': {
                'min': min(ratios) if ratios else 0,
                'max': max(ratios) if ratios else 0,
                'median': sorted(ratios)[len(ratios)//2] if ratios else 0,
            },
            'performance': {
                'avg_runtime_ms': sum(runtimes) / len(runtimes) if runtimes else 0,
                'avg_speedup': sum(speedups) / len(speedups) if speedups else 1.0,
            }
        }
    
    def compare_to_baseline(self, characteristics: List[TestCharacteristics], baseline_limit: Union[int, str]) -> Dict:
        """Compare all limits to a baseline limit"""
        if baseline_limit not in self.candidate_limits:
            return {}
        
        comparison = {}
        
        for char in characteristics:
            if not char.all_runtimes or baseline_limit not in char.all_runtimes:
                continue
            
            baseline_runtime = char.all_runtimes[baseline_limit]
            
            for limit, runtime in char.all_runtimes.items():
                if limit == baseline_limit:
                    continue
                
                if limit not in comparison:
                    comparison[limit] = {
                        'faster': 0,
                        'slower': 0,
                        'same': 0,
                        'improvements': [],
                        'regressions': []
                    }
                
                # Consider < 1% difference as "same"
                diff_pct = abs((runtime - baseline_runtime) / baseline_runtime) * 100
                
                if diff_pct < 1.0:
                    comparison[limit]['same'] += 1
                elif runtime < baseline_runtime:
                    comparison[limit]['faster'] += 1
                    improvement = (baseline_runtime - runtime) / baseline_runtime * 100
                    comparison[limit]['improvements'].append(improvement)
                else:
                    comparison[limit]['slower'] += 1
                    regression = (runtime - baseline_runtime) / baseline_runtime * 100
                    comparison[limit]['regressions'].append(regression)
        
        return comparison
    
    def derive_thresholds(self, clusters: Dict[Union[int, str], List[TestCharacteristics]]) -> Dict:
        """Derive threshold constants based on cluster boundaries.
        
        Strategy:
        1. Sort clusters by their typical output_size (median)
        2. Find separation points between adjacent clusters
        3. Generate clean, non-overlapping thresholds
        """
        # Filter out non-integer keys (like "baseline")
        numeric_limits = [k for k in clusters.keys() if isinstance(k, int)]
        
        if not numeric_limits:
            return {'thresholds': [], 'sorted_limits': []}
        
        # Compute statistics for each cluster
        cluster_stats = []
        for limit in numeric_limits:
            cluster = clusters[limit]
            stats = self.analyze_cluster(cluster)
            
            if not stats or stats['count'] == 0:
                continue
            
            cluster_stats.append({
                'limit': limit,
                'stats': stats,
                'median_output_size': stats['output_size']['median'],
                'max_output_size': stats['output_size']['max'],
                'min_output_size': stats['output_size']['min'],
                'count': stats['count'],
                'avg_speedup': stats['performance']['avg_speedup']
            })
        
        # Sort by minimum output_size (establish clear ordering)
        cluster_stats.sort(key=lambda x: x['min_output_size'])
        
        # Merge or re-order clusters that have significant overlap
        # This handles cases where similar output_sizes got assigned to different limits
        cleaned_clusters = []
        used_thresholds = set()
        
        for i, cluster_info in enumerate(cluster_stats):
            limit = cluster_info['limit']
            count = cluster_info['count']
            avg_speedup = cluster_info['avg_speedup']
            current_max = cluster_info['max_output_size']
            current_min = cluster_info['min_output_size']
            
            if i < len(cluster_stats) - 1:
                next_min = cluster_stats[i + 1]['min_output_size']
                
                # Find a unique threshold that separates this cluster from the next
                if current_max >= next_min:
                    # Overlap exists - use a point between current_min and next_min
                    # that avoids collision with already-used thresholds
                    threshold = (current_max + next_min) // 2
                    
                    # Ensure uniqueness
                    while threshold in used_thresholds and threshold < current_max * 2:
                        threshold = int(threshold * 1.1) + 1
                else:
                    # No overlap - use geometric mean
                    threshold = int((current_max * next_min) ** 0.5)
                
                used_thresholds.add(threshold)
            else:
                # Last cluster - no threshold needed (this is the else case)
                threshold = None
            
            # Round threshold to nearest square number
            if threshold:
                # Common square thresholds for convolution/matmul dimensions
                square_sizes = [
                    16*16,      # 256
                    32*32,      # 1,024
                    64*64,      # 4,096
                    128*128,    # 16,384
                    256*256,    # 65,536
                    384*384,    # 147,456
                    512*512,    # 262,144
                    864*864,    # 746,496
                    1024*1024,  # 1,048,576
                    1728*1728,  # 2,985,984
                    2048*2048,  # 4,194,304
                    3072*3072,  # 9,437,184
                    3456*3456,  # 11,943,936
                    4096*4096,  # 16,777,216
                ]
                
                # Find the closest square size from the list
                closest_square = min(square_sizes, key=lambda x: abs(x - threshold))
                
                # If within 35% of a common square, use it
                if abs(closest_square - threshold) / threshold < 0.35:
                    threshold = closest_square
                else:
                    # For large thresholds not in the list, round to nearest perfect square
                    import math
                    sqrt_val = int(math.sqrt(threshold))
                    
                    # Check if current is already close to a perfect square
                    if abs(sqrt_val * sqrt_val - threshold) / threshold < 0.01:
                        # Already very close to a perfect square
                        threshold = sqrt_val * sqrt_val
                    else:
                        # Round to nearest perfect square
                        lower_square = sqrt_val * sqrt_val
                        upper_square = (sqrt_val + 1) * (sqrt_val + 1)
                        
                        if abs(lower_square - threshold) < abs(upper_square - threshold):
                            threshold = lower_square
                        else:
                            threshold = upper_square
                
                # Ensure rounded threshold is still unique
                # Also check if we're within 1% of an already-used threshold - if so, merge
                closest_used = None
                min_diff = float('inf')
                for used_thresh in used_thresholds:
                    diff_pct = abs(threshold - used_thresh) / used_thresh if used_thresh > 0 else float('inf')
                    if diff_pct < 0.01 and diff_pct < min_diff:  # Within 1%
                        closest_used = used_thresh
                        min_diff = diff_pct
                
                if closest_used is not None:
                    # Merge with closest threshold
                    threshold = closest_used
                else:
                    # Ensure uniqueness by incrementing if needed
                    while threshold in used_thresholds:
                        # Move to next square size or add increment
                        next_squares = [s for s in square_sizes if s > threshold]
                        if next_squares:
                            threshold = next_squares[0]
                        else:
                            threshold += 1024
                    used_thresholds.add(threshold)
            
            cleaned_clusters.append({
                'limit': limit,
                'threshold': threshold,
                'output_size_range': [current_min, current_max],
                'count': count,
                'avg_speedup': avg_speedup
            })
        
        thresholds = cleaned_clusters
        
        # Merge clusters with very close thresholds (within 50% of each other)
        # This reduces overfitting and creates more general, robust heuristics
        # Aligns with user feedback to avoid too fine-grained recommendations
        merged_thresholds = []
        i = 0
        while i < len(thresholds):
            current = thresholds[i]
            
            # Try to merge with next clusters that have similar thresholds
            merge_group = [current]
            j = i + 1
            
            while j < len(thresholds):
                next_cluster = thresholds[j]
                
                # Skip if next cluster has no threshold (it's the else case)
                if next_cluster['threshold'] is None:
                    break
                
                # Check if thresholds are within 50% of each other
                # Also merge if both thresholds are small (< 10000) and within 2x of each other
                if current['threshold'] and next_cluster['threshold']:
                    threshold_diff = abs(next_cluster['threshold'] - current['threshold']) / current['threshold']
                    
                    # More aggressive merging for small thresholds
                    if current['threshold'] < 10000:
                        merge_tolerance = 1.0  # Within 100% (2x) for small thresholds
                    else:
                        merge_tolerance = 0.50  # Within 50% for larger thresholds
                    
                    if threshold_diff < merge_tolerance:
                        merge_group.append(next_cluster)
                        j += 1
                    else:
                        break
                else:
                    break
            
            # If we're merging multiple clusters, pick the best one based on speedup
            if len(merge_group) > 1:
                # Find the cluster with best average speedup
                best_cluster = max(merge_group, key=lambda x: x['avg_speedup'])
                
                # Combine counts and recalculate average speedup
                total_count = sum(c['count'] for c in merge_group)
                weighted_speedup = sum(c['count'] * c['avg_speedup'] for c in merge_group) / total_count
                
                # Use the highest threshold from the group
                max_threshold = max(c['threshold'] for c in merge_group if c['threshold'])
                
                merged_cluster = {
                    'limit': best_cluster['limit'],
                    'threshold': max_threshold,
                    'output_size_range': [
                        min(c['output_size_range'][0] for c in merge_group),
                        max(c['output_size_range'][1] for c in merge_group)
                    ],
                    'count': total_count,
                    'avg_speedup': weighted_speedup
                }
                merged_thresholds.append(merged_cluster)
            else:
                merged_thresholds.append(current)
            
            i = j if j > i else i + 1
        
        # Add any remaining clusters (like the else case)
        if i < len(thresholds):
            merged_thresholds.extend(thresholds[i:])
        
        # Sort merged thresholds by threshold value to ensure correct ordering
        # None thresholds (else case) should be last
        def threshold_sort_key(t):
            if t['threshold'] is None:
                return float('inf')
            return t['threshold']
        
        merged_thresholds.sort(key=threshold_sort_key)
        
        # CRITICAL: Enforce monotonic decreasing limitParallelLoops as outputSize increases
        # The pattern should be: smaller outputSize → larger limitParallelLoops
        # As we iterate through increasing thresholds, limits must decrease
        enforced_thresholds = []
        max_limit_seen = float('inf')
        
        for thresh in merged_thresholds:
            if thresh['threshold'] is None:
                # Else case - always add
                enforced_thresholds.append(thresh)
                continue
            
            current_limit = thresh['limit']
            
            # If this limit violates monotonicity (is larger than previous), skip it
            # We prefer to skip rather than adjust to maintain data fidelity
            if current_limit < max_limit_seen:
                enforced_thresholds.append(thresh)
                max_limit_seen = current_limit
            # If equal, we can keep it as it doesn't violate monotonicity
            elif current_limit == max_limit_seen:
                enforced_thresholds.append(thresh)
            # If current_limit > max_limit_seen, this violates the trend - skip this cluster
        
        # If we skipped all clusters, fall back to original
        if not enforced_thresholds or (len(enforced_thresholds) == 1 and enforced_thresholds[0]['threshold'] is None):
            enforced_thresholds = merged_thresholds
            max_limit_seen = float('inf')
        
        # Remove duplicate thresholds - keep the one with best speedup
        # This handles cases where different clusters rounded to the same threshold
        deduplicated = []
        
        for i, thresh in enumerate(enforced_thresholds):
            t_val = thresh['threshold']
            
            # Check if this is a duplicate of the previous threshold
            is_duplicate = False
            if i > 0 and deduplicated:
                prev_thresh = deduplicated[-1]['threshold']
                # Check for exact equality
                if prev_thresh is not None and t_val is not None and prev_thresh == t_val:
                    is_duplicate = True
            
            if t_val is None:
                # Always keep the else case
                deduplicated.append(thresh)
            elif not is_duplicate:
                # New unique threshold
                deduplicated.append(thresh)
            else:
                # Duplicate threshold - merge with previous entry
                # Pick the limit with better average speedup
                prev_entry = deduplicated[-1]
                if thresh['avg_speedup'] > prev_entry['avg_speedup']:
                    # Replace with better performing limit
                    deduplicated[-1] = {
                        'limit': thresh['limit'],
                        'threshold': t_val,
                        'output_size_range': [
                            min(prev_entry['output_size_range'][0], thresh['output_size_range'][0]),
                            max(prev_entry['output_size_range'][1], thresh['output_size_range'][1])
                        ],
                        'count': prev_entry['count'] + thresh['count'],
                        'avg_speedup': max(prev_entry['avg_speedup'], thresh['avg_speedup'])
                    }
                else:
                    # Keep existing limit, but update counts
                    deduplicated[-1]['count'] += thresh['count']
                    deduplicated[-1]['output_size_range'] = [
                        min(deduplicated[-1]['output_size_range'][0], thresh['output_size_range'][0]),
                        max(deduplicated[-1]['output_size_range'][1], thresh['output_size_range'][1])
                    ]
        
        return {
            'thresholds': deduplicated,
            'sorted_limits': [t['limit'] for t in deduplicated]
        }
    
    def detect_dataset_type(self, characteristics: List[TestCharacteristics]) -> str:
        """Detect if dataset is convolution-only, matmul-only, or mixed"""
        conv_count = 0
        matmul_count = 0
        
        for char in characteristics:
            op_type = self.detect_operation_type(char.test_name)
            if op_type == 'conv':
                conv_count += 1
            else:
                matmul_count += 1
        
        if matmul_count == 0:
            return 'conv-only'
        elif conv_count == 0:
            return 'matmul-only'
        else:
            return 'mixed'
    
    def detect_operation_type(self, test_name: str) -> str:
        """Detect if test is convolution or matmul based on test name"""
        # Weight backward convolutions typically have filter dimensions
        # Matmuls typically have -y 1 -x 1 (no spatial convolution)
        if '-y 1 -x 1' in test_name or 'matmul' in test_name.lower():
            return 'matmul'
        else:
            return 'conv'
    
    def analyze_early_return_candidates(self, characteristics: List[TestCharacteristics]) -> Dict:
        """Analyze cases where limit=1 (no split) performs best.
        
        Separates analysis for convolution vs matmul operations since they have
        different early return parameters in the C++ code.
        """
        # Find tests where limit=1 is the best (or within 5% of best)
        no_split_better_conv = []
        no_split_better_matmul = []
        
        for char in characteristics:
            if 1 not in char.all_runtimes:
                continue
            
            limit1_runtime = char.all_runtimes[1]
            
            # Check if any split limit is significantly better (>5%)
            split_helps = False
            for limit, runtime in char.all_runtimes.items():
                if isinstance(limit, int) and limit > 1:
                    improvement = (limit1_runtime - runtime) / limit1_runtime
                    if improvement > 0.05:  # More than 5% improvement
                        split_helps = True
                        break
            
            if not split_helps:
                op_type = self.detect_operation_type(char.test_name)
                if op_type == 'conv':
                    no_split_better_conv.append(char)
                else:
                    no_split_better_matmul.append(char)
        
        return {
            'conv': self._analyze_early_return_for_type(no_split_better_conv, 'conv', len(characteristics)),
            'matmul': self._analyze_early_return_for_type(no_split_better_matmul, 'matmul', len(characteristics))
        }
    
    def _analyze_early_return_for_type(self, no_split_better: List[TestCharacteristics], 
                                       op_type: str, total_tests: int) -> Dict:
        """Analyze early return characteristics for a specific operation type"""
        if not no_split_better:
            return {
                'count': 0,
                'percentage': 0.0,
                'recommendation': 'Split reduction helps for all test cases.'
            }
        
        # Analyze characteristics
        output_sizes = [c.output_size for c in no_split_better]
        k_sizes = [c.k_size for c in no_split_better]
        
        # Calculate ratio based on operation type
        if op_type == 'conv':
            # For conv: ratio = reductionSize / sqrt(outputChannelSize * batchSize)
            # We approximate this with k_size / sqrt(output_size)
            ratios = [c.k_size / (c.output_size ** 0.5) if c.output_size > 0 else 0 
                     for c in no_split_better]
        else:
            # For matmul: ratio = kSize / sqrt(mSize * nSize) / batchSize
            # We approximate this with k_size / output_size (since output ≈ m*n)
            ratios = [c.k_size / c.output_size if c.output_size > 0 else 0 
                     for c in no_split_better]
        
        return {
            'count': len(no_split_better),
            'percentage': len(no_split_better) / total_tests * 100 if total_tests > 0 else 0,
            'output_size': {
                'min': min(output_sizes),
                'max': max(output_sizes),
                'median': sorted(output_sizes)[len(output_sizes)//2],
            },
            'k_size': {
                'min': min(k_sizes),
                'max': max(k_sizes),
                'median': sorted(k_sizes)[len(k_sizes)//2],
            },
            'ratio': {
                'min': min(ratios) if ratios else 0,
                'max': max(ratios) if ratios else 0,
                'median': sorted(ratios)[len(ratios)//2] if ratios else 0,
            },
            'examples': no_split_better[:3]
        }
    
    def generate_early_return_code(self, early_return_analysis: Dict, dataset_type: str) -> str:
        """Generate C++ code for early return conditions.
        
        Generates separate recommendations for convolution and matmul operations,
        matching the structure and variable names in the original C++ code.
        Only shows recommendations for operation types present in the dataset.
        """
        code_lines = []
        
        # Convolution early returns
        conv_analysis = early_return_analysis.get('conv', {})
        matmul_analysis = early_return_analysis.get('matmul', {})
        
        # Only show convolution section if we have convolution tests
        if dataset_type in ['conv-only', 'mixed']:
            if conv_analysis.get('count', 0) > 0:
                code_lines.append("FOR CONVOLUTION (getWeightBackwardReductionSizes):")
                code_lines.append("-" * 80)
                code_lines.append("")
                
                output_min = conv_analysis['output_size']['min']
                k_min = conv_analysis['k_size']['min']
                ratio_max = int(conv_analysis['ratio']['max'])
                
                # Use convolution-specific parameter names
                code_lines.append(f"  // Based on {conv_analysis['count']} tests where split doesn't help")
                code_lines.append("  const int64_t largeParallelSize = 512;  // For outputChannelSize and batchSize")
                code_lines.append(f"  const int64_t largeReductionSize = {k_min};")
                code_lines.append(f"  const int64_t ratioThreshold = {ratio_max};")
                code_lines.append("")
                code_lines.append("  // Skip if output is large and distributed across many workgroups")
                code_lines.append("  if (outputChannelSize >= largeParallelSize &&")
                code_lines.append("      batchSize >= largeParallelSize) {")
                code_lines.append("    return std::nullopt;")
                code_lines.append("  }")
                code_lines.append("")
                code_lines.append("  // Skip if reduction is small relative to output")
                code_lines.append("  int64_t reductionSize = llvm::product_of(tileSizes);")
                code_lines.append("  int64_t ratio = reductionSize / std::sqrt(outputChannelSize * batchSize);")
                code_lines.append("  if (ratio <= ratioThreshold && reductionSize < largeReductionSize) {")
                code_lines.append("    return std::nullopt;")
                code_lines.append("  }")
            else:
                code_lines.append("FOR CONVOLUTION (getWeightBackwardReductionSizes):")
                code_lines.append("-" * 80)
                code_lines.append("")
                code_lines.append("Split reduction helps for all convolution test cases.")
                code_lines.append("Current early return thresholds seem appropriate:")
                code_lines.append("  const int64_t largeParallelSize = 512;")
                code_lines.append("  const int64_t largeReductionSize = 8192;")
                code_lines.append("  const int64_t ratioThreshold = 64;")
            
            # Add separator if showing both sections
            if dataset_type == 'mixed':
                code_lines.append("")
                code_lines.append("")
        
        # Only show matmul section if we have matmul tests
        if dataset_type in ['matmul-only', 'mixed']:
            if matmul_analysis.get('count', 0) > 0:
                code_lines.append("FOR MATMUL (getMatmulLikeReductionSizes):")
                code_lines.append("-" * 80)
                code_lines.append("")
                
                output_min = matmul_analysis['output_size']['min']
                k_min = matmul_analysis['k_size']['min']
                ratio_max = int(matmul_analysis['ratio']['max'])
                
                # Use matmul-specific parameter names
                code_lines.append(f"  // Based on {matmul_analysis['count']} tests where split doesn't help")
                code_lines.append(f"  const int64_t ratioThreshold = {ratio_max};")
                code_lines.append(f"  const int64_t largeKSize = {k_min};")
                code_lines.append("  const int64_t largeMNSize = 1024;")
                code_lines.append("")
                code_lines.append("  // Skip if M or N is large")
                code_lines.append("  if (mSize > largeMNSize || nSize > largeMNSize) {")
                code_lines.append("    return std::nullopt;")
                code_lines.append("  }")
                code_lines.append("")
                code_lines.append("  // Skip if reduction is small relative to M/N")
                code_lines.append("  int64_t ratio = kSize / std::sqrt(mSize * nSize) / batchSize;")
                code_lines.append("  if (ratio <= ratioThreshold && kSize < largeKSize) {")
                code_lines.append("    return std::nullopt;")
                code_lines.append("  }")
            else:
                code_lines.append("FOR MATMUL (getMatmulLikeReductionSizes):")
                code_lines.append("-" * 80)
                code_lines.append("")
                code_lines.append("Split reduction helps for all matmul test cases.")
                code_lines.append("Current early return thresholds seem appropriate:")
                code_lines.append("  const int64_t ratioThreshold = 384;")
                code_lines.append("  const int64_t largeKSize = 24576;")
                code_lines.append("  const int64_t largeMNSize = 1024;")
        
        return "\n".join(code_lines)
    
    def generate_cpp_code(self, thresholds: Dict) -> str:
        """Generate C++ code with recommended thresholds"""
        thresh_list = thresholds['thresholds']
        
        if not thresh_list:
            return "  // No thresholds derived\n  int64_t limitParallelLoops = 16;"
        
        code_lines = []
        code_lines.append("  int64_t limitParallelLoops;")
        
        # Final deduplication pass - skip any threshold that's identical to the previous one
        # This ensures no duplicate conditions in the generated code
        prev_threshold_val = None
        first = True
        
        for thresh in thresh_list:
            limit = thresh['limit']
            threshold = thresh['threshold']
            count = thresh['count']
            speedup = thresh['avg_speedup']
            
            if threshold is None:
                # This is the final else case
                continue
            
            # Skip if this threshold is identical to the previous one
            if threshold == prev_threshold_val:
                continue
            
            prev_threshold_val = threshold
            m_n_str = self._format_as_product(threshold)
            
            if first:
                code_lines.append(f"  if (outputSize < {m_n_str}) {{  // {count} tests, avg speedup: {speedup:.2f}x")
                first = False
            else:
                code_lines.append(f"  }} else if (outputSize < {m_n_str}) {{  // {count} tests, avg speedup: {speedup:.2f}x")
            
            code_lines.append(f"    limitParallelLoops = {limit};")
        
        # Find the else case (last cluster with no threshold)
        else_limit = None
        else_count = 0
        else_speedup = 1.0
        for thresh in thresh_list:
            if thresh['threshold'] is None:
                else_limit = thresh['limit']
                else_count = thresh['count']
                else_speedup = thresh['avg_speedup']
                break
        
        # Always use 16 for else clause to align with original code
        # This provides a sensible default fallback regardless of the data
        else_limit = 16
        
        code_lines.append(f"  }} else {{  // {else_count} tests, avg speedup: {else_speedup:.2f}x")
        code_lines.append(f"    limitParallelLoops = std::min<int64_t>({else_limit}, tileSizes[0]);")
        code_lines.append(f"  }}")
        
        return "\n".join(code_lines)
    
    def _format_as_product(self, value: int) -> str:
        """Format value as M*N if possible, strongly preferring perfect squares"""
        if value <= 0:
            return str(value)
        
        import math
        
        # Check if value is a perfect square or very close to one
        sqrt_val = int(math.sqrt(value))
        if sqrt_val * sqrt_val == value:
            return f"{sqrt_val} * {sqrt_val}"
        
        # If within a small absolute difference (10 units) of a perfect square, round to it
        # This handles cases where merging/rounding created values like 7234 instead of 7225 (85^2)
        lower_diff = value - (sqrt_val * sqrt_val)
        upper_diff = ((sqrt_val + 1) * (sqrt_val + 1)) - value
        
        if lower_diff < 10:
            return f"{sqrt_val} * {sqrt_val}"
        elif upper_diff < 10:
            return f"{sqrt_val + 1} * {sqrt_val + 1}"
        
        # Common perfect squares in convolution/matmul dimensions
        preferred_squares = [
            16, 32, 64, 96, 104, 128, 192, 256, 384, 512, 864, 1024, 
            1728, 2048, 2678, 3072, 3456, 4096
        ]
        
        for size in preferred_squares:
            if size * size == value:
                return f"{size} * {size}"
        
        # Common non-square patterns (for backwards compatibility)
        preferred_factors = [
            (3456, 4096), (3072, 4096), (2048, 4096), (1728, 1728),
            (864, 1024), (512, 1024), (256, 512), (128, 256), (64, 128), (32, 64), (16, 32)
        ]
        
        for m, n in preferred_factors:
            if m * n == value:
                return f"{m} * {n}"
        
        # Try to find factors closest to square root
        sqrt_val = int(math.sqrt(value))
        best_factor = None
        min_diff = float('inf')
        
        for i in range(max(1, sqrt_val - 50), sqrt_val + 51):
            if value % i == 0:
                j = value // i
                diff = abs(i - j)
                if diff < min_diff:
                    min_diff = diff
                    best_factor = (min(i, j), max(i, j))
        
        if best_factor:
            return f"{best_factor[0]} * {best_factor[1]}"
        
        return str(value)
    
    def generate_comprehensive_report(self, output_file: Path, baseline_limit: Union[int, str] = None, optimized_results: dict = None):
        """Generate comprehensive analysis report"""
        characteristics = self.extract_characteristics()
        clusters = self.cluster_by_limit(characteristics)
        thresholds = self.derive_thresholds(clusters)
        
        lines = []
        
        # Header
        lines.append("="*100)
        lines.append("LIMITPARALLELLOOPS OPTIMIZATION - COMPREHENSIVE ANALYSIS")
        lines.append("="*100)
        lines.append("")
        
        from datetime import datetime
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Total test cases: {len(characteristics)}")
        lines.append(f"Candidate limits tested: {self.candidate_limits}")
        lines.append("")
        
        # PART 1: Statistical Summary by Limit
        lines.append("="*100)
        lines.append("PART 1: PERFORMANCE SUMMARY BY LIMIT")
        lines.append("="*100)
        lines.append("")
        
        limit_stats = {}
        for limit in self.candidate_limits:
            runtimes = []
            for char in characteristics:
                if char.all_runtimes and limit in char.all_runtimes:
                    runtimes.append(char.all_runtimes[limit])
            
            if runtimes:
                metrics = self.compute_metrics(runtimes)
                limit_stats[limit] = metrics
                
                lines.append(f"limitParallelLoops = {limit}:")
                lines.append(f"  Geometric Mean:   {metrics['geometric_mean']:.2f} ms")
                lines.append(f"  Arithmetic Mean:  {metrics['arithmetic_mean']:.2f} ms")
                lines.append(f"  Median:           {metrics['median']:.2f} ms")
                lines.append(f"  P95:              {metrics['p95']:.2f} ms")
                lines.append(f"  Total Runtime:    {metrics['total_runtime']:.0f} ms")
                lines.append("")
        
        # Ranking by geometric mean (exclude baseline from ranking - it's the reference, not a candidate)
        if limit_stats:
            # Filter out baseline and non-integer limits for ranking
            candidate_limits_only = [(l, m) for l, m in limit_stats.items() 
                                     if isinstance(l, int) and l != baseline_limit]
            ranked_limits = sorted(candidate_limits_only, key=lambda x: x[1]['geometric_mean'])
            
            lines.append("Ranking by Geometric Mean (excluding baseline):")
            for rank, (limit, metrics) in enumerate(ranked_limits, 1):
                best_marker = " ⭐ BEST OVERALL" if rank == 1 else ""
                lines.append(f"  #{rank}: limit={limit} (GeoMean: {metrics['geometric_mean']:.2f}ms){best_marker}")
            lines.append("")
        else:
            ranked_limits = []
        
        # PART 2: Baseline Comparison (if specified)
        if baseline_limit and baseline_limit in self.candidate_limits:
            lines.append("="*100)
            lines.append(f"PART 2: COMPARISON TO BASELINE (limit={baseline_limit})")
            lines.append("="*100)
            lines.append("")
            
            comparison = self.compare_to_baseline(characteristics, baseline_limit)
            baseline_stats = limit_stats.get(baseline_limit, {})
            
            for limit in sorted(comparison.keys()):
                comp = comparison[limit]
                total = comp['faster'] + comp['slower'] + comp['same']
                win_rate = (comp['faster'] / total * 100) if total > 0 else 0
                
                lines.append(f"limit={limit} vs baseline:")
                lines.append(f"  Win Rate:         {win_rate:.1f}% ({comp['faster']}/{total} tests faster)")
                lines.append(f"  Faster:           {comp['faster']} tests")
                lines.append(f"  Slower:           {comp['slower']} tests")
                lines.append(f"  Same (±1%):       {comp['same']} tests")
                
                if comp['improvements']:
                    avg_improvement = statistics.mean(comp['improvements'])
                    max_improvement = max(comp['improvements'])
                    lines.append(f"  Avg Improvement:  {avg_improvement:.1f}%")
                    lines.append(f"  Max Improvement:  {max_improvement:.1f}%")
                
                if comp['regressions']:
                    avg_regression = statistics.mean(comp['regressions'])
                    max_regression = max(comp['regressions'])
                    lines.append(f"  Avg Regression:   {avg_regression:.1f}%")
                    lines.append(f"  Max Regression:   {max_regression:.1f}%")
                
                if limit in limit_stats and baseline_limit in limit_stats:
                    geo_improv = (1 - limit_stats[limit]['geometric_mean'] / 
                                 baseline_stats['geometric_mean']) * 100
                    lines.append(f"  GeoMean vs Base:  {geo_improv:+.2f}%")
                
                lines.append("")
        
        # PART 3: Cluster Analysis
        lines.append("="*100)
        lines.append("PART 3: CLUSTERS BY OPTIMAL LIMITPARALLELLOOPS")
        lines.append("="*100)
        lines.append("")
        
        for limit in sorted(clusters.keys(), reverse=True):
            cluster = clusters[limit]
            stats = self.analyze_cluster(cluster)
            
            lines.append(f"limitParallelLoops = {limit}:")
            lines.append(f"  Test count:       {stats['count']}")
            lines.append(f"  Output size:      {stats['output_size']['min']:,} - {stats['output_size']['max']:,} (median: {stats['output_size']['median']:,})")
            lines.append(f"  K size:           {stats['k_size']['min']:,} - {stats['k_size']['max']:,} (median: {stats['k_size']['median']:,})")
            lines.append(f"  Avg runtime:      {stats['performance']['avg_runtime_ms']:.2f} ms")
            lines.append(f"  Avg speedup:      {stats['performance']['avg_speedup']:.2f}x")
            lines.append("")
            
            # Show a few examples
            if len(cluster) <= 3:
                lines.append("  Examples:")
                for char in cluster:
                    lines.append(f"    • {char.test_name[:70]}...")
                    lines.append(f"      Runtime: {char.best_runtime:.2f}ms, Speedup: {char.speedup:.2f}x")
                lines.append("")
        
        # PART 4: Threshold Recommendations
        lines.append("="*100)
        lines.append("PART 4: RECOMMENDED THRESHOLDS FOR C++ CODE")
        lines.append("="*100)
        lines.append("")
        
        for thresh in thresholds['thresholds']:
            lines.append(f"limitParallelLoops = {thresh['limit']}:")
            if thresh['threshold'] is not None:
                lines.append(f"  When outputSize < {thresh['threshold']:,}")
            else:
                lines.append(f"  For all larger output sizes (else clause)")
            lines.append(f"  ({thresh['count']} tests, avg speedup: {thresh['avg_speedup']:.2f}x)")
            lines.append("")
        
        lines.append("="*100)
        lines.append("RECOMMENDED C++ CODE")
        lines.append("="*100)
        lines.append("")
        
        # Analyze early return candidates (where limit=1 is best)
        early_return_analysis = self.analyze_early_return_candidates(characteristics)
        
        # Detect dataset type
        dataset_type = self.detect_dataset_type(characteristics)
        
        lines.append("PART A: Early Return Thresholds")
        lines.append("-" * 100)
        lines.append("")
        
        # Check if any operation type has cases where split doesn't help
        conv_count = early_return_analysis.get('conv', {}).get('count', 0)
        matmul_count = early_return_analysis.get('matmul', {}).get('count', 0)
        total_no_split = conv_count + matmul_count
        
        if total_no_split > 0:
            lines.append(f"Found {total_no_split} test cases where split reduction doesn't help:")
            if conv_count > 0:
                conv_pct = early_return_analysis['conv']['percentage']
                lines.append(f"  - {conv_count} convolution tests ({conv_pct:.1f}%)")
            if matmul_count > 0:
                matmul_pct = early_return_analysis['matmul']['percentage']
                lines.append(f"  - {matmul_count} matmul tests ({matmul_pct:.1f}%)")
            lines.append("")
            
            # Show characteristics for each operation type
            if conv_count > 0:
                conv_analysis = early_return_analysis['conv']
                lines.append("Convolution cases where NO SPLIT is better:")
                lines.append(f"  Output Size: {conv_analysis['output_size']['min']:,} - {conv_analysis['output_size']['max']:,} (median: {conv_analysis['output_size']['median']:,})")
                lines.append(f"  K Size:      {conv_analysis['k_size']['min']:,} - {conv_analysis['k_size']['max']:,} (median: {conv_analysis['k_size']['median']:,})")
                lines.append(f"  Ratio:       {conv_analysis['ratio']['min']:.2f} - {conv_analysis['ratio']['max']:.2f} (median: {conv_analysis['ratio']['median']:.2f})")
                lines.append("")
            
            if matmul_count > 0:
                matmul_analysis = early_return_analysis['matmul']
                lines.append("Matmul cases where NO SPLIT is better:")
                lines.append(f"  Output Size: {matmul_analysis['output_size']['min']:,} - {matmul_analysis['output_size']['max']:,} (median: {matmul_analysis['output_size']['median']:,})")
                lines.append(f"  K Size:      {matmul_analysis['k_size']['min']:,} - {matmul_analysis['k_size']['max']:,} (median: {matmul_analysis['k_size']['median']:,})")
                lines.append(f"  Ratio:       {matmul_analysis['ratio']['min']:.2f} - {matmul_analysis['ratio']['max']:.2f} (median: {matmul_analysis['ratio']['median']:.2f})")
                lines.append("")
            
            lines.append("Recommended early return code:")
            lines.append("")
            early_return_code = self.generate_early_return_code(early_return_analysis, dataset_type)
            lines.append(early_return_code)
        else:
            lines.append("Split reduction helps for ALL test cases!")
            lines.append("No early returns needed - split reduction should always be attempted.")
            lines.append("")
            lines.append("Current early return thresholds in the code seem appropriate:")
            lines.append("")
            
            # Only show relevant sections based on dataset type
            if dataset_type in ['conv-only', 'mixed']:
                lines.append("FOR CONVOLUTION:")
                lines.append("  const int64_t largeParallelSize = 512;")
                lines.append("  const int64_t largeReductionSize = 8192;")
                lines.append("  const int64_t ratioThreshold = 64;")
                if dataset_type == 'mixed':
                    lines.append("")
            
            if dataset_type in ['matmul-only', 'mixed']:
                lines.append("FOR MATMUL:")
                lines.append("  const int64_t ratioThreshold = 384;")
                lines.append("  const int64_t largeKSize = 24576;")
                lines.append("  const int64_t largeMNSize = 1024;")
        lines.append("")
        lines.append("")
        
        lines.append("PART B: limitParallelLoops Logic")
        lines.append("-" * 100)
        lines.append("")
        
        # Add clarification about which operation type these recommendations are for
        if dataset_type == 'conv-only':
            lines.append("For CONVOLUTION operations (getWeightBackwardReductionSizes):")
        elif dataset_type == 'matmul-only':
            lines.append("For MATMUL operations (getMatmulLikeReductionSizes):")
        else:  # mixed
            lines.append("For BOTH convolution and matmul operations:")
            lines.append("(These thresholds apply to the limitParallelLoops logic in both functions)")
        lines.append("")
        lines.append("Replace the limitParallelLoops logic with:")
        lines.append("")
        cpp_code = self.generate_cpp_code(thresholds)
        lines.append(cpp_code)
        lines.append("")
        
        # PART 5: Top Performers
        lines.append("="*100)
        lines.append("PART 5: TOP 20 HIGHEST SPEEDUPS")
        lines.append("="*100)
        lines.append("")
        
        sorted_chars = sorted(characteristics, key=lambda c: c.speedup, reverse=True)[:20]
        for i, char in enumerate(sorted_chars, 1):
            lines.append(f"#{i}: {char.test_name[:70]}...")
            lines.append(f"     Best limit: {char.best_limit}, Runtime: {char.best_runtime:.2f}ms, Speedup: {char.speedup:.2f}x")
            lines.append(f"     OutputSize: {char.output_size:,}, KSize: {char.k_size:,}")
            lines.append("")
        
        # PART 6: Optimized Configuration Performance (if provided)
        if optimized_results:
            lines.append("="*100)
            lines.append("PART 6: OPTIMIZED CONFIGURATION - VALIDATED PERFORMANCE")
            lines.append("="*100)
            lines.append("")
            lines.append("After applying the recommendations above, the optimized configuration was")
            lines.append("tested on the same dataset. Results:")
            lines.append("")
            
            # Get baseline results
            baseline_results = {}
            for char in characteristics:
                if char.all_runtimes and baseline_limit in char.all_runtimes:
                    baseline_results[char.test_name] = char.all_runtimes[baseline_limit]
            
            # Compare optimized vs baseline
            if baseline_results:
                improvements = []
                regressions = []
                neutral = []
                total_baseline = 0
                total_optimized = 0
                geomean_ratios = []
                
                for test_name, optimized_time in optimized_results.items():
                    if test_name in baseline_results:
                        baseline_time = baseline_results[test_name]
                        total_baseline += baseline_time
                        total_optimized += optimized_time
                        
                        speedup = baseline_time / optimized_time
                        improvement_pct = ((baseline_time - optimized_time) / baseline_time) * 100
                        geomean_ratios.append(speedup)
                        
                        if speedup > 1.05:  # >5% improvement
                            improvements.append((test_name, speedup, improvement_pct, baseline_time, optimized_time))
                        elif speedup < 0.95:  # >5% regression
                            regressions.append((test_name, speedup, improvement_pct, baseline_time, optimized_time))
                        else:
                            neutral.append((test_name, speedup, improvement_pct, baseline_time, optimized_time))
                
                improvements.sort(key=lambda x: x[1], reverse=True)
                regressions.sort(key=lambda x: x[1])
                
                # Overall metrics
                total_tests = len(improvements) + len(regressions) + len(neutral)
                overall_speedup = total_baseline / total_optimized if total_optimized > 0 else 0
                overall_improvement = ((total_baseline - total_optimized) / total_baseline) * 100 if total_baseline > 0 else 0
                
                # Geometric mean speedup
                import math
                if geomean_ratios:
                    geomean_speedup = math.exp(sum(math.log(max(r, 0.001)) for r in geomean_ratios) / len(geomean_ratios))
                else:
                    geomean_speedup = 1.0
                
                lines.append("┌─────────────────────────────────────────────────────────────────────────┐")
                lines.append("│                    BASELINE vs OPTIMIZED                                 │")
                lines.append("├─────────────────────────────────────────────────────────────────────────┤")
                lines.append(f"│ Total Tests:          {total_tests:>10}                                       │")
                lines.append(f"│ Baseline Runtime:     {total_baseline:>10.0f} ms  (~{total_baseline/60000:.1f} min)                  │")
                lines.append(f"│ Optimized Runtime:    {total_optimized:>10.0f} ms  (~{total_optimized/60000:.1f} min)                  │")
                lines.append(f"│ Overall Speedup:      {overall_speedup:>10.2f}x                                      │")
                lines.append(f"│ Overall Improvement:  {overall_improvement:>10.2f}%                                      │")
                lines.append(f"│ Geometric Mean:       {geomean_speedup:>10.2f}x                                      │")
                lines.append("├─────────────────────────────────────────────────────────────────────────┤")
                lines.append(f"│ Tests Improved:       {len(improvements):>10} ({len(improvements)/total_tests*100:>5.1f}%)                            │")
                lines.append(f"│ Tests Neutral (±5%):  {len(neutral):>10} ({len(neutral)/total_tests*100:>5.1f}%)                            │")
                lines.append(f"│ Tests Regressed:      {len(regressions):>10} ({len(regressions)/total_tests*100:>5.1f}%)                            │")
                lines.append("└─────────────────────────────────────────────────────────────────────────┘")
                lines.append("")
                
                # Top improvements
                if improvements:
                    lines.append(f"Top 15 Improvements:")
                    lines.append("")
                    for i, (test, speedup, improvement, base, opt) in enumerate(improvements[:15], 1):
                        lines.append(f"{i:2d}. {speedup:6.2f}x ({improvement:+6.2f}%) | Baseline: {base:8.0f}ms → Optimized: {opt:8.0f}ms")
                        lines.append(f"    {test[:75]}")
                        if i < 15 and i < len(improvements):
                            lines.append("")
                    lines.append("")
                
                # Regressions (if any)
                if regressions:
                    lines.append(f"Regressions (>5% slower):")
                    lines.append("")
                    for i, (test, speedup, improvement, base, opt) in enumerate(regressions, 1):
                        lines.append(f"{i:2d}. {speedup:6.2f}x ({improvement:+6.2f}%) | Baseline: {base:8.0f}ms → Optimized: {opt:8.0f}ms")
                        lines.append(f"    {test[:75]}")
                        lines.append("")
                
                # Summary
                lines.append("="*100)
                lines.append("OVERALL RECOMMENDATION")
                lines.append("="*100)
                lines.append("")
                
                if len(regressions) == 0:
                    lines.append("✅ EXCELLENT RESULTS - READY FOR PRODUCTION")
                    lines.append("")
                    lines.append(f"The optimized configuration shows {overall_speedup:.2f}x overall speedup with")
                    lines.append("ZERO regressions. This is a safe, high-impact optimization.")
                elif len(regressions) / total_tests < 0.05:
                    lines.append("✅ STRONG RESULTS - RECOMMENDED FOR PRODUCTION")
                    lines.append("")
                    lines.append(f"The optimized configuration shows {overall_speedup:.2f}x overall speedup with")
                    lines.append(f"only {len(regressions)} regressions ({len(regressions)/total_tests*100:.1f}% of tests).")
                else:
                    lines.append("⚠️  MIXED RESULTS - REVIEW RECOMMENDED")
                    lines.append("")
                    lines.append(f"The optimized configuration shows {overall_speedup:.2f}x overall speedup, but")
                    lines.append(f"{len(regressions)} tests regressed ({len(regressions)/total_tests*100:.1f}%). Consider further tuning.")
                
                lines.append("")
            else:
                lines.append("⚠️  Could not compare: baseline results not found")
                lines.append("")
        
        # Write report
        report_text = "\n".join(lines)
        with open(output_file, 'w') as f:
            f.write(report_text)
        
        print(f"✓ Comprehensive analysis saved to: {output_file}")


def parse_baseline_limit(value):
    """Parse baseline limit - can be 'baseline' or an integer"""
    if value.lower() == 'baseline':
        return 'baseline'
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"baseline-limit must be 'baseline' or an integer, got: {value}")


def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive analysis of limitParallelLoops sweep results'
    )
    parser.add_argument('--results-file', required=True,
                       help='Path to limitParallelLoops_sweep_results.json')
    parser.add_argument('--output-dir', help='Directory to save analysis')
    parser.add_argument('--baseline-limit', type=parse_baseline_limit,
                       help='Baseline limit for comparison (e.g., 64 or "baseline")')
    parser.add_argument('--optimized-csv', 
                       help='Path pattern to optimized configuration CSV files (e.g., "results/optimized_config_*.csv")')
    
    args = parser.parse_args()
    
    results_file = Path(args.results_file)
    if not results_file.exists():
        print(f"Error: Results file not found: {results_file}")
        return 1
    
    output_dir = Path(args.output_dir) if args.output_dir else results_file.parent
    output_file = output_dir / 'comprehensive_analysis.txt'
    
    # Load optimized results if provided
    optimized_results = None
    if args.optimized_csv:
        import glob
        import csv
        
        optimized_results = {}
        csv_files = glob.glob(args.optimized_csv)
        
        if csv_files:
            print(f"Loading optimized results from {len(csv_files)} file(s)...")
            for csv_file in csv_files:
                try:
                    with open(csv_file, 'r') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            test_name = row['arguments']
                            runtime = float(row['iree_boo_experimental mean'])
                            optimized_results[test_name] = runtime
                except Exception as e:
                    print(f"Warning: Failed to load {csv_file}: {e}")
            
            if optimized_results:
                print(f"✓ Loaded {len(optimized_results)} optimized test results")
    
    analyzer = ComprehensiveAnalyzer(results_file)
    analyzer.generate_comprehensive_report(output_file, args.baseline_limit, optimized_results)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
