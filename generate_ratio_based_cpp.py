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


def parse_args_for_dimensions(args_str: str) -> Tuple[int, int, int]:
    """Extract m, n, k from argument string"""
    import re
    
    # Weight backward convolution pattern
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
        
        return output_size, 1, k_size
    
    # Matmul pattern
    mm_match = re.search(r'-n (\d+).*-c (\d+).*-k (\d+).*-y 1 -x 1', args_str)
    if mm_match:
        n, c, k = [int(mm_match.group(i)) for i in range(1, 4)]
        # Matmul: m=n*c, n=k, k=c (batch dimension)
        return n * c, k, c
    
    return 0, 0, 0


def load_and_analyze(results_file: Path) -> List[TestData]:
    """Load results and compute ratio for each test"""
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    test_data = []
    for test_name, analysis in data['analyses'].items():
        best_limit = analysis.get('best_limit')
        
        # Skip baseline
        if isinstance(best_limit, str):
            continue
        
        # Parse dimensions
        m, n, k = parse_args_for_dimensions(test_name)
        if not (m and k):
            continue
        
        output_size = m * n if n else m
        k_size = k
        
        # Compute ratio = kSize / sqrt(outputSize)
        ratio = k_size / math.sqrt(output_size) if output_size > 0 else 0
        
        test_data.append(TestData(
            test_name=test_name,
            best_limit=best_limit,
            best_runtime=analysis['best_runtime'],
            output_size=output_size,
            k_size=k_size,
            ratio=ratio
        ))
    
    return test_data


def cluster_by_ratio(test_data: List[TestData]) -> Dict[int, List[TestData]]:
    """Group tests by their optimal limitParallelLoops"""
    clusters = {}
    for test in test_data:
        limit = test.best_limit
        if limit not in clusters:
            clusters[limit] = []
        clusters[limit].append(test)
    return clusters


def derive_ratio_thresholds(clusters: Dict[int, List[TestData]]) -> List[Tuple[int, float]]:
    """Derive ratio thresholds that separate different limits
    
    Returns list of (limit, min_ratio_threshold) sorted by threshold descending
    """
    # Calculate median ratio for each limit
    limit_stats = []
    for limit, tests in sorted(clusters.items(), reverse=True):
        ratios = [t.ratio for t in tests]
        median_ratio = sorted(ratios)[len(ratios)//2]
        min_ratio = min(ratios)
        max_ratio = max(ratios)
        
        limit_stats.append({
            'limit': limit,
            'median_ratio': median_ratio,
            'min_ratio': min_ratio,
            'max_ratio': max_ratio,
            'count': len(tests)
        })
    
    # Sort by median ratio (descending - higher ratio means more aggressive splitting)
    limit_stats.sort(key=lambda x: x['median_ratio'], reverse=True)
    
    # Derive thresholds at boundaries
    thresholds = []
    for i, stats in enumerate(limit_stats):
        if i < len(limit_stats) - 1:
            # Threshold is midpoint between this cluster's min and next cluster's max
            next_stats = limit_stats[i + 1]
            threshold = (stats['min_ratio'] + next_stats['max_ratio']) / 2
        else:
            # Last cluster (smallest ratio) - use a very small threshold
            threshold = 0.1
        
        thresholds.append((stats['limit'], threshold))
    
    return thresholds


def generate_cpp_code(thresholds: List[Tuple[int, float]], clusters: Dict[int, List[TestData]]) -> str:
    """Generate C++ code using ratio-based decision tree"""
    
    lines = []
    lines.append("// ============================================================================")
    lines.append("// UNIFIED RATIO-BASED SPLIT REDUCTION")
    lines.append("// ============================================================================")
    lines.append("//")
    lines.append("// This approach uses a SINGLE decision tree based on:")
    lines.append("//   ratio = reductionSize / sqrt(outputSize)")
    lines.append("//")
    lines.append("// Benefits:")
    lines.append("//   ✓ Combines both parallel and reduction dimensions naturally")
    lines.append("//   ✓ No separate early return mechanism")
    lines.append("//   ✓ limitParallelLoops=1 explicitly means 'no split'")
    lines.append("//   ✓ Monotonic: higher ratio → more aggressive splitting")
    lines.append("//")
    lines.append("// Cluster statistics:")
    for limit in sorted(clusters.keys(), reverse=True):
        tests = clusters[limit]
        ratios = [t.ratio for t in tests]
        print(f"//   limit={limit}: {len(tests)} tests, ratio range: {min(ratios):.2f}-{max(ratios):.2f}")
        lines.append(f"//   limit={limit}: {len(tests)} tests, ratio range: {min(ratios):.2f}-{max(ratios):.2f}")
    lines.append("//")
    lines.append("")
    
    lines.append("// Calculate problem characteristics")
    lines.append("int64_t outputSize = outputChannelSize * batchSize * imageSize * depthSize;")
    lines.append("SmallVector<int64_t> tileSizes = std::move(*maybeSizes);")
    lines.append("int64_t reductionSize = llvm::product_of(tileSizes);")
    lines.append("")
    lines.append("// Compute ratio (reduction vs parallel balance)")
    lines.append("double ratio = (double)reductionSize / std::sqrt((double)outputSize);")
    lines.append("")
    lines.append("// Unified decision tree based on ratio")
    lines.append("int64_t limitParallelLoops;")
    
    # Generate if-else chain sorted by threshold (descending)
    for i, (limit, threshold) in enumerate(sorted(thresholds, key=lambda x: x[1], reverse=True)):
        if i == 0:
            lines.append(f"if (ratio > {threshold:.2f}) {{")
        elif i == len(thresholds) - 1:
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
    lines.append("")
    
    lines.append("// Apply split reduction for other limits")
    lines.append("// (existing splitting logic follows...)")
    lines.append("for (int64_t i = 0; i < tileSizes.size(); i++) {")
    lines.append("  int64_t lowerBound = llvm::divideCeil(tileSizes[i], limitParallelLoops);")
    lines.append("  std::optional<int64_t> maybeTileSize =")
    lines.append("      findSmallestFactorWithLowerBound(tileSizes[i], lowerBound);")
    lines.append("  if (!maybeTileSize) {")
    lines.append("    LDBG() << \"skipping op; failed to find a split factor\";")
    lines.append("    return std::nullopt;")
    lines.append("  }")
    lines.append("  limitParallelLoops /= (tileSizes[i] / maybeTileSize.value());")
    lines.append("  tileSizes[i] = maybeTileSize.value();")
    lines.append("  if (maybeTileSize.value() > 1) {")
    lines.append("    break;")
    lines.append("  }")
    lines.append("}")
    lines.append("")
    lines.append("return tileSizes;")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Generate ratio-based C++ code for split reduction'
    )
    parser.add_argument('--results-file', required=True, type=Path,
                       help='Path to limitParallelLoops_sweep_results.json')
    parser.add_argument('--output', type=Path,
                       help='Output file for generated C++ code (default: print to stdout)')
    
    args = parser.parse_args()
    
    if not args.results_file.exists():
        print(f"Error: Results file not found: {args.results_file}")
        return 1
    
    # Load and analyze data
    print("Loading results and computing ratios...", file=sys.stderr)
    test_data = load_and_analyze(args.results_file)
    print(f"Loaded {len(test_data)} test cases", file=sys.stderr)
    
    # Cluster by optimal limit
    print("Clustering by optimal limitParallelLoops...", file=sys.stderr)
    clusters = cluster_by_ratio(test_data)
    
    # Derive ratio thresholds
    print("Deriving ratio-based thresholds...", file=sys.stderr)
    thresholds = derive_ratio_thresholds(clusters)
    
    print("\nDerived thresholds:", file=sys.stderr)
    for limit, threshold in sorted(thresholds, key=lambda x: x[1], reverse=True):
        print(f"  limit={limit}: ratio > {threshold:.2f}", file=sys.stderr)
    
    # Generate C++ code
    print("\nGenerating C++ code...\n", file=sys.stderr)
    cpp_code = generate_cpp_code(thresholds, clusters)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(cpp_code)
        print(f"✓ C++ code saved to: {args.output}", file=sys.stderr)
    else:
        print(cpp_code)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
