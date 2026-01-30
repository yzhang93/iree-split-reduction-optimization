#!/usr/bin/env python3
"""
Apply C++ recommendations from comprehensive_analysis.txt to SetSplitReductionSizes.cpp
"""

import re
import sys
import shutil
from pathlib import Path
from typing import Dict, Optional


def extract_recommendations(analysis_file: Path) -> Dict:
    """Extract recommended constants and logic from analysis file"""
    
    with open(analysis_file, 'r') as f:
        content = f.read()
    
    recommendations = {
        'constants': {},
        'limit_logic': None
    }
    
    # Extract early return constants
    const_pattern = r'const int64_t (largeParallelSize|largeReductionSize|ratioThreshold|largeKSize|largeMNSize)\s*=\s*(\d+);'
    for match in re.finditer(const_pattern, content):
        var_name = match.group(1)
        value = match.group(2)
        recommendations['constants'][var_name] = value
        print(f"  Found: {var_name} = {value}")
    
    # Extract limitParallelLoops logic from PART 4B
    # Look for the code block between "int64_t limitParallelLoops;" and the closing brace
    part4b_idx = content.find("PART 4B:")
    if part4b_idx == -1:
        part4b_idx = content.find("limitParallelLoops Logic")
    
    if part4b_idx != -1:
        # Find the start of the if-else logic
        logic_start = content.find("if (outputSize <", part4b_idx)
        if logic_start != -1:
            # Find the matching closing brace for the entire if-else chain
            # Count braces to find the end
            brace_count = 0
            i = logic_start
            start_found = False
            while i < len(content):
                if content[i] == '{':
                    brace_count += 1
                    start_found = True
                elif content[i] == '}':
                    brace_count -= 1
                    if start_found and brace_count == 0:
                        # This is the end of the if-else chain
                        logic_end = i + 1
                        recommendations['limit_logic'] = content[logic_start:logic_end].strip()
                        print(f"✓ Extracted limitParallelLoops logic ({len(recommendations['limit_logic'])} chars)")
                        break
                i += 1
    
    return recommendations


def apply_to_cpp(cpp_file: Path, recommendations: Dict) -> bool:
    """Apply recommendations to C++ file"""
    
    if not cpp_file.exists():
        print(f"Error: C++ file not found: {cpp_file}")
        return False
    
    # Create backup
    backup_file = cpp_file.parent / f"{cpp_file.name}.before_optimization"
    shutil.copy(cpp_file, backup_file)
    print(f"✓ Created backup: {backup_file}")
    
    with open(cpp_file, 'r') as f:
        content = f.read()
    
    modified = False
    
    # Update early return constants
    for var_name, new_value in recommendations['constants'].items():
        pattern = rf'(const int64_t {var_name}\s*=\s*)(\d+)(;)'
        match = re.search(pattern, content)
        if match:
            old_value = match.group(2)
            if old_value != new_value:
                content = re.sub(pattern, rf'\g<1>{new_value}\g<3>', content)
                print(f"  Updated {var_name}: {old_value} → {new_value}")
                modified = True
    
    # Update limitParallelLoops logic
    if recommendations['limit_logic']:
        # Find the function with limitParallelLoops logic
        # Look for pattern: int64_t limitParallelLoops; followed by if-else
        pattern = r'(int64_t limitParallelLoops;\s*(?://[^\n]*)?\s*(?:/\*.*?\*/)?\s*)(if\s*\((?:outputSize|kSize)[^}]+\}(?:\s*else\s+if[^}]+\})*\s*else\s*\{[^}]+\})'
        
        def replace_logic(match):
            nonlocal modified
            prefix = match.group(1)
            # Add comment before the new logic
            new_code = f"{prefix}// OPTIMIZED: Data-driven thresholds from sweep analysis\n  {recommendations['limit_logic']}"
            modified = True
            print("  ✓ Updated limitParallelLoops logic")
            return new_code
        
        content = re.sub(pattern, replace_logic, content, flags=re.DOTALL)
    
    if modified:
        with open(cpp_file, 'w') as f:
            f.write(content)
        print(f"✓ Applied recommendations to {cpp_file}")
        return True
    else:
        print("ℹ No changes needed (already optimal or no recommendations found)")
        return False


def main():
    if len(sys.argv) < 3:
        print("Usage: apply_recommendations.py <analysis_file> <cpp_file>")
        sys.exit(1)
    
    analysis_file = Path(sys.argv[1])
    cpp_file = Path(sys.argv[2])
    
    if not analysis_file.exists():
        print(f"Error: Analysis file not found: {analysis_file}")
        sys.exit(1)
    
    print("Extracting recommendations from analysis...")
    recommendations = extract_recommendations(analysis_file)
    
    if not recommendations['constants'] and not recommendations['limit_logic']:
        print("⚠ No recommendations found in analysis file")
        sys.exit(1)
    
    print(f"\nApplying to {cpp_file}...")
    success = apply_to_cpp(cpp_file, recommendations)
    
    if success:
        print("\n✅ Recommendations applied successfully!")
        print(f"   Backup saved to: {cpp_file}.before_optimization")
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()
