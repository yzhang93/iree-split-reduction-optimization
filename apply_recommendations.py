#!/usr/bin/env python3
"""
Apply C++ recommendations from comprehensive_analysis.txt to
SetSplitReductionSizes.cpp.

The recommendations in comprehensive_analysis.txt PART A/B are specifically
for the CONVOLUTION heuristic (historically named
``getWeightBackwardReductionSizes``, now ``getConvolutionReductionSizes``).
We therefore apply the substitutions *only* inside that function's body so
the matmul-like heuristic in the same file is left untouched.

We also suppress local variables that the original code declared for use in
the default branch of the replaced if-else chain (``startTileSize``), so the
file still builds cleanly under ``-Werror,-Wunused-variable``.
"""

import re
import sys
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple


CONV_FUNCTION_NAMES = (
    "getConvolutionReductionSizes",
    "getWeightBackwardReductionSizes",
)

# Locals that exist in the conv function *before* ``int64_t limitParallelLoops;``
# but are only referenced inside the default branch of the original if-else
# chain. If the new logic does not reference them, we insert ``(void)<name>;``
# after the declaration so ``-Wunused-variable`` doesn't trip.
CONV_SUPPRESSION_CANDIDATES = ("startTileSize",)


def extract_recommendations(analysis_file: Path) -> Dict:
    """Extract recommended constants and logic from the analysis text."""

    with open(analysis_file, "r") as f:
        content = f.read()

    recommendations: Dict = {"constants": {}, "limit_logic": None}

    const_pattern = (
        r"const int64_t "
        r"(largeParallelSize|largeReductionSize|ratioThreshold|"
        r"largeKSize|largeMNSize|largeOutputSize)"
        r"\s*=\s*([\d\s*+\-]+?);"
    )
    for match in re.finditer(const_pattern, content):
        var_name = match.group(1)
        value = match.group(2).strip()
        recommendations["constants"][var_name] = value
        print(f"  Found: {var_name} = {value}")

    # Extract limitParallelLoops logic from PART B.
    part4b_idx = content.find("PART 4B:")
    if part4b_idx == -1:
        part4b_idx = content.find("PART B:")
    if part4b_idx == -1:
        part4b_idx = content.find("limitParallelLoops Logic")

    if part4b_idx != -1:
        logic_start = content.find("if (outputSize <", part4b_idx)
        if logic_start == -1:
            logic_start = content.find("if (outputSize <=", part4b_idx)
        if logic_start != -1:
            logic_end = _find_ifelse_end(content, logic_start)
            if logic_end:
                recommendations["limit_logic"] = content[
                    logic_start:logic_end
                ].strip()
                print(
                    f"\u2713 Extracted limitParallelLoops logic "
                    f"({len(recommendations['limit_logic'])} chars)"
                )

    return recommendations


def _find_ifelse_end(content: str, start: int) -> Optional[int]:
    """Return index past the closing brace of an if / else-if / else chain
    beginning at ``start``."""
    brace_count = 0
    block_started = False
    i = start
    while i < len(content):
        ch = content[i]
        if ch == "{":
            brace_count += 1
            block_started = True
        elif ch == "}":
            brace_count -= 1
            if block_started and brace_count == 0:
                end = i + 1
                j = i + 1
                while j < len(content) and content[j] in " \t\n":
                    j += 1
                if content[j : j + 4] == "else":
                    # Continue consuming the else / else-if branch.
                    i = j
                    continue
                return end
        i += 1
    return None


def _find_function_body(content: str, names: Tuple[str, ...]) -> Optional[Tuple[int, int]]:
    """Return ``(body_start, body_end)`` for the first function in ``names``
    found in ``content``, where ``body_start`` is the index *after* the opening
    ``{`` of the body and ``body_end`` is the index of the matching ``}``.
    Returns ``None`` if no function is found.
    """
    for name in names:
        # Match ``name(...) ... {`` allowing the signature to span lines.
        sig_match = re.search(rf"\b{re.escape(name)}\s*\(", content)
        if not sig_match:
            continue
        # Scan forward from the signature to the first ``{`` that opens the
        # body. Paren-balance so we don't stop inside the parameter list.
        i = sig_match.end()
        paren = 1
        while i < len(content) and paren > 0:
            if content[i] == "(":
                paren += 1
            elif content[i] == ")":
                paren -= 1
            i += 1
        # Skip whitespace and any trailing return-type tokens up to ``{``.
        while i < len(content) and content[i] != "{":
            i += 1
        if i >= len(content):
            continue
        body_start = i + 1
        depth = 1
        j = body_start
        while j < len(content) and depth > 0:
            if content[j] == "{":
                depth += 1
            elif content[j] == "}":
                depth -= 1
                if depth == 0:
                    return body_start, j
            j += 1
    return None


def _apply_within(
    content: str,
    span: Tuple[int, int],
    transform,
) -> Tuple[str, bool]:
    """Apply ``transform(body) -> (new_body, modified)`` to ``content[span]``
    and return the (potentially) updated full content plus a modified flag.
    """
    start, end = span
    body = content[start:end]
    new_body, modified = transform(body)
    if modified and new_body != body:
        return content[:start] + new_body + content[end:], True
    return content, False


def _update_constants(body: str, constants: Dict[str, str]) -> Tuple[str, bool]:
    modified = False
    for var_name, new_value in constants.items():
        pattern = rf"(const int64_t {var_name}\s*=\s*)([\d\s*+\-]+?)(;)"
        m = re.search(pattern, body)
        if not m:
            continue
        old_value = m.group(2).strip()
        if old_value == new_value:
            continue
        body = re.sub(pattern, rf"\g<1>{new_value}\g<3>", body, count=1)
        print(f"  Updated {var_name}: {old_value} \u2192 {new_value}")
        modified = True
    return body, modified


def _update_limit_logic(body: str, new_logic: str) -> Tuple[str, bool]:
    """Replace the first ``int64_t limitParallelLoops;`` if-else chain inside
    ``body`` with ``new_logic``, preserving the declaration and indentation.
    Also inserts ``(void)`` suppressions for locals that the new logic does
    not reference.
    """
    decl_re = re.compile(r"([ \t]*)int64_t limitParallelLoops;\s*\n")
    decl_match = decl_re.search(body)
    if not decl_match:
        return body, False

    indent = decl_match.group(1)
    after_decl = decl_match.end()

    # Skip whitespace / line-comments / block-comments between the declaration
    # and the start of the ``if`` statement.
    i = after_decl
    while i < len(body):
        # Skip whitespace.
        while i < len(body) and body[i] in " \t\n":
            i += 1
        if body.startswith("//", i):
            nl = body.find("\n", i)
            if nl == -1:
                return body, False
            i = nl + 1
            continue
        if body.startswith("/*", i):
            end = body.find("*/", i)
            if end == -1:
                return body, False
            i = end + 2
            continue
        break

    if not body.startswith("if", i):
        return body, False

    chain_end = _find_ifelse_end(body, i)
    if chain_end is None:
        return body, False

    # Build replacement text, indenting each line of new_logic to match.
    new_lines = [
        f"{indent}// OPTIMIZED: Data-driven thresholds from sweep analysis",
    ]
    # Suppress locals from the old chain that the new logic no longer uses.
    for name in CONV_SUPPRESSION_CANDIDATES:
        if not re.search(rf"\b{name}\b", new_logic):
            # Only add the suppression if the variable is actually declared
            # earlier in the body (avoids suppressing an unrelated variable).
            if re.search(rf"\bint64_t\s+{name}\b", body[:decl_match.start()]):
                new_lines.append(
                    f"{indent}(void){name};  // OPTIMIZER: no longer used by recommended logic"
                )
    for raw in new_logic.splitlines():
        stripped = raw.lstrip()
        if not stripped:
            new_lines.append("")
            continue
        # Heuristically re-indent: if the original line started with 2 spaces,
        # treat that as the "body indent" and replace it with ``indent``.
        leading = len(raw) - len(stripped)
        # Normalize leading to a multiple of 2 spaces relative to the first
        # line (which should be the top-level ``if``).
        extra = max(leading - 2, 0)
        new_lines.append(f"{indent}{' ' * extra}{stripped}")

    replacement = "\n".join(new_lines) + "\n"

    new_body = body[: decl_match.end()] + replacement + body[chain_end:]
    # Collapse accidental double newlines after the replacement.
    new_body = re.sub(r"\n{3,}", "\n\n", new_body)
    print("  \u2713 Updated limitParallelLoops logic (conv function)")
    return new_body, True


def apply_to_cpp(cpp_file: Path, recommendations: Dict) -> bool:
    if not cpp_file.exists():
        print(f"Error: C++ file not found: {cpp_file}")
        return False

    backup_file = cpp_file.parent / f"{cpp_file.name}.before_optimization"
    if not backup_file.exists() or backup_file.read_bytes() != cpp_file.read_bytes():
        shutil.copy(cpp_file, backup_file)
        print(f"\u2713 Created backup: {backup_file}")
    else:
        print(f"\u2713 Backup already up to date: {backup_file}")

    with open(cpp_file, "r") as f:
        content = f.read()

    span = _find_function_body(content, CONV_FUNCTION_NAMES)
    if span is None:
        print(
            "\u274c Could not locate convolution function "
            f"({' / '.join(CONV_FUNCTION_NAMES)}) in {cpp_file}; aborting."
        )
        return False

    modified_any = False

    if recommendations["constants"]:
        content, changed = _apply_within(
            content,
            span,
            lambda body: _update_constants(body, recommendations["constants"]),
        )
        modified_any = modified_any or changed
        if changed:
            span = _find_function_body(content, CONV_FUNCTION_NAMES)  # re-scope

    if recommendations["limit_logic"] and span is not None:
        content, changed = _apply_within(
            content,
            span,
            lambda body: _update_limit_logic(body, recommendations["limit_logic"]),
        )
        modified_any = modified_any or changed

    if modified_any:
        with open(cpp_file, "w") as f:
            f.write(content)
        print(f"\u2713 Applied recommendations to {cpp_file}")
        return True
    print("\u2139 No changes needed (already optimal or no recommendations found)")
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

    if not recommendations["constants"] and not recommendations["limit_logic"]:
        print("\u26a0 No recommendations found in analysis file")
        sys.exit(1)

    print(f"\nApplying to {cpp_file}...")
    success = apply_to_cpp(cpp_file, recommendations)

    if success:
        print("\n\u2705 Recommendations applied successfully!")
        print(f"   Backup saved to: {cpp_file}.before_optimization")
        sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    main()
