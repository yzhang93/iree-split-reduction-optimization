#!/bin/bash
#
# Create a small subset of test cases for quick testing
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Parse arguments
INPUT_FILE="${1:-../prod_weight_shapes.txt}"
OUTPUT_FILE="${2:-../small_test_shapes.txt}"
NUM_LINES="${3:-20}"

if [[ ! -f "$INPUT_FILE" ]]; then
    echo "Error: Input file not found: $INPUT_FILE"
    echo ""
    echo "Usage: $0 [INPUT_FILE] [OUTPUT_FILE] [NUM_LINES]"
    echo ""
    echo "Examples:"
    echo "  $0                                                    # Create 20-line subset from prod"
    echo "  $0 ../prod_weight_shapes.txt ../small_test.txt 30    # Create 30-line subset"
    echo "  $0 ../proxy_weight_shapes.txt ../quick_test.txt 10   # Create 10-line subset from proxy"
    exit 1
fi

# Get total lines
TOTAL_LINES=$(wc -l < "$INPUT_FILE")

if [[ $NUM_LINES -ge $TOTAL_LINES ]]; then
    echo "Warning: Requested $NUM_LINES lines, but file only has $TOTAL_LINES lines"
    echo "Copying entire file..."
    cp "$INPUT_FILE" "$OUTPUT_FILE"
else
    echo "Creating subset: $NUM_LINES lines from $TOTAL_LINES total"
    
    # Two strategies:
    # 1. For small subset: take first N lines (fastest)
    # 2. For representative subset: sample evenly
    
    if [[ $NUM_LINES -le 50 ]]; then
        echo "Taking first $NUM_LINES lines..."
        head -n "$NUM_LINES" "$INPUT_FILE" > "$OUTPUT_FILE"
    else
        # Sample evenly distributed lines
        STEP=$((TOTAL_LINES / NUM_LINES))
        echo "Sampling every ${STEP}th line..."
        awk -v step=$STEP 'NR == 1 || NR % step == 0' "$INPUT_FILE" | head -n "$NUM_LINES" > "$OUTPUT_FILE"
    fi
fi

# Show results
CREATED_LINES=$(wc -l < "$OUTPUT_FILE")
echo ""
echo "✓ Created: $OUTPUT_FILE"
echo "  Lines: $CREATED_LINES (from $TOTAL_LINES in original)"
echo ""
echo "To use this test file:"
echo "  ./run_optimization.sh baseline $OUTPUT_FILE"
echo "  ./run_optimization.sh quick $OUTPUT_FILE"
echo "  ./run_optimization.sh full $OUTPUT_FILE"
