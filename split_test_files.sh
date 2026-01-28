#!/bin/bash
#
# Split test files into convolution and matmul test files
# Matmuls have: -y 1 -x 1
# Convolutions have: other y, x values
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parse arguments
INPUT_FILE="${1}"
OUTPUT_PREFIX="${2}"

if [[ -z "$INPUT_FILE" || -z "$OUTPUT_PREFIX" ]]; then
    echo "Usage: $0 INPUT_FILE OUTPUT_PREFIX"
    echo ""
    echo "Splits a test file into separate convolution and matmul files."
    echo ""
    echo "Examples:"
    echo "  $0 ../prod_weight_shapes.txt ../prod"
    echo "    Creates: ../prod_conv.txt and ../prod_matmul.txt"
    echo ""
    echo "  $0 ../proxy_weight_shapes.txt ../proxy"
    echo "    Creates: ../proxy_conv.txt and ../proxy_matmul.txt"
    exit 1
fi

if [[ ! -f "$INPUT_FILE" ]]; then
    echo "Error: Input file not found: $INPUT_FILE"
    exit 1
fi

# Output files
CONV_FILE="${OUTPUT_PREFIX}_conv.txt"
MATMUL_FILE="${OUTPUT_PREFIX}_matmul.txt"

# Clear output files if they exist
> "$CONV_FILE"
> "$MATMUL_FILE"

# Split based on -y 1 -x 1 pattern (matmuls)
# Matmuls have: -y 1 -x 1
# Convolutions have: -y <other> -x <other>

TOTAL_LINES=0
MATMUL_COUNT=0
CONV_COUNT=0

while IFS= read -r line; do
    TOTAL_LINES=$((TOTAL_LINES + 1))
    
    # Check if this is a matmul (has "-y 1" AND "-x 1")
    if [[ "$line" =~ -y\ 1 ]] && [[ "$line" =~ -x\ 1 ]]; then
        echo "$line" >> "$MATMUL_FILE"
        MATMUL_COUNT=$((MATMUL_COUNT + 1))
    else
        echo "$line" >> "$CONV_FILE"
        CONV_COUNT=$((CONV_COUNT + 1))
    fi
done < "$INPUT_FILE"

echo "============================================"
echo "Split Test Files"
echo "============================================"
echo ""
echo "Input:  $INPUT_FILE ($TOTAL_LINES lines)"
echo ""
echo "Output:"
echo "  Convolutions: $CONV_FILE ($CONV_COUNT lines)"
echo "  Matmuls:      $MATMUL_FILE ($MATMUL_COUNT lines)"
echo ""
echo "Verification:"
if [[ $((CONV_COUNT + MATMUL_COUNT)) -eq $TOTAL_LINES ]]; then
    echo "  ✓ Sum matches: $CONV_COUNT + $MATMUL_COUNT = $TOTAL_LINES"
else
    echo "  ✗ Warning: Sum mismatch!"
fi
echo ""
echo "To use these files:"
echo "  ./run_optimization.sh baseline $CONV_FILE $MATMUL_FILE"
echo "  ./run_optimization.sh quick $CONV_FILE"
echo "  ./run_optimization.sh full $MATMUL_FILE"
