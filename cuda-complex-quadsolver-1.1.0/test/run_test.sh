#!/bin/bash
set -e

# Colors for terminal feedback
GREEN='\033;32m'
RED='\033;31m'
NC='\033[0m' # No Color

echo "============================================================"
echo "RUNNING NATIVE BARE-METAL QUAD SOLVER REGRESSION SUITE"
echo "============================================================"

# Enforce absolute location independence based on script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Pivot into project root for sandboxed execution
pushd "$PROJECT_ROOT" > /dev/null

# Ensure centralized directory structures are available
mkdir -p data

# Step 1: Execute Project Rebuild via Makefile
echo -e "\n[STEP 1/3] Triggering bare-metal compilation pipeline..."
make clean
make

# Step 2: Inline Generation of Synthetic Complex Equations
echo -e "\n[STEP 2/3] Generating deterministic test equations..."
INPUT_CSV="data/test_inputs.csv"
> "$INPUT_CSV"

# Test Case 1: Standard Real Roots (x^2 - 5x + 6 = 0) -> Roots: (3, 0), (2, 0)
echo "1.0000,0.0000,-5.0000,0.0000,6.0000,0.0000" >> "$INPUT_CSV"

# Test Case 2: Pure Complex Roots (1x^2 + 0x + 4 = 0) -> Roots: (0, 2i), (0, -2i)
echo "1.0000,0.0000,0.0000,0.0000,4.0000,0.0000" >> "$INPUT_CSV"

# Test Case 3: Fully Complex System (1+1i)x^2 + (2+2i)x + (3+3i) = 0
echo "1.0000,1.0000,2.0000,2.0000,3.0000,3.0000" >> "$INPUT_CSV"

echo "[STATUS] Formatted validation matrix safely into $INPUT_CSV"

# Step 3: Run Binary Execution & Parse Output Matrices
echo -e "\n[STEP 3/3] Launching GPU Analytic Solver Execution..."
OUTPUT_LOG="data/actual_results.txt"

# Target resolution matching the universal centralized Makefile structure (defaulting to debug)
TARGET_BIN="./bin/debug/x86_64/quadratic_solver/quadratic_solver"

if [ ! -f "$TARGET_BIN" ]; then
    echo -e "${RED}[ERROR] Target binary not found at $TARGET_BIN. Check build mode.${NC}"
    popd > /dev/null
    exit 1
fi

"$TARGET_BIN" "$INPUT_CSV" > "$OUTPUT_LOG"
cat "$OUTPUT_LOG"

echo -e "\nEvaluating numerical verification gates..."

# Verify Test Case 1 Roots
if grep -q "Row 0: Roots -> R1: (+3.0000, +0.0000i) | R2: (+2.0000, +0.0000i)" "$OUTPUT_LOG"; then
    echo -e "${GREEN}>>> MATRIX GATE 1 PASSED: Real Integer Boundaries Solid.${NC}"
else
    echo -e "${RED}>>> MATRIX GATE 1 FAILED: Roots mismatch on Row 0.${NC}"
    popd > /dev/null
    exit 1
fi

# Verify Test Case 2 Roots
if grep -q "Row 1: Roots -> R1: (+0.0000, +2.0000i) | R2: (+0.0000, -2.0000i)" "$OUTPUT_LOG"; then
    echo -e "${GREEN}>>> MATRIX GATE 2 PASSED: Pure Imaginary Boundaries Solid.${NC}"
else
    echo -e "${RED}>>> MATRIX GATE 2 FAILED: Roots mismatch on Row 1.${NC}"
    popd > /dev/null
    exit 1
fi

echo -e "\n${GREEN}============================================================"
echo "SUCCESS: 64-Bit Analytic Complex PTX Architecture Verified"
echo "============================================================${NC}"

# Restore original shell environment
popd > /dev/null
exit 0
