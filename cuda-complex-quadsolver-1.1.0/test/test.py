#!/usr/bin/env python3
import subprocess
import sys
import cmath
import csv
import os

# Definieer testgevallen: coëfficiënten (a1, a2, b1, b2, c1, c2)
test_cases = [
    # Test Case 1: x^2 - 5x + 6 = 0 -> Roots: 3, 2
    (1.0, 0.0, -5.0, 0.0, 6.0, 0.0),
    # Test Case 2: x^2 + 4 = 0 -> Roots: 2i, -2i
    (1.0, 0.0, 0.0, 0.0, 4.0, 0.0),
    # Test Case 3: Complex systeem
    (1.0, 1.0, 2.0, 2.0, 3.0, 3.0)
]

def solve_quadratic_analytic(a, b, c):
    """Berekent analytisch de exacte complexe wortels via cmath."""
    discriminant = cmath.sqrt(b**2 - 4*a*c)
    r1 = (-b + discriminant) / (2*a)
    r2 = (-b - discriminant) / (2*a)
    return r1, r2

def main():
    input_csv = "data/test_inputs.csv"
    output_csv = "output.csv"
    binary_path = "./bin/debug/x86_64/quadratic_solver/quadratic_solver"

    os.makedirs("data", exist_ok=True)

    # Stap 1: Genereer input CSV
    print("[STEP 1/3] Generating test inputs via Python...")
    with open(input_csv, "w", newline="") as f:
        writer = csv.writer(f)
        for case in test_cases:
            writer.writerow(case)

    # Stap 2: Voer de binary uit
    print("[STEP 2/3] Executing bare-metal GPU solver...")
    if not os.path.exists(binary_path):
        print(f"[ERROR] Binary not found at {binary_path}. Build first.")
        sys.exit(1)

    result = subprocess.run([binary_path, input_csv], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    if result.returncode != 0:
        print(f"[ERROR] Solver exited with code {result.returncode}")
        sys.exit(1)

    # Stap 3: Vergelijk resultaten met Python referentie
    print("[STEP 3/3] Verifying numerical accuracy...")
    if not os.path.exists(output_csv):
        print(f"[ERROR] Output file {output_csv} not generated.")
        sys.exit(1)

    with open(output_csv, "r") as f:
        reader = csv.reader(f)
        rows = list(reader)

    tolerance = 1e-3
    passed = True

    for i, row in enumerate(rows):
        if i >= len(test_cases):
            break
        
        # Parse coëfficiënten uit input
        a = complex(float(row[0]), float(row[1]))
        b = complex(float(row[2]), float(row[3]))
        c = complex(float(row[4]), float(row[5]))

        # Parse GPU berekende wortels uit output
        gpu_r1 = complex(float(row[6]), float(row[7]))
        gpu_r2 = complex(float(row[8]), float(row[9]))

        # Bereken gouden standaard via Python
        ref_r1, ref_r2 = solve_quadratic_analytic(a, b, c)

        # Vergelijk met tolerantie (check beide combinaties ivm volgorde wortels)
        match_direct = (abs(gpu_r1 - ref_r1) < tolerance and abs(gpu_r2 - ref_r2) < tolerance)
        match_swapped = (abs(gpu_r1 - ref_r2) < tolerance and abs(gpu_r2 - ref_r1) < tolerance)

        if match_direct or match_swapped:
            print(f"  > Row {i}: PASSED (GPU: {gpu_r1}, {gpu_r2} | Ref: {ref_r1:.4f}, {ref_r2:.4f})")
        else:
            print(f"  > Row {i}: FAILED (GPU: {gpu_r1}, {gpu_r2} | Ref: {ref_r1:.4f}, {ref_r2:.4f})")
            passed = False

    if passed:
        print("\nSUCCESS: All regression gates passed with floating-point precision!")
        sys.exit(0)
    else:
        print("\nFAILURE: Numerical mismatch detected.")
        sys.exit(1)

if __name__ == "__main__":
    main()
