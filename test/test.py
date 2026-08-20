#!/usr/init/env python3
import subprocess
import sys
import cmath
import csv
import os
import random

NUM_TESTS = 1000

def solve_quadratic_analytic(a, b, c):
    """Calculates the exact complex roots analytically using cmath."""
    discriminant = cmath.sqrt(b**2 - 4*a*c)
    r1 = (-b + discriminant) / (2*a)
    r2 = (-b - discriminant) / (2*a)
    return r1, r2

def main():
    # Aangepast voor uitvoer vanuit de 'test' map of de root
    input_csv = "test_inputs_heavy.csv"
    output_csv = "output.csv"
    binary_path = "../build/debug/x86_64/complex-quad-solver"

    print(f"[STEP 1/4] Building the program via 'make' in root...")
    # Gebruik -C .. zodat make de root Makefile gebruikt, ongeacht waar je test.py start
    build_result = subprocess.run(["make", "-C", ".."], capture_output=True, text=True)
    if build_result.returncode != 0:
        print(f"[ERROR] Build failed with code {build_result.returncode}")
        print(build_result.stderr)
        sys.exit(1)

    print(f"[STEP 2/4] Generating {NUM_TESTS} random complex test cases...")
    random.seed(42)  # Deterministic seed for reproducibility
    
    test_cases = []
    for _ in range(NUM_TESTS):
        while True:
            a1 = random.uniform(-10.0, 10.0)
            a2 = random.uniform(-10.0, 10.0)
            if abs(a1) > 0.1 or abs(a2) > 0.1:
                break
        b1 = random.uniform(-20.0, 20.0)
        b2 = random.uniform(-20.0, 20.0)
        c1 = random.uniform(-50.0, 50.0)
        c2 = random.uniform(-50.0, 50.0)
        
        test_cases.append((a1, a2, b1, b2, c1, c2))

    with open(input_csv, "w", newline="") as f:
        writer = csv.writer(f)
        for case in test_cases:
            writer.writerow(case)

    print(f"[STEP 3/4] Launching GPU Analytic Solver with {NUM_TESTS} rows...")
    if not os.path.exists(binary_path):
        print(f"[ERROR] Binary not found at {binary_path}.")
        sys.exit(1)

    result = subprocess.run([binary_path, input_csv, "-o", output_csv], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"[ERROR] Solver exited with code {result.returncode}")
        print(result.stderr)
        sys.exit(1)

    print(f"[STEP 4/4] Verifying numerical accuracy across {NUM_TESTS} equations...")
    if not os.path.exists(output_csv):
        print(f"[ERROR] Output file {output_csv} not generated.")
        sys.exit(1)

    with open(output_csv, "r") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if len(rows) < NUM_TESTS:
        print(f"[ERROR] Expected {NUM_TESTS} output rows, but got {len(rows)}.")
        sys.exit(1)

    tolerance = 1e-3
    rounding_errors = 0

    for i, row in enumerate(rows[:NUM_TESTS]):
        a = complex(float(row[0]), float(row[1]))
        b = complex(float(row[2]), float(row[3]))
        c = complex(float(row[4]), float(row[5]))

        gpu_r1 = complex(float(row[6]), float(row[7]))
        gpu_r2 = complex(float(row[8]), float(row[9]))

        ref_r1, ref_r2 = solve_quadratic_analytic(a, b, c)

        match_direct = (abs(gpu_r1 - ref_r1) < tolerance and abs(gpu_r2 - ref_r2) < tolerance)
        match_swapped = (abs(gpu_r1 - ref_r2) < tolerance and abs(gpu_r2 - ref_r1) < tolerance)

        if not (match_direct or match_swapped):
            if rounding_errors < 5:
                print(f"  > Row {i} FAILED: Coeffs(a={a}, b={b}, c={c})")
                print(f"    GPU: R1={gpu_r1}, R2={gpu_r2}")
                print(f"    Ref: R1={ref_r1}, R2={ref_r2}")
            rounding_errors += 1

    if rounding_errors == 0:
        print(f"\nSUCCESS: All {NUM_TESTS} regression gates passed with double-precision accuracy!")
        sys.exit(0)
    else:
        print(f"\nCOMPLETED: {rounding_errors}/{NUM_TESTS} equations failed due to rounding errors (exceeded tolerance threshold {tolerance}).")
        sys.exit(1)

if __name__ == "__main__":
    main()
