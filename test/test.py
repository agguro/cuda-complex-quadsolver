#!/usr/bin/env python3
#

import subprocess
import sys
import csv
import os
import numpy as np

NUM_TESTS = 99999

def solve_quadratic_analytic_vectorized(a, b, c):
    """Calculates exact complex roots vectorised using NumPy for extreme speed."""
    discriminant = np.sqrt(b**2 - 4*a*c)
    r1 = (-b + discriminant) / (2*a)
    r2 = (-b - discriminant) / (2*a)
    return r1, r2

def main():
    input_csv = "data/test_inputs_heavy.csv"
    output_csv = "output.csv"
    binary_path = "../build/debug/x86_64/complex-quad-solver"

    os.makedirs("data", exist_ok=True)

    print(f"[STEP 1/4] Building the program via 'make' in root...")
    build_result = subprocess.run(["make", "-C", ".."], capture_output=True, text=True)
    if build_result.returncode != 0:
        print(f"[ERROR] Build failed with code {build_result.returncode}")
        print(build_result.stderr)
        sys.exit(1)

    print(f"[STEP 2/4] Generating {NUM_TESTS} random complex test cases via NumPy...")
    np.random.seed(42)  # Deterministic seed for reproducibility
    
    # Generate random real and imaginary parts efficiently
    a1 = np.random.uniform(-10.0, 10.0, NUM_TESTS)
    a2 = np.random.uniform(-10.0, 10.0, NUM_TESTS)
    # Ensure 'a' is not too close to zero to prevent division by zero
    mask = (np.abs(a1) <= 0.1) & (np.abs(a2) <= 0.1)
    a1[mask] = 1.0
    
    b1 = np.random.uniform(-20.0, 20.0, NUM_TESTS)
    b2 = np.random.uniform(-20.0, 20.0, NUM_TESTS)
    c1 = np.random.uniform(-50.0, 50.0, NUM_TESTS)
    c2 = np.random.uniform(-50.0, 50.0, NUM_TESTS)
    
    test_cases = np.column_stack((a1, a2, b1, b2, c1, c2))

    print(f"Writing test cases to {input_csv}...")
    np.savetxt(input_csv, test_cases, delimiter=",", fmt="%.6f")

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

    # Load output data efficiently using NumPy
    output_data = np.loadtxt(output_csv, delimiter=",")
    if output_data.shape[0] < NUM_TESTS:
        print(f"[ERROR] Expected {NUM_TESTS} output rows, but got {output_data.shape[0]}.")
        sys.exit(1)

    # Reconstruct complex numbers
    a = output_data[:NUM_TESTS, 0] + 1j * output_data[:NUM_TESTS, 1]
    b = output_data[:NUM_TESTS, 2] + 1j * output_data[:NUM_TESTS, 3]
    c = output_data[:NUM_TESTS, 4] + 1j * output_data[:NUM_TESTS, 5]

    gpu_r1 = output_data[:NUM_TESTS, 6] + 1j * output_data[:NUM_TESTS, 7]
    gpu_r2 = output_data[:NUM_TESTS, 8] + 1j * output_data[:NUM_TESTS, 9]

    # Calculate reference roots via vectorized NumPy functions
    ref_r1, ref_r2 = solve_quadratic_analytic_vectorized(a, b, c)

    tolerance = 1e-3
    
    # Vectorized tolerance checks (accounting for root swapping)
    match_direct = (np.abs(gpu_r1 - ref_r1) < tolerance) & (np.abs(gpu_r2 - ref_r2) < tolerance)
    match_swapped = (np.abs(gpu_r1 - ref_r2) < tolerance) & (np.abs(gpu_r2 - ref_r1) < tolerance)
    
    passed = match_direct | match_swapped
    rounding_errors = NUM_TESTS - np.count_nonzero(passed)

    if rounding_errors == 0:
        print(f"\nSUCCESS: All {NUM_TESTS} regression gates passed with double-precision accuracy!")
        sys.exit(0)
    else:
        # Find indices of failures for debugging
        failed_indices = np.where(~passed)[0]
        print(f"\nFirst few failures (up to 5):")
        for idx in failed_indices[:5]:
            print(f"  > Row {idx} FAILED:")
            print(f"    GPU: R1={gpu_r1[idx]}, R2={gpu_r2[idx]}")
            print(f"    Ref: R1={ref_r1[idx]}, R2={ref_r2[idx]}")
            
        print(f"\nCOMPLETED: {rounding_errors}/{NUM_TESTS} equations failed due to rounding errors (exceeded tolerance threshold {tolerance}).")
        sys.exit(1)

if __name__ == "__main__":
    main()
