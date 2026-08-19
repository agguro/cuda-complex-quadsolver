# CUDA Complex Quadratic Solver

Bare-metal high-throughput complex quadratic equation solver implemented with:
* **x86_64 Assembly** (System V ABI Compliance)
* **NVIDIA PTX** (Native IEEE-754 Float64 Precision)
* **CUDA Driver API** (Zero-overhead direct module loading)

The engine performs direct, non-iterative analytic resolution of millions of simultaneous complex quadratic expressions of the form $ax^2 + bx + c = 0$, where all variables and coefficients ($a, b, c$) possess distinct real and imaginary components.

---

## Architecture & Layout

```text
Input Matrix (.csv)
        |
        v
x86_64 Assembly Host Layer
        |
        +-- sys_mmap (Direct virtual memory ingestion)
        +-- sscanf parser (High-density float data packing)
        +-- 16-Byte Stack-Aligned Frame Guarding
        └── CUDA Driver API Orchestration
                |
                v
NVIDIA PTX Core Kernel
        |
        +-- Complex discriminant transformation: b^2 - 4ac
        +-- Non-iterative half-plane sign stabilization
        +-- Floating point conjugate division block
                |
                v
GPU Device Output VRAM Buffer
        |
        v
Standard Output Layer (PLT printf)
```

---

## Project Structure

```text
.
├── bin/
│   └── x86_64/
│       └── quad_solver          # Final compiled native host engine
├── build/
│   └── x86_64/
│       ├── quadratic_solver.o   # Compiled CPU assembly host
│       └── solver_kernel.cubin  # Assembled GPU binary slice
├── kernels/
│   └── quadratic_solver_kernel.ptx # Double precision complex solver kernel
├── src/
│   └── x86_64/
│       └── quadratic_solver.s   # Pure System V host orchestration source
├── test/
│   └── test.sh                  # Pure shell validation harness
├── Makefile                     # Bare-metal build definitions
└── LICENSE                      # System tracking terms
```

---

## Ingestion Stream Layout

Input files must map to packed comma-separated ASCII streams containing 6 distinct float parameters per row:
```text
a_real, a_imag, b_real, b_imag, c_real, c_imag
```
Each parsed calculation frame is automatically balanced onto a 64-byte quadword row window inside your VRAM grid mapping space.

---

## Build and Automation Verification

To clean existing temporary object allocations, trigger compiling sequences, build the device binaries, and execute the automated verification test run at once:

```bash
./test/test.sh
```

---

## Execution Format

Run the application by passing your target calculation parameters file explicitly:
```bash
./bin/x86_64/quad_solver data/inputs.csv
```

---

## License

This analytical engine is licensed under the Apache 2.0 open source tracking provisions.
