# ==============================================================================
# BARE-METAL GPU QUADRATIC SOLVER MAKEFILE
# ==============================================================================

AS      := as
LD      := gcc
PTXAS   := ptxas

# Target architecture matching your server GPU (sm_61+)
ARCH    := sm_61

# Directories
SRC_DIR   := src/x86_64
KERNEL_DIR:= kernels
BUILD_DIR := build/x86_64
BIN_DIR   := bin/x86_64

# Targets
TARGET    := $(BIN_DIR)/quad_solver
CUBIN     := $(BUILD_DIR)/solver_kernel.cubin

# Host Objects
OBJS      := $(BUILD_DIR)/quadratic_solver.o

# Flags
ASFLAGS   := --64
PTXFLAGS  := -v -arch=$(ARCH)
LDFLAGS   := -no-pie -lcuda

.PHONY: all clean directories

all: directories $(CUBIN) $(TARGET)

directories:
	@mkdir -p $(BUILD_DIR)
	@mkdir -p $(BIN_DIR)

# Compile PTX to CUBIN (Verifies GPU instructions compile optimally)
$(CUBIN): $(KERNEL_DIR)/quadratic_solver_kernel.ptx
	@echo "[GPU] Assembling PTX Kernel..."
	$(PTXAS) $(PTXFLAGS) $< -o $@

# Assemble host x86_64 code
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.s
	@echo "[CPU] Assembling Host Code..."
	$(AS) $(ASFLAGS) $< -o $@

# Link final executable against Driver API
$(TARGET): $(OBJS)
	@echo "[LINK] Tying objects to libcuda.so..."
	$(LD) $(OBJS) $(LDFLAGS) -o $@
	@echo ">>> BUILD COMPLETE: $(TARGET)"

clean:
	@echo "[CLEAN] Purging objects and binaries..."
	rm -rf bin/ build/
