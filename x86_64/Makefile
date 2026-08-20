# ==============================================================================
# x86_64 Components Makefile
# ==============================================================================

CURRENT_DIR := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
SRC_ROOT    := $(abspath $(CURRENT_DIR)/..)
MODE        ?= debug

BUILD_DIR   := $(SRC_ROOT)/build/$(MODE)/x86_64
BIN_DIR     := $(SRC_ROOT)/build/$(MODE)/x86_64
CUDA_BUILD  := $(SRC_ROOT)/build/$(MODE)/cuda

AS          := as
LD          := ld

# Voeg -I$(CUDA_BUILD) toe zodat .incbin de gbm_monte_carlo.cubin kan vinden
ASFLAGS     := --64 -msyntax=att -mmnemonic=att -I$(CUDA_BUILD)
LDFLAGS     := -m elf_x86_64 -dynamic-linker /lib64/ld-linux-x86-64.so.2 -lc -lcuda

ifeq ($(MODE),debug)
    ASFLAGS += -g
    LDFLAGS += -g
else
    LDFLAGS += -s
endif

TARGETS = cuda complex-quad-solver

.PHONY: all debug release clean $(TARGETS)

all: debug

debug:
	@$(MAKE) --no-print-directory MODE=debug build_all

release:
	@$(MAKE) --no-print-directory MODE=release build_all

build_all: directories $(TARGETS)
ifeq ($(MODE),release)
	@rm -rf $(BUILD_DIR)/*.o $(BUILD_DIR)/*.lst 2>/dev/null || true
endif

directories:
	@mkdir -p $(BUILD_DIR)

complex-quad-solver: directories
	@echo "Building x86_64 complex-quad-solver [$(MODE)]..."
	$(AS) $(ASFLAGS) -c complex-quad-solver.s -o $(BUILD_DIR)/complex-quad-solver.o -a=$(BUILD_DIR)/complex-quad-solver.lst
	$(LD) $(LDFLAGS) $(BUILD_DIR)/complex-quad-solver.o -o $(BIN_DIR)/complex-quad-solver

clean:
	@echo "Cleaning x86_64 artifacts..."
	rm -f $(BUILD_DIR)

