# 1. Compiler Discovery
# 'shell which nvcc' finds the compiler in the user's current PATH (e.g., Anaconda or System)
# If not found, it defaults to the standard NVIDIA install path.
NVCC := $(shell which nvcc 2>/dev/null)
ifeq ($(NVCC),)
    NVCC := /usr/local/cuda/bin/nvcc
endif

# 2. Architecture Autodetect
# This automatically finds the Compute Capability of the machine running the Makefile.
# If no GPU is found, it defaults to a broad compatibility architecture (sm_70).
ARCH := $(shell $(NVCC) --version >/dev/null 2>&1 && \
         echo "-arch=native" || echo "-arch=sm_70")

# 3. Compilation Flags
# -O3: Maximum optimization
# -std=c++17: Language standard
# -Xcompiler -Wall: Passes 'all warnings' to the underlying C++ compiler (gcc/clang)
NVCC_FLAGS := -std=c++17 -O3 $(ARCH) -Xcompiler -Wall

# 4. Project Structure
TARGET := gpu_blast
SRCS := main.cu gpu_blast.cu
HEADERS := gpu_blast.h lookup_table.cuh
DATA_DIR := ncbi_data

# 5. Build Rules
all: $(TARGET) copy_data

$(TARGET): $(SRCS) $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) $(SRCS) -o $(TARGET)

copy_data:
	@mkdir -p results
	@if [ -d "$(DATA_DIR)" ]; then \
		echo "Found $(DATA_DIR), ensuring it exists in build folder..."; \
		cp -r $(DATA_DIR) . 2>/dev/null || true; \
	else \
		echo "Warning: $(DATA_DIR) not found. Sequence files might be missing."; \
	fi

clean:
	rm -rf $(TARGET) results/
	@echo "Cleanup complete."

.PHONY: all clean copy_data