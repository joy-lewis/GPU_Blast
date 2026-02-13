# 1. Compiler Discovery
NVCC := $(shell which nvcc 2>/dev/null)
ifeq ($(NVCC),)
    NVCC := /usr/local/cuda/bin/nvcc
endif

# 2. Universal Architecture (Fat Binary)
# This targets common professional/academic GPUs:
# sm_75: Turing (RTX 20-series, T4)
# sm_80: Ampere (A100)
# sm_86: Ampere (RTX 30-series)
# sm_89: Ada (RTX 40-series, L40)
# sm_90: Hopper/Blackwell (RTX 6000, H100) - includes PTX for forward compatibility
GENCODE := -gencode arch=compute_75,code=sm_75 \
           -gencode arch=compute_80,code=sm_80 \
           -gencode arch=compute_86,code=sm_86 \
           -gencode arch=compute_89,code=sm_89 \
           -gencode arch=compute_90,code=compute_90

# 3. Compilation Flags
NVCC_FLAGS := -std=c++17 -O3 $(GENCODE) -Xcompiler -Wall

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
	@if [ -d "$(DATA_DIR)" ]; then cp -r $(DATA_DIR) . 2>/dev/null || true; fi

clean:
	rm -rf $(TARGET) results/

.PHONY: all clean copy_data