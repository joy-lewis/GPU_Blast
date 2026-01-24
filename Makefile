build: gpu.cu
	nvcc -arch=sm_86 -Werror all-warnings -lcurand -O2 -o gpu gpu.cu
	
	