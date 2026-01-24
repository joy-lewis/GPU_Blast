build: gpu_example.cu
	nvcc -arch=sm_86 -Werror all-warnings -lcurand -O2 -o gpu_ex gpu_example.cu
	
	