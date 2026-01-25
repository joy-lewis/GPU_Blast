build: gpu_blast.cu
	nvcc -arch=sm_86 -Werror all-warnings -lcurand -O2 -o gpu_blast gpu_blast.cu
	
	