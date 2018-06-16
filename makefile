_K = 4096

all:
	g++ kmeans.cc -o kmeans_cpu   -std=c++11 -O3 -DK_FLAG=$(_K)
	nvcc kmeans.cu -o kmeans_gpu  -std=c++11 -I ../NVIDIA_CUDA-9.1_Samples/common/inc -DK_FLAG=$(_K)
