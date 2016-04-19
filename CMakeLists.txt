
# CMakeLists.txt 
cmake_minimum_required(VERSION 2.8)
find_package(CUDA QUIET REQUIRED)

set(CUDA_NVCC_FLAGS "--expt-extended-lambda") 

# Specify binary name and source file to build it from
cuda_add_executable(
	GPUTree
	src/kernel.cu
	src/data.cpp
	src/cuda_helpers.cu
	src/TreeNode.cpp
)

