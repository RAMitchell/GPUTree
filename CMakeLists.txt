
# CMakeLists.txt 
cmake_minimum_required(VERSION 2.8)
find_package(CUDA QUIET REQUIRED)

set(CUDA_NVCC_FLAGS "--expt-extended-lambda") 

FILE(GLOB SOURCES src/*)

cuda_add_executable(
	GPUTree
	${SOURCES}
)

cuda_add_library(
	GPUTreelib
	SHARED
	${SOURCES}
)


