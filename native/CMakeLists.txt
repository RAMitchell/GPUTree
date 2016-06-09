
# CMakeLists.txt 
cmake_minimum_required(VERSION 2.8)
find_package(CUDA QUIET REQUIRED)

set(CUDA_NVCC_FLAGS "--expt-extended-lambda") 

set( CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${PROJECT_SOURCE_DIR}/../lib)
set( CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${PROJECT_SOURCE_DIR}/../lib)
set( CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${PROJECT_SOURCE_DIR}/../lib)
set( CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG ${PROJECT_SOURCE_DIR}/../lib)

set(SOURCES 
	src/cuda_helpers.cu
	src/cuda_helpers.cuh
	src/GPUTree.cu
	src/GPUTree.cuh
)

set(EXE_SOURCES 
	${SOURCES} 
	src/main.cu
	src/data.cpp
	src/data.h
)

#cuda_add_executable(
#	GPUTreeApp
#	${EXE_SOURCES}
#	)
 

cuda_add_library(
	GPUTree
	SHARED
	${SOURCES}
)



