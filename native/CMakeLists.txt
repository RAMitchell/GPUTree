
# CMakeLists.txt 
cmake_minimum_required(VERSION 2.8)
find_package(CUDA QUIET REQUIRED)

set(CUDA_NVCC_FLAGS "--expt-extended-lambda") 

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

cuda_add_executable(
	GPUTreeApp
	${EXE_SOURCES}
)
 

cuda_add_library(
	GPUTree
	SHARED
	${SOURCES}
)

SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/../lib)

set_property(TARGET GPUTree PROPERTY LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
#set_target_properties(GPUTree PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
#set_target_properties(GPUTree PROPERTIES LIBRARY_OUTPUT_DIRECTORY_RELEASE ${PROJECT_BINARY_DIR}/lib)
#set_target_properties(GPUTree PROPERTIES LIBRARY_OUTPUT_DIRECTORY_DEBUG ${PROJECT_BINARY_DIR}/lib)
