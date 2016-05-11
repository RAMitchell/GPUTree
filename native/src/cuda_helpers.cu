
#include <thrust/system_error.h>
#include <thrust/system/cuda/error.h>
#include <thrust/device_vector.h>
#include <sstream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"



void throw_on_cuda_error(cudaError_t code, const char *file, int line)
{
	if (code != cudaSuccess)
	{
		std::cout << file;
		std::cout << line;
		std::stringstream ss;
		ss << file << "(" << line << ")";
		std::string file_and_line;
		ss >> file_and_line;
		throw thrust::system_error(code, thrust::cuda_category(), file_and_line);
	}
}


__device__ int tid()
{
	return blockIdx.x *blockDim.x + threadIdx.x;
}

//Utility function: rounds up integer division.
//Can overflow on large numbers and does not work with negatives
__host__ __device__ int div_round_up(int a, int b){
	return (a + b - 1) / b;
}



