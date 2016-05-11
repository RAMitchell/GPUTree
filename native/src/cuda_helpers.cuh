#pragma once
#include <thrust/system_error.h>
#include <thrust/system/cuda/error.h>
#include <thrust/device_vector.h>
#include <sstream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <ctime>
#include <algorithm>

#define safe_cuda(ans) { throw_on_cuda_error((ans), __FILE__, __LINE__); }

void throw_on_cuda_error(cudaError_t code, const char *file, int line);

__device__ int tid();

//Utility function: rounds up integer division.
//Can overflow on large numbers and does not work with negatives
__host__ __device__ int div_round_up(int a, int b);


template <typename T>
thrust::device_ptr<T> dptr(T*d_ptr){
	return thrust::device_pointer_cast(d_ptr);
}

#define NTIMERS
struct Timer{
	size_t start;
	Timer(){
		reset();
	}
	void reset(){
		start = clock();
	}
	double elapsed(){
		return ((double)clock() - start) / CLOCKS_PER_SEC;
	}
	void printElapsed(char * label){
#ifndef NTIMERS
		safe_cuda(cudaDeviceSynchronize());
		std::cout << label << ": " << elapsed() << "s\n";
#endif
	}

};

template <typename T>
void print(const thrust::device_vector<T>& v)
{
	thrust::host_vector<T> h = v;
	for (auto elem : h)
		std::cout << " " << elem;
	std::cout << "\n";
}

template <typename T>
void print(char *label, const thrust::device_vector<T>& v, const char * format = "%d ",int max = 10)
{
	thrust::host_vector<T> h_v = v;

	std::cout << label << ":\n";
	for (int i = 0; i < std::min((int)h_v.size(), max); i++)
	{
		printf(format, h_v[i]);
	}
	std::cout << "\n";
}

class range {
public:
	class iterator {
		friend class range;
	public:
		__host__ __device__
		long int operator *() const { return i_; }
		__host__ __device__
		const iterator &operator ++() { i_ += step_; return *this; }
		__host__ __device__
		iterator operator ++(int) { iterator copy(*this); i_ += step_; return copy; }

		__host__ __device__
		bool operator ==(const iterator &other) const { return i_ >= other.i_; }
		__host__ __device__
		bool operator !=(const iterator &other) const { return i_ < other.i_; }

		__host__ __device__
		void step(int s){ step_ = s; }
	protected:
		__host__ __device__
		iterator(long int start) : i_(start) { }

	//private:
	public:
		unsigned long i_;
		int step_ = 1;
	};

	__host__ __device__
	iterator begin() const { return begin_; }
	__host__ __device__
	iterator end() const { return end_; }
	__host__ __device__
	range(long int  begin, long int end) : begin_(begin), end_(end) {}
	__host__ __device__
	void step(int s) { begin_.step(s); }
private:
	iterator begin_;
	iterator end_;
};

template <typename T>
__device__ range grid_stride_range(T begin, T end){
	begin += blockDim.x * blockIdx.x + threadIdx.x;
	range r(begin, end);
	r.step(gridDim.x * blockDim.x);
	return r;
}

//Converts device_vector to raw pointer
template <typename T>
T * raw(thrust::device_vector<T>& v){
	return raw_pointer_cast(v.data());
}