#include <stdio.h>

#include "GPUTree.cuh"
#include "cuda_helpers.cuh"

#include <cub/cub.cuh>
#include <thrust/gather.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/binary_search.h>


//Functor for getting the instance ID from sort order
struct GetInstanceIDFunc : public thrust::unary_function<int, int>
{
	int n_instances;

	GetInstanceIDFunc(int n_instances)
		: n_instances(n_instances)
	{
	}

	__host__ __device__
	int operator()(int x) const
	{
		return x % n_instances;
	}
};

//Custom functor to enable the reduction to return the index of the max item
struct InfoReduce
{
	float* d_info;

	InfoReduce(float* d_info) : d_info(d_info)
	{
	}

	__device__ CUB_RUNTIME_FUNCTION __forceinline__
	int operator()(const int& i1, const int& i2) const
	{
		if (i1 == -1 && i2 == -1)
		{
			return -1;
		}

		else if (i1 == -1)
		{
			return i2;
		}
		else if (i2 == -1)
		{
			return i1;
		}
		else
		{
			return d_info[i1] > d_info[i2] ? i1 : i2;
		}
	}
};

__device__ float entropy(int n_a, int n)
{
	if (n_a == 0 || n_a == n || n == 0)
	{
		return 0;
	}
	float p_a = (float)n_a / n;
	float p_b = 1 - p_a;
	return -(p_a * log2f(p_a)) - (p_b * log2f(p_b));
}

//We can infer the attribute index from the sort order
__device__ int getAttributeIndex(const int i, const int* d_sort_order, const int n_instances)
{
	return d_sort_order[i] / n_instances;
};

__device__ float partial_entropy(int n_a, int n)
{
	if (n_a == 0 || n_a == n || n == 0)
	{
		return 0;
	}
	float p_a = (float)n_a / n;
	return -(p_a * log2f(p_a));
}

__device__ float infogain(const int i, const int* d_sort_order, const float* d_attributes, const int* d_node_index, const int* d_node_offsets, const int* d_node_count, const int* d_scan_classes, const int n_attributes, const int n_instances, const int class_value, const int n_class_values)
{
	//If this attribute value is the same as the next attribute to the left then just return 0
	//We cannot use this as a split point
	if (i > 0)
	{
		if (d_attributes[d_sort_order[i]] == d_attributes[d_sort_order[i - 1]])
		{
			return 0;
		}
	}

	//Current node of the tree we are in
	int node = d_node_index[i];

	//Total instances in this node
	int node_total = (d_node_offsets[node + 1] - d_node_offsets[node]) / n_attributes;

	//Positive instances in this node for a given class value
	int node_positive = d_node_count[node * n_class_values + class_value];

	//Index of the start of this segment
	int attribute_start = d_node_offsets[node] + getAttributeIndex(i, d_sort_order, n_instances) * node_total;

	//The index of the current instance within this node
	int instance_i = i - attribute_start;

	float node_entropy = partial_entropy(node_positive, node_total);

	int left_positive = d_scan_classes[i] - d_scan_classes[attribute_start];
	int left_negative = instance_i - left_positive;
	int right_positive = node_positive - left_positive;
	int right_negative = node_total - left_positive - left_negative - right_positive;

	float entropy_left = partial_entropy(left_positive, left_negative + left_positive);
	float entropy_right = partial_entropy(right_positive, right_negative + right_positive);

	return node_entropy - ((float)instance_i / node_total) * entropy_left - ((float)(node_total - instance_i) / node_total) * entropy_right;
}

__global__ void infogain_kernel(float* d_infogain, int* d_sort_order, float* d_attributes, int* d_node_index, int* d_node_offsets, int* d_node_positive_count, int* d_scan_classes, int n_attributes, int n_instances, int n, const int class_value, const int n_class_values)
{
	for (auto i : grid_stride_range(0, n))
	{
		d_infogain[i] += infogain(i, d_sort_order, d_attributes, d_node_index, d_node_offsets, d_node_positive_count, d_scan_classes, n_attributes, n_instances, class_value, n_class_values);
	}
}

void sort_attributes(CubMemory& cub_memory, float* d_attributes_in, float* d_attributes_out, int* d_indices_in, int* d_indices_out, int n, int n_attributes, int n_instances)
{
	//Calculate offsets
	thrust::host_vector<int> h_offsets(n_attributes + 1);
	h_offsets[n_attributes] = n;
	for (int i = 0; i < n_attributes; i++)
	{
		h_offsets[i] = i * n_instances;
	}

	thrust::device_vector<int> offsets = h_offsets;
	cub::DeviceSegmentedRadixSort::SortPairs(cub_memory.d_temp_storage, cub_memory.temp_storage_bytes, d_attributes_in, d_attributes_out, d_indices_in, d_indices_out, n, n_attributes, raw(offsets), raw(offsets) + 1);

	cub_memory.allocate();

	cub::DeviceSegmentedRadixSort::SortPairs(cub_memory.d_temp_storage, cub_memory.temp_storage_bytes, d_attributes_in, d_attributes_out, d_indices_in, d_indices_out, n, n_attributes, raw(offsets), raw(offsets) + 1);
}

__global__ void histogram(int* d_node_count, const int* d_tmp_classes, const int* d_node_index, const int* d_sort_order, const int n, const int n_instances, const int n_class_values)
{
	for (auto i : grid_stride_range(0, n))
	{
		//Only count items from the first segment of each node
		if (getAttributeIndex(i, d_sort_order, n_instances) != 0)
		{
			continue;
		}

		int class_value = d_tmp_classes[i];
		int node_index = d_node_index[i];


		atomicAdd(&d_node_count[node_index * n_class_values + class_value], 1);
	}
}

__global__ void distribution(float* d_distribution, int* d_node_count, int *d_current_node_count,int* d_node_offsets, const int n_nodes, const int n_class_values, const int n_attributes)
{
	const int n_dist = n_nodes * n_class_values;
	for (auto i : grid_stride_range(0, n_dist))
	{
		int node_index = i / n_class_values;
		//Use node offsets to get total instances in node
		int node_total = (d_node_offsets[node_index + 1] - d_node_offsets[node_index]) / n_attributes;
		int count = d_node_count[i];

		d_current_node_count[i] = count;
		if (node_total == 0)
		{
			d_distribution[i] = 0;
		}
		else
		{
			d_distribution[i] = (float)count / node_total;

			assert(d_distribution[i] <= 1.0);
		}
	}
}

//Calculates class distribution for all nodes on the current level
//Also populates node counts which are used in infogain calculation
void calculate_distribution(thrust::device_vector<int>& tmp_classes, float* d_current_distribution, int  *d_current_counts,thrust::device_vector<int>& node_count, thrust::device_vector<int>& node_index, thrust::device_vector<int>& sort_order, thrust::device_vector<int>& node_offsets, const int n, const int n_nodes, const int n_instances, const int n_class_values, const int n_attributes)
{
	const int block_threads = 256;
	const int items_per_thread = 4;

	//Get node counts
	thrust::fill(node_count.begin(), node_count.end(), 0);
	const int n_blocks_hist = std::ceil((double)n / (block_threads * items_per_thread));

	histogram << < n_blocks_hist, block_threads >> >(raw(node_count), raw(tmp_classes), raw(node_index), raw(sort_order), n, n_instances, n_class_values);
	safe_cuda(cudaDeviceSynchronize());

	//Calculate distribution
	const int n_blocks_dist = std::ceil((double)(n_nodes * n_class_values) / (block_threads * items_per_thread));

	distribution << <n_blocks_dist, block_threads >> >(d_current_distribution, raw(node_count),d_current_counts, raw(node_offsets), n_nodes, n_class_values, n_attributes);
	safe_cuda(cudaDeviceSynchronize());
}

void calculate_infogain(CubMemory& cub_memory, thrust::device_vector<float>& info, thrust::device_vector<float>& attributes, thrust::device_vector<int>& sort_order, thrust::device_vector<int>& tmp_classes, thrust::device_vector<int>& node_count, thrust::device_vector<int>& node_offsets, thrust::device_vector<int>& node_index, int n, int n_attributes, int n_instances, int n_class_values, int n_nodes)
{
	thrust::device_vector<int> scan_classes(n);
	int* d_scan_classes = raw(scan_classes);
	int* d_tmp_classes = raw(tmp_classes);
	auto counting = thrust::make_counting_iterator<int>(0);

	//Reset infogain values
	thrust::fill(info.begin(), info.end(), 0);

	for (int val = 0; val < n_class_values; val++)
	{
		//Copy in 1 flags for the values we are working on in this iteration
		thrust::fill(scan_classes.begin(), scan_classes.end(), 0);
		thrust::for_each(counting, counting + n, [=]__device__(int i)
		                 {
			                 if (d_tmp_classes[i] == val)
			                 {
				                 d_scan_classes[i] = 1;
			                 }
		                 });

		//Scan classes
		cub::DeviceScan::ExclusiveSum(cub_memory.d_temp_storage, cub_memory.temp_storage_bytes, raw(scan_classes), raw(scan_classes), n);

		//Calculate partial infogain
		const int block_threads = 256;
		const int items_per_thread = 4;
		const int n_blocks = std::ceil((double)n / (block_threads * items_per_thread));

		infogain_kernel << <n_blocks, block_threads >> >(raw(info), raw(sort_order), raw(attributes), raw(node_index), raw(node_offsets), raw(node_count), raw(scan_classes), n_attributes, n_instances, n, val, n_class_values);

		safe_cuda(cudaDeviceSynchronize());
	}
}


extern "C"
{
	EXPORT int generate_tree(const float* in_attributes, const int n_attributes, const char* in_classes, const int n_instances, const int n_class_values, const int n_levels, TreeNode* out_tree)
	{
		if (n_attributes == 0 || n_instances == 0 || n_class_values == 0 || n_levels == 0)
		{
			return 0;
		}

		int n = n_attributes * n_instances;

		int max_nodes = std::pow(2, n_levels + 1) - 1;
		thrust::device_vector<int> split_attribute(max_nodes, -1);
		thrust::device_vector<float> split_value(max_nodes, 0);
		thrust::device_vector<float> distribution(max_nodes * n_class_values, 0);
		thrust::device_vector<int > counts(max_nodes * n_class_values, 0);

		thrust::device_vector<float> attributes(in_attributes, in_attributes + n);
		thrust::device_vector<char> classes(in_classes, in_classes + n_instances);

		//Temporary array for storing scanned classes
		thrust::device_vector<int> tmp_classes(n);

		//Contains node index (at the current level of the tree) for each item
		thrust::device_vector<int> node_index(n, 0);

		//Sort

		thrust::device_vector<int> sort_order(n);
		thrust::device_vector<int> indices(n);
		thrust::sequence(indices.begin(), indices.end());
		//We will temporarily use this as the output sorted attributes but later to hold infogain
		thrust::device_vector<float> info(n);

		CubMemory cub_memory;

		Timer t;
		sort_attributes(cub_memory, raw(attributes), raw(info), raw(indices), raw(sort_order), n, n_attributes, n_instances);
		t.printElapsed("first sort");
		t.reset();

		//Prepare other data structures
		thrust::device_vector<char> sort_key(n);
		thrust::device_vector<char> sort_key_out(n);

		thrust::device_vector<int> node_count(max_nodes * n_class_values); //Stores counts of class value for each node.
		thrust::device_vector<int> node_best_infogain_index(max_nodes);

		//node offsets
		thrust::device_vector<int> node_offsets(max_nodes + 1, n);
		node_offsets[0] = 0;

		//Iterators
		auto counting = thrust::make_counting_iterator<int>(0);
		auto attribute_perm = thrust::make_permutation_iterator(attributes.begin(), sort_order.begin());
		GetInstanceIDFunc instance_id_func(n_instances);
		auto instance_id = thrust::make_transform_iterator(sort_order.begin(), instance_id_func);

		for (int level = 0; level < n_levels + 1; level++)
		{
			t.reset();

			//Number of nodes at this level
			int n_nodes_level = 1 << level;

			//Copy in class
			thrust::gather(instance_id, instance_id + n, classes.begin(), tmp_classes.begin());

			t.printElapsed("gather");
			t.reset();

			//Calculate distribution for nodes of current level
			float* d_current_distribution = raw(distribution) + (((1 << level) - 1) * n_class_values);
			int * d_current_counts = raw(counts) + (((1 << level) - 1) * n_class_values);

			calculate_distribution(tmp_classes, d_current_distribution, d_current_counts, node_count, node_index, sort_order, node_offsets, n, n_nodes_level, n_instances, n_class_values, n_attributes);

			if (level == n_levels)
			{
				break;
			}
			calculate_infogain(cub_memory, info, attributes, sort_order, tmp_classes, node_count, node_offsets, node_index, n, n_attributes, n_instances, n_class_values, n_nodes_level);

			float* d_info = raw(info);
			int* d_node_index = raw(node_index);
			int* d_node_offsets = raw(node_offsets);
			int* d_sort_order = raw(sort_order);

			//Find the indices of the elements with the highest infogain
			InfoReduce reduce_op(d_info);
			cub::DeviceSegmentedReduce::Reduce(cub_memory.d_temp_storage, cub_memory.temp_storage_bytes, raw(indices), raw(node_best_infogain_index), n_nodes_level, raw(node_offsets), raw(node_offsets) + 1, reduce_op, - 1);
			t.printElapsed("infogain reduce");
			t.reset();

			//Record best splits
			int* d_node_best_infogain_index = raw(node_best_infogain_index);
			//Increment pointers to current tree level
			float* d_current_split_value = raw(split_value) + ((1 << level) - 1);
			int* d_current_split_attribute = raw(split_attribute) + ((1 << level) - 1);

			thrust::for_each(counting, counting + n_nodes_level, [=]__device__(int i)
			                 {
				                 int best_index = d_node_best_infogain_index[i];

				                 if (best_index == -1)
				                 {
					                 return;
				                 }
				                 else if (d_info[best_index] == 0)
				                 {
									 d_node_best_infogain_index[i] = -1;
					                 return;
				                 }

				                 //Use the weka method where split point is the mean of two nodes
				                 int lower_index = best_index - 1 >= 0 ? best_index - 1 : 0;
				                 float attribute_split = (attribute_perm[best_index] + attribute_perm[lower_index]) / 2.0;
				                 int attribute_index = getAttributeIndex(best_index, d_sort_order, n_instances);

				                 d_current_split_value[i] = attribute_split;
				                 d_current_split_attribute[i] = attribute_index;
			                 });


			t.printElapsed("create node");
			t.reset();

			//Prepare sort keys
			float* d_attributes = raw(attributes);
			char* d_sort_key = raw(sort_key);
			thrust::for_each(counting, counting + n, [=]__device__(int i)
			                 {
				                 int node = d_node_index[i];
				                 int attribute_index = d_current_split_attribute[node];
				                 float split_value = d_current_split_value[node];

				                 int query_index = attribute_index * n_instances + instance_id[i];

				                 if (d_attributes[query_index] < split_value)
				                 {
					                 d_sort_key[i] = 0;
				                 }
				                 else
				                 {
					                 d_sort_key[i] = 1;
				                 }
			                 });

			t.printElapsed("prepare sort key");
			t.reset();
			//Use tmp_classes as the input array to sort
			tmp_classes = sort_order;

			t.printElapsed("copy input array to sort");
			t.reset();
			cub::DeviceSegmentedRadixSort::SortPairs(cub_memory.d_temp_storage, cub_memory.temp_storage_bytes, raw(sort_key), raw(sort_key_out), raw(tmp_classes), raw(sort_order), n, n_nodes_level, raw(node_offsets), raw(node_offsets) + 1, 0, 1);

			t.printElapsed("sort one bit");
			t.reset();
			//Calculate new node offsets
			int* d_new_node_offsets = raw(node_offsets) + n_nodes_level;
			
			thrust::for_each(counting, counting + n_nodes_level, [=]__device__(int i)
			                 {
				                 int best_index = d_node_best_infogain_index[i];

				                 //Account for empty nodes
				                 if (best_index == -1)
				                 {
					                 d_new_node_offsets[i + 1] = d_node_offsets[i];
					                 return;
				                 }

				                 //Total instances in this nodes
				                 int node_total = (d_node_offsets[i + 1] - d_node_offsets[i]) / n_attributes;
								 
				                 //Instances to the left of the split point
				                 int instances_left = best_index - (d_node_offsets[i] + d_current_split_attribute[i] * node_total);

				                 //Write into array 1 place to the right so we dont have a read/write conflict - the sort will put it into correct place later
				                 d_new_node_offsets[i + 1] = d_node_offsets[i] + instances_left * n_attributes;
			                 });

			t.printElapsed("calculate node offsets");
			t.reset();

			//Sort new offsets into place
			thrust::sort(node_offsets.begin(), node_offsets.end());

			t.printElapsed("sort offsets");
			t.reset();

			//Populate node indices
			thrust::for_each(counting, counting + n, [=]__device__(int i)
			                 {
				                 int old_index = d_node_index[i];

				                 //The new node we are in will be a child of the previous node
				                 if (i >= d_node_offsets[old_index * 2 + 1])
				                 {
					                 d_node_index[i] = old_index * 2 + 1;
				                 }
				                 else
				                 {
					                 d_node_index[i] = old_index * 2;
				                 }
			                 });

			t.printElapsed("create node indices");
			t.reset();
		}

		safe_cuda(cudaDeviceSynchronize());

		//Copy data out
		thrust::host_vector<float> h_distribution_tree = distribution;
		thrust::host_vector<int > h_counts_tree = counts;
		thrust::host_vector<float> h_split_value = split_value;
		thrust::host_vector<int> h_split_attribute = split_attribute;

		for (int i = 0; i < max_nodes; i++)
		{
			out_tree[i].attributeIndex = h_split_attribute[i];
			out_tree[i].attributeValue = h_split_value[i];

			//Copy distribution
			for (int j = 0; j < n_class_values; j++)
			{
				out_tree[i].distribution[j] = h_distribution_tree[i * n_class_values + j];
				out_tree[i].counts[j] = h_counts_tree[i * n_class_values + j];
			}
		}

		return 0;
	}

	EXPORT bool test_cuda()
	{
		int device_count;
		cudaError_t e = cudaGetDeviceCount(&device_count);
		if (e == cudaSuccess && device_count > 0)
		{
			return true;
		}
		else
		{
			return false;
		}
	}

	EXPORT void force_context_init()
	{
		safe_cuda(cudaFree(0));
		safe_cuda(cudaDeviceSynchronize());
	}
}
