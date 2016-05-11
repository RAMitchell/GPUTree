#include <stdio.h>
#include <algorithm>
#include <numeric>

#include "GPUTree.cuh"
#include "cuda_helpers.cuh"

#include <cub/cub.cuh>
#include <thrust/gather.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/binary_search.h>


//Custom functor to enable the reduction to return the index of the max item
struct  InfoReduce{

	float *d_info;

	InfoReduce(float *d_info) : d_info(d_info){}

	__device__ CUB_RUNTIME_FUNCTION __forceinline__
	int operator()(const int & i1, const int &i2)const {
		if (i1 == -1 && i2 == -1)
		{
			return  -1;
		}

		else if (i1 == -1)
		{
			return i2;
		}
		else if (i2 == -1)
		{
			return i1;
		}
		else{
			return d_info[i1] > d_info[i2] ? i1 : i2;
		}
	}
};

__device__ float entropy(int n_a, int n){

	if (n_a == 0 || n_a == n || n == 0){
		return 0;
	}
	float p_a = (float)n_a / (n);
	float p_b = 1 - p_a;
	return -(p_a * log2f(p_a)) - (p_b * log2f(p_b));
}

//We can infer the attribute index from the sort order
__device__ int getAttributeIndex(int i, int *d_sort_order, int n_instances){
	return d_sort_order[i] / n_instances;
};

__device__ float infogain(int i, int *d_sort_order, float *d_attributes, int *d_node_index, int *d_node_offsets, int *d_node_positive_count, int *d_tmp_classes, int n_attributes, int n_instances){

		//If this attribute value is the same as the next attribute to the left then just return 0
		//We cannot use this as a split point
		if (i > 0){
			if (d_attributes[d_sort_order[i]] == d_attributes[d_sort_order[i - 1]]){
				return 0;
			}
		}

		//Current node of the tree we are in
		int node = d_node_index[i];

		//Total instances in this node
		int node_total = (d_node_offsets[node+1]-d_node_offsets[node])/n_attributes;

		//Total positive instances in this node
		int node_positive = d_node_positive_count[node]/n_attributes;

		//Index of the start of this segment
		int  attribute_start = d_node_offsets[node] + getAttributeIndex(i, d_sort_order, n_instances) * node_total;

		//The index of the current instance within this node
		int instance_i = i - attribute_start;

		float node_entropy = entropy(node_positive, node_total);

		int left_positive = d_tmp_classes[i] - d_tmp_classes[attribute_start];
		int left_negative = instance_i - left_positive;
		int right_positive = node_positive - left_positive;
		int right_negative = node_total - left_positive - left_negative - right_positive;

		float entropy_left = entropy(left_positive, left_negative + left_positive);
		float entropy_right = entropy(right_positive, right_negative + right_positive);

		return node_entropy - ((float)instance_i / node_total) * entropy_left - ((float)(node_total - instance_i) / node_total) * entropy_right;
	
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void  infogain_kernel(float *d_infogain, int *d_sort_order, float *d_attributes, int *d_node_index, int *d_node_offsets, int *d_node_positive_count, int *d_tmp_classes, int n_attributes, int n_instances, int n){

	for (auto i : grid_stride_range(0, n)){
		d_infogain[i] = infogain(i, d_sort_order, d_attributes, d_node_index, d_node_offsets, d_node_positive_count, d_tmp_classes, n_attributes, n_instances);
	}

}


extern "C"{

	
EXPORT int  generate_tree(const float * in_attributes,const  int n_attributes, const char * in_classes, const  int n_instances, const int n_levels, TreeNode*out_tree){

	int n = n_attributes * n_instances;

	int total_nodes = std::pow(2, n_levels) - 1;
	thrust::device_vector<  TreeNode  >  decision_tree(total_nodes);

	thrust::device_vector<float > attributes(in_attributes, in_attributes + n);
	thrust::device_vector<char > classes(n);
	thrust::device_vector<int  > tmp_classes(n );

	thrust::device_vector<int  > node_index(n,0 );

	thrust::host_vector<int> h_offsets(n_attributes+1);
	h_offsets[n_attributes] = n;

	//Calculate offsets
	for (int i = 0; i < n_attributes; i++){
		thrust::copy(in_classes, in_classes + n_instances, classes.begin() + i * n_instances);
		h_offsets[i] = i * n_instances;
	}

	thrust::device_vector<int> offsets = h_offsets;	

	//Perform sort
	thrust::device_vector<int > sort_order(n);
	thrust::device_vector<int >  indices(n); //TODO: Can we get away without allocating this array?
	thrust::sequence(indices.begin(), indices.end());
	//We will temporarily use this as the output sorted attributes but later to hold infogain
	thrust::device_vector<float >   info(n);

	void *d_temp_storage = NULL;
	size_t temp_storage_bytes = 0;

	Timer t;

	cub::DeviceSegmentedRadixSort::SortPairs(d_temp_storage,temp_storage_bytes,raw( attributes),raw(info), raw(indices),raw(sort_order),n,n_attributes,raw(offsets),raw(offsets)+1);

	safe_cuda(cudaMalloc(&d_temp_storage, temp_storage_bytes));

	cub::DeviceSegmentedRadixSort::SortPairs(d_temp_storage,temp_storage_bytes,raw( attributes),raw( info), raw( indices),raw(sort_order),n,n_attributes,raw(offsets),raw(offsets)+1);

	t.printElapsed("first sort");
	t.reset();

	//Prepare other data structures
	thrust::device_vector<char > sort_key(n);
	thrust::device_vector<char > sort_key_out(n);

	//Max number of nodes at any given level
	int max_nodes = std::pow(2, n_levels - 1);
	thrust::device_vector<int >  node_positive_count(max_nodes);
	thrust::device_vector< int  >  node_best_infogain_index(max_nodes);

	//node offsets
	thrust::device_vector<int > node_offsets(max_nodes + 1, n);
	node_offsets[0] = 0;

	//Iterators
	auto counting = thrust::make_counting_iterator<int >(0);
	auto discard = thrust::make_discard_iterator();
	auto constant_one = thrust::make_constant_iterator<int >(1);
	auto  attribute_perm = thrust::make_permutation_iterator(attributes.begin(),sort_order.begin());

	for (int level = 0; level < n_levels; level++){

		t.reset();

		//Number of nodes at this level
		int n_nodes = 1 << level;

		//Get pointer to current level's nodes
		TreeNode* d_current_nodes = raw(decision_tree) + ((1 << level) - 1);

		//Copy in class
		thrust::gather(sort_order.begin(), sort_order.end(), classes.begin(), tmp_classes.begin());
		t.printElapsed("gather");
		t.reset();

		//Count positives in each node - when we use this we must divide it by the number of attributes
		cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, raw(tmp_classes), raw(node_positive_count), n_nodes, raw(node_offsets), raw(node_offsets) + 1);

		t.printElapsed("count positives");
		t.reset();

		//Scan classes - segmented on attribute and node boundaries
		int *d_node_index = raw(node_index);
		int *d_node_offsets = raw(node_offsets);
		int *d_sort_order = raw(sort_order);

		cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, raw(tmp_classes), raw(tmp_classes), n);

		t.printElapsed("scan classes");
		t.reset();

		//Calculate infogain
		int *d_tmp_classes = raw(tmp_classes);
		int *d_node_positive_count = raw(node_positive_count);

		const int block_threads = 256;
		const int items_per_thread = 4;
		const int n_blocks = std::ceil((double)n / (block_threads * items_per_thread));

		infogain_kernel<block_threads, items_per_thread> << <n_blocks, block_threads >> >(raw( info), raw(sort_order) , raw(attributes), raw(node_index), raw(node_offsets), raw(node_positive_count), raw(tmp_classes), n_attributes, n_instances, n);

		safe_cuda(cudaDeviceSynchronize());
		t.printElapsed("infogain  calculate");
		t.reset();

		float *d_info = raw(info);

		InfoReduce  reduce_op(d_info);
		cub::DeviceSegmentedReduce::Reduce(d_temp_storage, temp_storage_bytes, raw(indices), raw(node_best_infogain_index), n_nodes, raw(node_offsets), raw(node_offsets) + 1, reduce_op,  - 1);
		t.printElapsed("infogain reduce");
		t.reset();		
		

		//Create a node for best splits
		int *d_node_best_infogain_index = raw(node_best_infogain_index);
		thrust::for_each(counting, counting + n_nodes, [=]__device__(int i){
			int  best_index = d_node_best_infogain_index[i];

			if (best_index == -1){
				return;
			}

			//Use the weka method where split point is the mean of two nodes
			int lower_index =  best_index - 1 >= 0 ? best_index - 1 : 0;
			float attribute_split = (attribute_perm[best_index] + attribute_perm[lower_index]) / 2.0;
			int attribute_index = getAttributeIndex(best_index, d_sort_order, n_instances);

			//Total instances in this node
			int node_total = (d_node_offsets[i+1]-d_node_offsets[i])/n_attributes;

			//Total positive instances in this node
			int node_positive = d_node_positive_count[i]/n_attributes;

			//Index of the start of this segment
			int  attribute_start = d_node_offsets[i] + getAttributeIndex(best_index, d_sort_order, n_instances) * node_total;

			//The index of the current instance within this node
			int instance_i =  best_index - attribute_start;

			int left_positive = d_tmp_classes[best_index] - d_tmp_classes[attribute_start];
			int left_negative = instance_i - left_positive;
			int right_positive = node_positive - left_positive;
			int right_negative = node_total - left_positive - left_negative - right_positive;

			float  left_prob =  (float) left_positive/ (left_positive+left_negative);
			float  right_prob = (float)  right_positive/ ( right_positive+right_negative);

			d_current_nodes[i] = TreeNode(attribute_index, attribute_split, d_info[best_index],left_prob, right_prob);

		});		
		

		//Exit the loop here if we are done - don't bother preparing data structures for next loop
		if (level == n_levels - 1){
			break;
		}

		t.printElapsed("create node");
		t.reset();

		//Prepare sort keys
		float *d_attributes = raw(attributes);
		char *d_sort_key = raw(sort_key);
		thrust::for_each(counting, counting + n, [=]__device__(int i){
			int node = d_node_index[i];
			int attribute_index = d_current_nodes[node].attributeIndex;
			float split_value = d_current_nodes[node].attributeValue;

			int instance = d_sort_order[i] % n_instances;

			int query_index = attribute_index * n_instances + instance;

			if (d_attributes[query_index] < split_value){
				d_sort_key[i] = 0;
			}
			else{
				d_sort_key[i] = 1;
			}

		});

		t.printElapsed("prepare sort key");
		t.reset();
		//Use tmp_classes as the input array to sort
		tmp_classes = sort_order;

		t.printElapsed("copy input array to sort");
		t.reset();
		cub::DeviceSegmentedRadixSort::SortPairs(d_temp_storage,temp_storage_bytes,raw(sort_key),raw(sort_key_out),raw(tmp_classes),raw( sort_order),n,n_nodes,raw(node_offsets),raw(node_offsets)+1,0,1);

		t.printElapsed("sort one bit");
		t.reset();
		//Calculate new node offsets
		int *d_new_node_offsets = raw(node_offsets) + n_nodes;
		thrust::for_each(counting, counting + n_nodes, [=]__device__(int i){

			int  best_index = d_node_best_infogain_index[i];

			//Account for empty nodes
			if (best_index == -1){
				d_new_node_offsets[i + 1] = d_node_offsets[i];
				return;
			}

			//Total instances in this nodes
			int node_total = (d_node_offsets[i+1]-d_node_offsets[i])/n_attributes;
			
			//Instances to the left of the split point
			int instances_left = best_index - (d_node_offsets[i] + d_current_nodes[i].attributeIndex * node_total);

			//Write into array 1 place to the right so we dont have a read/write conflict - the sort will put it into correct place later
			d_new_node_offsets[i+1] = d_node_offsets[i] + instances_left * n_attributes;
		});

		t.printElapsed("calculate node offsets");
		t.reset();

		//Sort new offsets into place
		thrust::copy(node_offsets.begin(), node_offsets.end(), tmp_classes.begin());
		cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, raw(tmp_classes), raw(node_offsets),  node_offsets.size());

		t.printElapsed("sort offsets");
		t.reset();

		//Populate node indices
		thrust::for_each(counting, counting + n, [=]__device__(int i){
			int old_index = d_node_index[i];

			//The new node we are in will be a child of the previous node
			if (i >= d_node_offsets[old_index * 2 + 1]){
				d_node_index[i] = old_index * 2 + 1;
			}
			else{
				d_node_index[i] = old_index * 2;
			}
		});

		t.printElapsed("create node indices");
		t.reset();
	}

	safe_cuda(cudaDeviceSynchronize());
	safe_cuda(cudaFree(d_temp_storage));

	thrust::copy(decision_tree.begin(), decision_tree.end(), out_tree);

	return 0;

}

	EXPORT bool test_cuda(){
		int device_count;
		cudaError_t e = cudaGetDeviceCount(&device_count);
		if (e == cudaSuccess&&device_count>0){
			return true;
		}
		else{
			return  false;
		}
	}

	EXPORT void force_context_init(){
		safe_cuda(cudaFree(0));
		safe_cuda(cudaDeviceSynchronize());
	}
}
