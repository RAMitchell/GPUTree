#include "infogain.cuh"
#include "cuda_helpers.cuh"
#include "TreeNode.h"
#include <thrust/gather.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <cub/cub.cuh>

__device__ float entropy(int n_a, int n){

	int n_b = n - n_a;
	if (n_a == 0 || n_a == n || n == 0){
		return 0;
	}
	float p_a = (float)n_a / (n_a + n_b);
	float p_b = 1 - p_a;
	return -(p_a * std::log2(p_a)) - (p_b * std::log2(p_b));
}

void calculate_infogain(int n, cub::DoubleBuffer<int> &db_sample_index, thrust::host_vector<thrust::device_vector<float >> &attributes, cub::DoubleBuffer<float >&db_sort_key,thrust::device_vector<int> &node_index, thrust::device_vector<int >&classes,thrust::device_vector<int >&tmp_classes,
	thrust::device_vector<int > & node_positive_count,
	thrust::device_vector<int > & node_total_count,
	thrust::device_vector<int > & node_start_index,
	thrust::device_vector<int >& node_best_infogain_index,
	 TreeNode*d_current_nodes,
	int attribute,
	int n_nodes,
	void *d_temp_storage,
	size_t temp_storage_bytes
	){
	
	
	//Make iterators
	auto discard = thrust::make_discard_iterator();
	auto counting = thrust::make_counting_iterator<int >(0);
	auto constant_one = thrust::make_constant_iterator<int >(1);

	//Copy in sort key
	thrust::gather(dptr(db_sample_index.Current()), dptr(db_sample_index.Current()) + n, attributes[attribute].begin(), dptr(db_sort_key.Current()));

	// Segmented sort
	cub::DeviceSegmentedRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, db_sort_key, db_sample_index, n, n_nodes, raw( node_start_index), raw( node_start_index)+1);

	//Copy in class
	thrust::gather(dptr(db_sample_index.Current()), dptr(db_sample_index.Current()) + n, classes.begin(), tmp_classes.begin());

	//Count positives
	//thrust::reduce_by_key(node_index.begin(), node_index.end(), tmp_classes.begin(), discard, node_positive_count.begin());
	cub::DeviceSegmentedReduce::Sum(d_temp_storage,temp_storage_bytes,raw(tmp_classes),raw( node_positive_count),n_nodes,raw(node_start_index), raw(node_start_index)+1);

	//Scan classes
	thrust::exclusive_scan_by_key(node_index.begin(), node_index.end(), tmp_classes.begin(), tmp_classes.begin());

	//Calculate infogain
	int *d_tmp_classes = raw(tmp_classes);
	int *d_node_index = raw(node_index);
	int *d_node_offsets = raw(node_start_index);
	int *d_node_positive_count = raw(node_positive_count);

	auto infogain = [=]__device__(int i){
		//Current node of the tree we are in
		int node = d_node_index[i];

		//Total instances in this node
		int node_total = d_node_offsets[node+1]-d_node_offsets[node];

		//Total positive instances in this node
		int node_positive = d_node_positive_count[node];

		//Our current position within this node
		int node_i = i - d_node_offsets[node];

		float node_entropy = entropy(node_positive, node_total);

		int left_positive = d_tmp_classes[i];
		int left_negative = node_i - left_positive;
		int right_positive = node_positive - left_positive;
		int right_negative = node_total - left_positive - left_negative - right_positive;

		float entropy_left = entropy(left_positive, left_negative + left_positive);
		float entropy_right = entropy(right_positive, right_negative + right_positive);

		return node_entropy - ((float)node_i / node_total) * entropy_left - ((float)(node_total - node_i) / node_total) * entropy_right;

	};

	//Get best infogain index for each node
	//Additionally we may not select a split point in the middle of a cluster of same valued attributes

	float *d_sort_key = db_sort_key.Current();
	thrust::reduce_by_key(node_index.begin(), node_index.end(), counting, discard, node_best_infogain_index.begin(), thrust::equal_to<int>(), [=]__device__(int i1, int i2){

		//Ensure a same valued attribute never gets returned ahead of a non-same valued attribute
		if (i1 > 0){
			if (d_sort_key[i1] == d_sort_key[i1 - 1]){
				return i2;
			}
		}
		if (i2 > 0){
			if (d_sort_key[i2] == d_sort_key[i2 - 1]){
				return i1;
			}
		}

		return infogain(i1) < infogain(i2) ? i2 : i1;
	});


	// Debugging
	/*
	std::cout << "attribute: " << attribute << "\n";
	thrust::device_vector<int > debug_classes(classes.size());
	thrust::gather(sample_index.begin(), sample_index.end(), classes.begin(), debug_classes.begin());
	thrust::device_vector<float >  info(sort_key.size());
	float *d_info = raw(info);
	thrust::for_each(counting, counting + info.size(), [=]__device__(int i){
		d_info[i] = infogain(i);
	});
	print(sample_index);
	print(node_index);
	print(debug_classes);
	print(sort_key);
	thrust::host_vector<float > h_info = info;
	for (int i = 0; i < info.size(); i++){
		printf("%1.2f ", h_info[i]);
	}
	std::cout << "\n";
	*/


	//If infogain is better replace node

	int *d_node_best_infogain_index = raw(node_best_infogain_index);
	float *d_attribute = raw(attributes[attribute]);


	thrust::for_each(counting, counting + n_nodes, [=]__device__(int i){
		int contender_index = d_node_best_infogain_index[i];
		float contender_infogain = infogain(contender_index);
		if (contender_infogain > d_current_nodes[i].infogain){
			//Use the weka method where split point is the mean of two nodes
			int lower_contender_index = contender_index - 1 >= 0 ? contender_index - 1 : 0;
			float attribute_split = (d_sort_key[contender_index] + d_sort_key[lower_contender_index]) / 2.0;
			d_current_nodes[i] = TreeNode(attribute, attribute_split, contender_infogain);
		}
	});

}