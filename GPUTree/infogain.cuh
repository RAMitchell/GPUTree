#pragma once
#include <thrust/device_vector.h>

class TreeNode;

void calculate_infogain(thrust::device_vector<int >& sample_index, thrust::host_vector<thrust::device_vector<float >> &attributes, thrust::device_vector<float >&sort_key, thrust::device_vector<int> &node_index, thrust::device_vector<int >&classes, thrust::device_vector<int >&tmp_classes,
	thrust::device_vector<int > & node_positive_count,
	thrust::device_vector<int > & node_total_count,
	thrust::device_vector<int > & node_start_index,
	thrust::device_vector<int >& node_best_infogain_index,
	TreeNode*d_current_nodes,
	int attribute,
	int n_nodes);
