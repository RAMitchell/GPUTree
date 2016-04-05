#include <stdio.h>
#include <vector>
#include <map>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <algorithm>
#include <numeric>

#include "cuda_helpers.cuh"
#include "data.h"
#include "TreeNode.h"
#include "infogain.cuh"

#include <thrust/gather.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/binary_search.h>







void infogainSplit(Data&data, int n_levels){

	int  n_attributes = data.attributes.size();
	int n = data.classes.size();

	//Move to device memory and create device pointers for attributes
	thrust::device_vector<int> classes = data.classes;
	thrust::host_vector<thrust::device_vector<float >> attributes(n_attributes);
	float **d_attributes;
	float **h_attributes = new float *[n_attributes];

	for (int i = 0; i < n_attributes; i++){
		attributes[i] = data.attributes[i];
		h_attributes[i] = raw(attributes[i]);
	}

	safe_cuda(cudaMalloc(&d_attributes, sizeof(float *) * n_attributes));
	safe_cuda(cudaMemcpy(d_attributes, h_attributes, sizeof(float *) * n_attributes, cudaMemcpyHostToDevice));



	thrust::device_vector<int > sample_index(n);
	int* d_sample_index = raw(sample_index);
	thrust::sequence(sample_index.begin(), sample_index.end());
	thrust::device_vector<float > sort_key(n);
	float *d_sort_key = raw(sort_key);
	thrust::device_vector<int > tmp_classes(n);
	thrust::device_vector<int >  node_index(n, 0);
	int *d_node_index = raw(node_index);

	//Max number of nodes at any given level
	int max_nodes = std::pow(2, n_levels - 1);
	thrust::device_vector<int >  node_positive_count(max_nodes);
	thrust::device_vector<int >  node_total_count(max_nodes);
	thrust::device_vector<int >  node_start_index(max_nodes,INT_MAX);
	node_start_index[0] = 0;
	thrust::device_vector< int  >  node_best_infogain_index(max_nodes);

	//Total nodes
	int total_nodes = std::pow(2, n_levels) - 1;
	thrust::device_vector<  TreeNode  >  decision_tree(total_nodes);


	//Make iterators
	auto discard = thrust::make_discard_iterator();
	auto counting = thrust::make_counting_iterator<int >(0);
	auto constant_one = thrust::make_constant_iterator<int >(1);
	auto node_positive_count_perm = thrust::make_permutation_iterator(node_positive_count.begin(), node_index.begin());
	auto node_total_count_perm = thrust::make_permutation_iterator(node_total_count.begin(), node_index.begin());
	auto node_start_index_perm = thrust::make_permutation_iterator(node_start_index.begin(), node_index.begin());



	for (int level = 0; level < n_levels; level++){
		//Number of nodes at this level
		int n_nodes = 1 << level;
		TreeNode* d_current_nodes = raw(decision_tree) + ((1 << level) - 1);
		//Calculate Max infogain
		for (int attribute = 0; attribute < n_attributes; attribute++){

			calculate_infogain(sample_index, attributes, sort_key, node_index, classes, tmp_classes, node_positive_count, node_total_count, node_start_index, node_best_infogain_index, d_current_nodes, attribute, n_nodes);
		}

		//Sort values within each node according to attribute with highest infogain

		//Copy attributes
		thrust::for_each(counting, counting + n, [=]__device__(int i){
			int node = d_node_index[i];
			int best_attribute = d_current_nodes[node].attributeIndex;

			d_sort_key[i] = d_attributes[best_attribute][d_sample_index[i]];
		});

		//Two level segmented sort
		thrust::stable_sort_by_key(sort_key.begin(), sort_key.end(), thrust::make_zip_iterator(thrust::make_tuple(sample_index.begin(), node_index.begin())));
		thrust::stable_sort_by_key(node_index.begin(), node_index.end(), thrust::make_zip_iterator(thrust::make_tuple(sample_index.begin(), sort_key.begin())));

		//Calculate new node boundaries


		//Use segmented reduction to find split points
		thrust::reduce_by_key(node_index.begin(), node_index.end(), counting, discard, node_start_index.begin() + n_nodes, thrust::equal_to<int>(), [=]__device__(int i1, int i2){
			int node = d_node_index[i1];
			float split_value = d_current_nodes[node].attributeValue;
			float value1 = d_sort_key[i1];
			float value2 = d_sort_key[i2];

			if (value2 < split_value || value1 < split_value){
				return i2;
			}
			else{
				return i1;
			}

		});

		//Sort node start indices
		thrust::sort(node_start_index.begin(), node_start_index.end());

		//Populate node indices
		thrust::lower_bound(node_start_index.begin(), node_start_index.end(), counting, counting + n, node_index.begin(), thrust::less_equal<int >());


		//Correct indices to start at zero
		thrust::for_each(counting, counting + n, [=]__device__(int i){
			d_node_index[i] -= 1;
		});


		/*
		print(sample_index);
		print(node_index);
		print(sort_key);
		*/
	}

	printTree(decision_tree, data);

	safe_cuda(cudaFree(d_attributes));
	delete[] h_attributes;


}
int main(int argc, char **argv)
{

	if (argc != 3){
		std::cout << "usage: GPUTree.exe <filename.arff> <n levels>\n";
	}

	std::string filename(argv[1]);
	int n_levels = std::atoi(argv[2]);

	Data data(filename);

	try{
		Timer t;
		infogainSplit(data, n_levels);
		t.printElapsed("Tree build time");
	}
	catch (thrust::system_error &e){
		std::cerr << e.what() << "\n";

	}
	return 0;
}


