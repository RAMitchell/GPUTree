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

#include <cub/cub.cuh>


void infogainSplit(Data&data, int n_levels){

	int n_attributes = data.attributes.size();
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

	thrust::device_vector<int > sample_index_buf(n);
	thrust::device_vector<int > sample_index_alt_buf(n);
	cub::DoubleBuffer<int> db_sample_index(raw(sample_index_buf), raw(sample_index_alt_buf));

	//Populate indices
	thrust::sequence(dptr(db_sample_index.Current()), dptr(db_sample_index.Current())+n);

	thrust::device_vector<float > sort_key_buf(n);
	thrust::device_vector<float > sort_key_alt_buf(n);
	cub::DoubleBuffer<float > db_sort_key(raw(sort_key_buf), raw(sort_key_alt_buf));

	thrust::device_vector<int > tmp_classes(n);
	thrust::device_vector<int >  node_index(n, 0);
	int *d_node_index = raw(node_index);

	//Max number of nodes at any given level
	int max_nodes = std::pow(2, n_levels - 1);
	thrust::device_vector<int >  node_positive_count(max_nodes);
	thrust::device_vector<int >  node_total_count(max_nodes);
	thrust::device_vector<int >  node_start_index(max_nodes+1,n);
	node_start_index[0] = 0;
	thrust::device_vector< int  >  node_best_infogain_index(max_nodes);

	//Total nodes
	int total_nodes = std::pow(2, n_levels) - 1;
	thrust::device_vector<  TreeNode  >  decision_tree(total_nodes);


	//Make iterators
	auto discard = thrust::make_discard_iterator();
	auto counting = thrust::make_counting_iterator<int >(0);
	auto constant_one = thrust::make_constant_iterator<int >(1);

	//Allocate temporary storage for radix sort
	void *d_temp_storage = NULL;
	size_t temp_storage_bytes = 0;
	cub::DeviceSegmentedRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, db_sort_key, db_sample_index, n, max_nodes, raw( node_start_index), raw( node_start_index)+1);
	safe_cuda(cudaMalloc(&d_temp_storage, temp_storage_bytes));

	for (int level = 0; level < n_levels; level++){
		//Number of nodes at this level
		int n_nodes = 1 << level;

		//Get pointer to current level's nodes
		TreeNode* d_current_nodes = raw(decision_tree) + ((1 << level) - 1);

		//Calculate Max infogain
		for (int attribute = 0; attribute < n_attributes; attribute++){

			calculate_infogain(n, db_sample_index, attributes, db_sort_key, node_index, classes, tmp_classes, node_positive_count, node_total_count, node_start_index, node_best_infogain_index, d_current_nodes, attribute, n_nodes, d_temp_storage, temp_storage_bytes);
		}

		//Sort values within each node according to attribute with highest infogain

		//Copy attributes
		{
			float *d_sort_key = db_sort_key.Current();
			int *d_sample_index = db_sample_index.Current();
			thrust::for_each(counting, counting + n, [=]__device__(int i){
				int node = d_node_index[i];
				int best_attribute = d_current_nodes[node].attributeIndex;

				d_sort_key[i] = d_attributes[best_attribute][d_sample_index[i]];
			});
		}


		//Segmented sort
		cub::DeviceSegmentedRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, db_sort_key, db_sample_index, n, n_nodes, raw( node_start_index), raw( node_start_index)+1);

		//Calculate new node boundaries

		//Use segmented reduction to find split points
		float*d_sort_key = db_sort_key.Current();
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



	}

	printTree(decision_tree, data);

	safe_cuda(cudaFree(d_attributes));
	safe_cuda(cudaFree(d_temp_storage));
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


