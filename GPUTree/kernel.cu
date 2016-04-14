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

#include <thrust/gather.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/binary_search.h>

#include <cub/cub.cuh>


struct CustomScan{

	int *d_node_index;
	int *d_sort_order;
	int *d_tmp_classes;
	int n_instances;

	CustomScan(int *d_node_index, int *d_sort_order, int *d_tmp_classes,int n_instances) :d_node_index(d_node_index), d_sort_order(d_sort_order),d_tmp_classes(d_tmp_classes), n_instances(n_instances){}

	__device__ CUB_RUNTIME_FUNCTION __forceinline__
	int operator()(const int & left, const int &right)const {
		int left_node = d_node_index[left];
		int right_node = d_node_index[right];

		//Are left and right in different nodes?
		if (left_node != right_node){
			return d_tmp_classes[right];
		}

		//Are left and right in different attributes?
		if ((d_sort_order[left] / n_instances ) != (d_sort_order[right]/n_instances)){
			return d_tmp_classes[right];
		}
		
		return d_tmp_classes[left] + d_tmp_classes[right];
	}
};

__device__ float entropy(int n_a, int n){

	int n_b = n - n_a;
	if (n_a == 0 || n_a == n || n == 0){
		return 0;
	}
	float p_a = (float)n_a / (n_a + n_b);
	float p_b = 1 - p_a;
	return -(p_a * std::log2(p_a)) - (p_b * std::log2(p_b));
}

void infogainSplit(Data&data, int n_levels){

	int n_attributes = data.attributes.size();
	int n_instances = data.classes.size();
	int n = n_attributes * n_instances;

	int total_nodes = std::pow(2, n_levels) - 1;
	thrust::device_vector<  TreeNode  >  decision_tree(total_nodes);

	thrust::device_vector<float > attributes(n );
	thrust::device_vector<int  > classes(n );
	thrust::device_vector<int  > tmp_classes(n );

	thrust::device_vector<int  > node_index(n,0 );

	thrust::host_vector<int> h_offsets(n_attributes+1);
	h_offsets[n_attributes] = n;
	thrust::device_vector<int> offsets(n_attributes+1);

	//Copy attributes/class & calculate offsets
	for (int i = 0; i < n_attributes; i++){
		thrust::copy(data.attributes[i].begin(), data.attributes[i].end(), attributes.begin() + n_instances * i);
		thrust::copy(data.classes.begin(), data.classes.end(), classes.begin() + n_instances * i);
		h_offsets[i] = i * n_instances;
	}

	offsets = h_offsets;

	//Perform sort
	thrust::device_vector<int > sort_order(n);
	thrust::device_vector<int >  indices(n);
	//We will temporarily use this as the output sorted attributes but later to hold infogain
	thrust::device_vector<float >   info(n);
	thrust::sequence(indices.begin(), indices.end());

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

	//Node offsets
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

		//We can infer the attribute index from the sort order
		auto getAttributeIndex = [=]__device__(int index){
			return d_sort_order[index] / n_instances;
		};
		
		cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, raw(tmp_classes), raw(tmp_classes), n);

		t.printElapsed("scan classes");
		t.reset();

		//Calculate infogain
		int *d_tmp_classes = raw(tmp_classes);
		int *d_node_positive_count = raw(node_positive_count);

		auto infogain = [=]__device__(int i){

			//If this attribute value is the same as the next attribute to the left then just return 0
			//We cannot use this as a split point
			if (i > 0){
				if (attribute_perm[i] == attribute_perm[i - 1]){
					return 0.0f;
				}
			}

			//Current node of the tree we are in
			int node = d_node_index[i];

			//Total instances in this node
			int node_total = (d_node_offsets[node+1]-d_node_offsets[node])/n_attributes;

			//Total positive instances in this node
			int node_positive = d_node_positive_count[node]/n_attributes;

			//The index of the current instance within this node
			int instance_i = (i - d_node_offsets[node]) % node_total;

			float node_entropy = entropy(node_positive, node_total);

			//Index of the start of this segment
			int  attribute_start = d_node_offsets[node] + getAttributeIndex(i) * node_total;

			int left_positive = d_tmp_classes[i] - d_tmp_classes[attribute_start];
			int left_negative = instance_i - left_positive;
			int right_positive = node_positive - left_positive;
			int right_negative = node_total - left_positive - left_negative - right_positive;

			float entropy_left = entropy(left_positive, left_negative + left_positive);
			float entropy_right = entropy(right_positive, right_negative + right_positive);

			return node_entropy - ((float)instance_i / node_total) * entropy_left - ((float)(node_total - instance_i) / node_total) * entropy_right;

		};

		float *d_info = raw(info);
		thrust::for_each(counting, counting + n, [=]__device__(int i){
			d_info[i] = infogain(i);
		});

		thrust::reduce_by_key(node_index.begin(), node_index.end(), counting, discard, node_best_infogain_index.begin(), thrust::equal_to<int>(), [=]__device__(int i1, int i2){
			return d_info[i1] > d_info[i2] ? i1 : i2;
		});
		t.printElapsed("infogain reduce");
		t.reset();

		//Create a node for best splits
		int *d_node_best_infogain_index = raw(node_best_infogain_index);
		thrust::for_each(counting, counting + n_nodes, [=]__device__(int i){
			int  best_index = d_node_best_infogain_index[i];
			//Use the weka method where split point is the mean of two nodes
			int lower_index =  best_index - 1 >= 0 ? best_index - 1 : 0;
			float attribute_split = (attribute_perm[best_index] + attribute_perm[lower_index]) / 2.0;
			int attribute_index = getAttributeIndex(best_index);

			d_current_nodes[i] = TreeNode(attribute_index, attribute_split, d_info[best_index]);

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
			//Total instances in this node
			int node_total = (d_node_offsets[i+1]-d_node_offsets[i])/n_attributes;
			//Instances to the left of the split point
			int instances_left = best_index - (d_node_offsets[i] + d_current_nodes[i].attributeIndex * node_total);

			//Write into array 1 place to the right so we dont have a read/write conflict - the sort will put it into correct place later
			d_new_node_offsets[i+1] = d_node_offsets[i] + instances_left * n_attributes;
		});

		t.printElapsed("calculate node offsets");
		t.reset();
		//Sort new offsets into place
		thrust::sort(node_offsets.begin(), node_offsets.end());

		t.printElapsed("sort offsets");
		t.reset();
		//Populate node indices
		thrust::lower_bound(node_offsets.begin(), node_offsets.end(), counting, counting + n, node_index.begin(), thrust::less_equal<int >());

		//Correct indices to start at zero
		thrust::for_each(counting, counting + n, [=]__device__(int i){
			d_node_index[i] -= 1;
		});

		t.printElapsed("create node indices");
		t.reset();
	}

	printTree(decision_tree, data);
	safe_cuda(cudaFree(d_temp_storage));

}

int main(int argc, char **argv)
{

	if (argc != 3){
		std::cout << "usage: GPUTree.exe <filename.arff> <n levels>\n";
	}

	std::string filename(argv[1]);
	int n_levels = std::atoi(argv[2]);

	//Force cuda context initialisation
	safe_cuda(cudaFree(0));

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


