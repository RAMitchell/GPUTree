#include <iostream>
#include "data.h"
#include "cuda_helpers.cuh"
#include "GPUTree.cuh"

void printTreeRecurse(std::vector< TreeNode> &tree, int index, int level, data&d){

	if (index >= tree.size()){
		return;
	}

	for (int i = 0; i < level; i++){
		std::cout << "\t";
	}

	if (tree[index].infogain == 0){
		std::cout << "<-->\n";

	}
	else{
		std::cout << "<" << d.attribute_names[tree[index].attributeIndex] << ", " << tree[index].attributeValue << ", " << tree[index].infogain << ">\n";
	}

	printTreeRecurse(tree, index * 2 + 1, level + 1,d);
	printTreeRecurse(tree, index * 2 + 2, level + 1,d);


}

void printTree(std::vector<TreeNode> &tree, data&d){
	std::cout << "<attribute, attribute value, infogain>\n";
	printTreeRecurse(tree, 0, 0,d);
}

int main(int argc, char **argv)
{

	if (argc != 3){
		std::cout << "usage: GPUTree.exe <filename.arff> <n levels>\n";
		return 0;
	}

	std::string filename(argv[1]);
	int n_levels = std::atoi(argv[2]);

	//Force cuda context initialisation
	safe_cuda(cudaFree(0));

	data d(filename);

	double ** all_attributes = new double*[d.attributes.size()];
	for (int i = 0; i < d.attributes.size(); i++){
		all_attributes[i] = d.attributes[i].data();
	}

	int max_nodes = std::pow(2, n_levels) - 1;
	std::vector<TreeNode> tree(max_nodes);

	try{
		Timer t;
		generate_tree_weka( all_attributes, d.attributes.size(), d.attributes[0].size(), d.class_index, n_levels, tree.data());
		cudaDeviceSynchronize();
		t.printElapsed("Tree build time");
	}
	catch (thrust::system_error &e){
		std::cerr << e.what() << "\n";

	}

	printTree(tree, d);

	delete[]  all_attributes;

	return 0;
}
