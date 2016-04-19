#include "TreeNode.h"

void printTreeRecurse(thrust::host_vector< TreeNode> &tree, int index, int level, Data&d){

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

void printTree(thrust::device_vector< TreeNode> &d_tree, Data&d){
	std::cout << "<attribute, attribute value, infogain>\n";
	thrust::host_vector< TreeNode> tree = d_tree;
	printTreeRecurse(tree, 0, 0,d);
}
