#pragma once
#include "data.h"
#include <thrust/device_vector.h>

struct TreeNode{
	int attributeIndex;
	float attributeValue;
	float infogain;

	__host__ __device__ TreeNode() :attributeIndex( 0 ), attributeValue(0), infogain(0){};

	__host__ __device__ TreeNode(int attributeIndex, float attributeValue, float infogain) : attributeIndex(attributeIndex), attributeValue(attributeValue), infogain(infogain){};

};

void printTreeRecurse(thrust::host_vector< TreeNode> &tree, int index, int level, Data&d);

void printTree(thrust::device_vector< TreeNode> &d_tree, Data&d);
