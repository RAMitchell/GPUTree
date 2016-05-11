#pragma once

#include "cuda_runtime.h"

#ifdef WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT 
#endif

struct TreeNode{
	int attributeIndex;
	float attributeValue;
	float infogain;

	//The probability of a class being positive given it is < attributeValue
	float left_prob;
	//The probability of a class being positive given it is >= attributeValue
	float  right_prob;

	__host__ __device__ TreeNode() :attributeIndex( 0 ), attributeValue(0), infogain(0), left_prob(0),right_prob(0){};

	__host__ __device__ TreeNode(int attributeIndex, float attributeValue, float infogain, float  left_prob, float  right_prob) : attributeIndex(attributeIndex), attributeValue(attributeValue), infogain(infogain),  left_prob( left_prob), right_prob( right_prob){};

};




//Exported functions
extern "C"{
	//Generates an array of tree nodes
	EXPORT int  generate_tree(const float * in_attributes, const int n_attributes, const char * in_classes, const int n_instances, const int n_levels, TreeNode*out_tree);

	//Tests for the presence of a Nvidia GPU
	//Also forces initialisation of the cuda runtime
	EXPORT bool test_cuda();
}

