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
	float * distribution;
	int * counts;
};




//Exported functions
extern "C"{
	//Generates an array of tree nodes
	EXPORT int  generate_tree(const float * in_attributes, const int n_attributes, const char * in_classes, const int n_instances,const int n_class_values, const int n_levels, TreeNode*out_tree);

	//Tests for the presence of a Nvidia GPU
	EXPORT bool test_cuda();
	
	//Call a dummy cuda function to force run-time initialisation
	EXPORT void force_context_init();
}

