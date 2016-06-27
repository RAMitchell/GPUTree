#GPUTree

Implements a Weka package for fast decision tree construction.

Currently supports numerical attributes and nominal class.

GPU algorithm is implemented as a native CUDA library which is then called by Weka using JNA.

Will require a CUDA capable GPU to run.
