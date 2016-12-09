#include "sigmoidlayer.h"
#include "cuda_runtime.h"
#include "math.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdexcept>

__global__ void SigmoidLayer_Forward_cu(SigmoidNode *node, double *in, double *out)
{
	int weightCount = node[blockIdx.x].weightCount;
	double val = 0;
	for (int i = 0; i < weightCount; i++)
	{
		val += node[blockIdx.x].weights[i] * in[blockIdx.x];
	}

	out[blockIdx.x] = 1.0 / (1.0 + exp(-val));
}

__global__ void SigmoidLayer_Backward_cu(SigmoidNode *node, double *forward, double *in, double *out, double learnRate)
{
	double error = forward[blockIdx.x] * (1 - forward[blockIdx.x]) * in[blockIdx.x];

	int weightCount = node[blockIdx.x].weightCount;
	double val = 0;
	for (int i = 0; i < weightCount; i++)
	{
		node[blockIdx.x].weights[i] += error * forward[i] * learnRate;
	}

	out[blockIdx.x] = 1.0 / (1.0 + exp(-val));
}

void SigmoidLayer_Forward(SigmoidNode *node, double *input, double *output, int nodeCount)
{
	SigmoidLayer_Forward_cu<<<nodeCount, 1>>>(node, input, output);

	if (cudaGetLastError() != cudaError::cudaSuccess)
	{
		throw std::runtime_error("Sigmoid Forward CUDA method returned an error");
	}

	if (cudaDeviceSynchronize() != cudaError::cudaSuccess)
	{
		throw std::runtime_error("Sigmoid Forward CUDA syncronize returned an error");
	}
}

void SigmoidLayer_Backward(SigmoidNode *node, double *forward, double *input, double *output, int nodeCount, double learnRate)
{
	SigmoidLayer_Backward_cu<<<nodeCount, 1>>>(node, forward, input, output, learnRate);

	if (cudaGetLastError() != cudaError::cudaSuccess)
	{
		throw std::runtime_error("Sigmoid Forward CUDA method returned an error");
	}

	if (cudaDeviceSynchronize() != cudaError::cudaSuccess)
	{
		throw std::runtime_error("Sigmoid Forward CUDA syncronize returned an error");
	}
}

