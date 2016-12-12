#include "sigmoidlayer.h"
#include "layerexception.h"
#include <cassert>
#include "cuda_runtime.h"
#include "layer.h"

extern void SigmoidLayer_Forward(SigmoidNode *node, double *input, double *output, int nodeCount);
extern void SigmoidLayer_Backward(SigmoidNode *node, double* forward, double *input, double *output, int nodeCount, double learnRate);

SigmoidLayer::SigmoidLayer(int dimentionx, int dimentiony, int dimentionz)
{
	nodeCount = dimentionx * dimentiony * dimentionz;
	Layer::Initialize(
		"sigmoid",
		nodeCount,
		nodeCount,
		nodeCount,
		nodeCount,
		true);

	SigmoidNode* hnodes = nodeHostMem.get();
	for (int index = 0; index < nodeCount; index++)
	{
		hnodes->weightCount = 2;
		hnodes->weights[0] = 1; // We make the assumtion here that the previous layer has 2 nodes
		hnodes->weights[1] = 1;
		hnodes++;
	}

	std::memset(forwardHostMem.get(), 0, nodeCount * sizeof(double));

	std::memset(backwardHostMem.get(), 0, nodeCount * sizeof(double));

	if (cudaMemcpy(nodeDeviceMem, nodeHostMem.get(), nodeCount * sizeof(SigmoidNode), cudaMemcpyHostToDevice) != cudaError::cudaSuccess)
	{
		throw std::runtime_error("Sigmoid cudaMemcpy returned an error");
	}

	if (cudaMemcpy(forwardDeviceMem, forwardHostMem.get(), nodeCount * sizeof(double), cudaMemcpyHostToDevice) != cudaError::cudaSuccess)
	{
		throw std::runtime_error("Sigmoid cudaMemcpy returned an error");
	}
}

void SigmoidLayer::Dispose()
{
	Layer::Dispose();
}

void SigmoidLayer::Forward(double* input, int inputSize)
{
	throw LayerException("Forward variant not valid for Sigmoid layer");
}

void SigmoidLayer::Forward(INNetworkLayer* previousLayer, INNetworkLayer* nextLayer)
{
	if (cudaMemcpy(inputDeviceMem, previousLayer->GetForward(), nodeCount * sizeof(double), cudaMemcpyHostToDevice) != cudaError::cudaSuccess)
	{
		throw std::runtime_error("Sigmoid forward cudaMemcpy returned an error");
	}

	SigmoidLayer_Forward(nodeDeviceMem, inputDeviceMem, forwardDeviceMem, nodeCount);

	if (cudaMemcpy(forwardHostMem.get(), forwardDeviceMem, nodeCount * sizeof(double), cudaMemcpyDeviceToHost) != cudaError::cudaSuccess)
	{
		throw std::runtime_error("Sigmoid forward cudaMemcpy returned an error");
	}
}

void SigmoidLayer::Backward(double* input, int inputSize, double learnRate)
{
	throw LayerException("Backward variant not valid for Sigmoid layer");
}

void SigmoidLayer::Backward(INNetworkLayer* previousLayer, INNetworkLayer* nextLayer, double learnRate)
{
	if (cudaMemcpy(inputDeviceMem, nextLayer->GetBackward(), nodeCount * sizeof(double), cudaMemcpyHostToDevice) != cudaError::cudaSuccess)
	{
		throw std::runtime_error("Sigmoid backward cudaMemcpy returned an error");
	}

	SigmoidLayer_Backward(nodeDeviceMem, forwardDeviceMem, inputDeviceMem, backwardDeviceMem, nodeCount, learnRate);

	if (cudaMemcpy(backwardHostMem.get(), backwardDeviceMem, nodeCount * sizeof(double), cudaMemcpyDeviceToHost) != cudaError::cudaSuccess)
	{
		throw std::runtime_error("Sigmoid backward cudaMemcpy returned an error");
	}
}

double* SigmoidLayer::GetForward()
{
	return forwardHostMem.get();
}

double* SigmoidLayer::GetBackward()
{
	return backwardHostMem.get();
}

int SigmoidLayer::GetForwardNodeCount()
{
	return nodeCount;
}

int SigmoidLayer::GetBackwardNodeCount()
{
	return nodeCount;
}