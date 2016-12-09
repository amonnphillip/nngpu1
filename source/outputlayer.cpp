#include "outputlayer.h"
#include "layerexception.h"
#include "cuda_runtime.h"
#include "layer.h"
#include <cassert>

OutputLayer::OutputLayer(int dimentionx, int dimentiony, int dimentionz)
{
	nodeCount = dimentionx * dimentiony * dimentionz;
	Layer::Initialize(
		"output",
		nodeCount,
		nodeCount,
		0,
		0,
		false);
}

void OutputLayer::Dispose()
{
	Layer::Dispose();
}

void OutputLayer::Forward(double* input, int inputSize)
{
	throw LayerException("Forward variant not valid for OutputLayer layer");
}

void OutputLayer::Forward(INNetworkLayer* previousLayer, INNetworkLayer* nextLayer)
{
	assert(previousLayer->GetForwardNodeCount() == nodeCount);

	memcpy(forwardHostMem.get(), previousLayer->GetForward(), nodeCount * sizeof(double));
}

void OutputLayer::Backward(double* input, int inputSize, double learnRate)
{
	assert(inputSize == nodeCount);

	double* forward = forwardHostMem.get();
	double* backward = backwardHostMem.get();
	for (int index = 0; index < nodeCount; index++)
	{
		*backward = *input - *forward;
		forward++;
		input++;
		backward++;
	}
}

void OutputLayer::Backward(INNetworkLayer* previousLayer, INNetworkLayer* nextLayer, double learnRate)
{
	throw LayerException("Backward variant not valid for OutputLayer layer");
}

double* OutputLayer::GetForward()
{
	return forwardHostMem.get();;
}

double* OutputLayer::GetBackward()
{
	return backwardHostMem.get();
}

int OutputLayer::GetForwardNodeCount()
{
	return nodeCount;
}

int OutputLayer::GetBackwardNodeCount()
{
	return nodeCount;
}