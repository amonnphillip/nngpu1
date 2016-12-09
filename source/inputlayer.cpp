#include "inputlayer.h"
#include "layerexception.h"
#include <cassert>
#include "cuda_runtime.h"
#include "layer.h"

InputLayer::InputLayer(int dimentionx, int dimentiony, int dimentionz)
{
	nodeCount = dimentionx * dimentiony * dimentionz;
	Layer::Initialize(
		"input",
		nodeCount,
		0,
		0,
		0,
		false);
}

void InputLayer::Dispose()
{
	Layer::Dispose();
}

void InputLayer::Forward(double* input, int inputSize)
{
	assert(inputSize == nodeCount);

	memcpy(forwardHostMem.get(), input, nodeCount * sizeof(double));
}

void InputLayer::Forward(INNetworkLayer* previousLayer, INNetworkLayer* nextLayer)
{
	throw LayerException("Forward variant not valid for InputLayer layer");
}

void InputLayer::Backward(double* input, int inputSize, double learnRate)
{
	throw LayerException("Backward variant not valid for InputLayer layer");
}

void InputLayer::Backward(INNetworkLayer* previousLayer, INNetworkLayer* nextLayer, double learnRate)
{
	
}

double* InputLayer::GetForward()
{
	return forwardHostMem.get();
}

double* InputLayer::GetBackward()
{
	return nullptr;
}

int InputLayer::GetForwardNodeCount()
{
	return nodeCount;
}

int InputLayer::GetBackwardNodeCount()
{
	return 0;
}