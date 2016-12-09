#pragma once

#include "innetworklayer.h"
#include "layer.h"

struct SigmoidNode
{
	static const int maxWeights = 2;
	int weightCount = 0;
	double weights[maxWeights];
};

class SigmoidLayer : public Layer<SigmoidNode, double, double, double>, public INNetworkLayer
{
public:
	int nodeCount = 0;
	int forwardCount = 0;
	int backwardCount = 0;
	int inputCount = 0;

public:
	SigmoidLayer(int dimentionx, int dimentiony, int dimentionz);
	virtual void Dispose();
	virtual void Forward(double* input, int inputSize);
	virtual void Forward(INNetworkLayer* previousLayer, INNetworkLayer* nextLayer);
	virtual void Backward(double* input, int inputSize, double learnRate);
	virtual void Backward(INNetworkLayer* previousLayer, INNetworkLayer* nextLayer, double learnRate);
	virtual double* GetForward();
	virtual double* GetBackward();
	virtual int GetForwardNodeCount();
	virtual int GetBackwardNodeCount();
};
