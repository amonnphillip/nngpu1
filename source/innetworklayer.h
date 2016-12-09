#pragma once

class INNetworkLayer
{
public:
	virtual void Forward(double* input, int inputSize) = 0;
	virtual void Forward(INNetworkLayer* previousLayer, INNetworkLayer* nextLayer) = 0;
	virtual void Backward(double* input, int inputSize, double learnRate) = 0;
	virtual void Backward(INNetworkLayer* previousLayer, INNetworkLayer* nextLayer, double learnRate) = 0;
	virtual void Dispose() = 0;
	virtual double* GetForward() = 0;
	virtual double* GetBackward() = 0;
	virtual int GetForwardNodeCount() = 0;
	virtual int GetBackwardNodeCount() = 0;
};