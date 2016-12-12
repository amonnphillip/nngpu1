#pragma once
#include "innetworklayer.h"
#include <cassert>
#include <vector>

class NNetwork
{
private:
	std::vector<INNetworkLayer*> layers;

public:
	template<class layertype> 
	void Add(int dimentionx, int dimentiony, int dimentionz)
	{
		layertype* layer = new layertype(dimentionx, dimentiony, dimentionz);
		layers.push_back(layer);
	}
	void Forward(double* input, int inputSize);
	void Backward(double* input, int expectedSize, double learnRate);
	double* GetLayerForward(int layerIndex);
	double* GetLayerBackward(int layerIndex);
	void Dispose();

};
