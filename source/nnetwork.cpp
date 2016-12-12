#include <iterator>
#include "nnetwork.h"
#include "inputlayer.h"
#include "sigmoidlayer.h"

void NNetwork::Forward(double* input, int inputSize)
{
	INNetworkLayer* previousLayer = nullptr;
	INNetworkLayer* nextLayer = nullptr;
	for (int index = 0; index < layers.size(); index++)
	{
		INNetworkLayer* layer = layers.at(index);
		if (index + 1 < layers.size())
		{
			nextLayer = layers.at(index + 1);
		} 
		else
		{
			nextLayer = nullptr;
		}

		if (index == 0)
		{
			layer->Forward(input, inputSize);
		}
		else
		{
			layer->Forward(previousLayer, nextLayer);
		}

		previousLayer = layer;
	}
}

void NNetwork::Backward(double* expected, int expectedSize, double learnRate)
{
	INNetworkLayer* nextLayer = nullptr;
	for (int index = (int)(layers.size() - 1); index > 0; index--)
	{
		INNetworkLayer* layer = layers.at(index);
		INNetworkLayer* previousLayer = layers.at(index - 1);

		if (index == layers.size() - 1)
		{
			layer->Backward(expected, expectedSize, learnRate);
		}
		else
		{
			layer->Backward(previousLayer, nextLayer, learnRate);
		}

		nextLayer = layer;
	}
}

double* NNetwork::GetLayerForward(int layerIndex)
{
	return layers.at(layerIndex)->GetForward();
}

double* NNetwork::GetLayerBackward(int layerIndex)
{
	return layers.at(layerIndex)->GetBackward();
}

void NNetwork::Dispose()
{
	while (!layers.empty())
	{
		layers.pop_back();
	}
}