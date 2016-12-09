#include "cuda_runtime.h"
#include "sigmoidlayer.h"
#include "nnetwork.h"
#include "inputlayer.h"
#include "sigmoidlayer.h"
#include "outputlayer.h"
#include <iostream>

int main()
{
	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);

	// Create the (very small) network
	NNetwork* nn = new NNetwork();
	nn->Add<InputLayer>(2, 1, 1);
	nn->Add<SigmoidLayer>(2, 1, 1);
	nn->Add<OutputLayer>(2, 1, 1);

	// Train the network
	int interations = 10000;
	while (interations > 0)
	{
		double input[] = { 1, 1 };
		nn->Forward(input, 2);

		double* nnoutput = nn->GetLayerForward(1);

		double expected[] = { 1, 0 };
		nn->Backward(expected, 2, 0.1);


		std::cout << "output:\r\n";
		for (int index = 0; index < 2; index++)
		{
			std::cout << nnoutput[index] << " ";
		}
		std::cout << "\r\n";

		nnoutput = nn->GetLayerBackward(2);
		std::cout << "error:\r\n";
		for (int index = 0; index < 2; index++)
		{
			std::cout << nnoutput[index] << " ";
		}
		std::cout << "\r\n\r\n";

		interations--;
	}
	
	// Dispose of the resouces we allocated and close
	nn->Dispose();
	delete nn;

	cudaDeviceReset();
}