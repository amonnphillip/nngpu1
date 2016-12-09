#pragma once
#include <memory>
#include <cassert>
#include <stdexcept>
#include "cuda_runtime.h"

template<typename nodetype, typename inputtype, typename forwardtype, typename backwardtype>
class Layer
{
protected:
	std::string layerName = "";
	std::unique_ptr<nodetype> nodeHostMem;
	std::unique_ptr<forwardtype> forwardHostMem;
	std::unique_ptr<backwardtype> backwardHostMem;
	nodetype *nodeDeviceMem = nullptr;
	forwardtype *forwardDeviceMem = nullptr;
	backwardtype *backwardDeviceMem = nullptr;
	inputtype *inputDeviceMem = nullptr;

	void Initialize(std::string layerName, int forwardSize, int backwardSize, int nodeSize, int inputSize, bool hasDeviceMemory)
	{
		assert(layerName.length() != 0);
		assert(nodeHostMem.get() == nullptr);
		assert(forwardHostMem.get() == nullptr);
		assert(backwardHostMem.get() == nullptr);
		assert(nodeDeviceMem == nullptr);
		assert(forwardDeviceMem == nullptr);
		assert(backwardDeviceMem == nullptr);
		assert(inputDeviceMem == nullptr);

		this->layerName = layerName;

		nodeHostMem = std::unique_ptr<nodetype>(new nodetype[nodeSize]);
		forwardHostMem = std::unique_ptr<forwardtype>(new forwardtype[forwardSize]);
		backwardHostMem = std::unique_ptr<backwardtype>(new backwardtype[backwardSize]);

		if (hasDeviceMemory)
		{
			if (nodeSize != 0)
			{
				if (cudaMalloc((void**)&nodeDeviceMem, nodeSize * sizeof(nodetype)) != cudaError::cudaSuccess)
				{
					throw std::bad_alloc();
				}
			}

			if (forwardSize != 0)
			{
				if (cudaMalloc((void**)&forwardDeviceMem, forwardSize * sizeof(forwardtype)) != cudaError::cudaSuccess)
				{
					throw std::bad_alloc();
				}
			}

			if (backwardSize != 0)
			{
				if (cudaMalloc((void**)&backwardDeviceMem, backwardSize * sizeof(backwardtype)) != cudaError::cudaSuccess)
				{
					throw std::bad_alloc();
				}
			}

			if (inputSize != 0)
			{
				if (cudaMalloc((void**)&inputDeviceMem, inputSize * sizeof(inputtype)) != cudaError::cudaSuccess)
				{
					throw std::bad_alloc();
				}
			}
		}
	}

	virtual void Dispose()
	{
		if (nodeDeviceMem != nullptr)
		{
			if (cudaFree(nodeDeviceMem) != cudaError::cudaSuccess)
			{
				throw std::bad_alloc();
			}
		}

		if (forwardDeviceMem != nullptr)
		{
			if (cudaFree(forwardDeviceMem) != cudaError::cudaSuccess)
			{
				throw std::bad_alloc();
			}
		}

		if (backwardDeviceMem != nullptr)
		{
			if (cudaFree(backwardDeviceMem) != cudaError::cudaSuccess)
			{
				throw std::bad_alloc();
			}
		}

		if (inputDeviceMem != nullptr)
		{
			if (cudaFree(inputDeviceMem) != cudaError::cudaSuccess)
			{
				throw std::bad_alloc();
			}
		}
	}
};