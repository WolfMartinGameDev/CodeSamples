#include "ClothSimulation.h"
#include <cuda_gl_interop.h>
#include <framework/cuda/CheckError.h>

#include <vector>

ClothSimulation::ClothSimulation(int gridDim, float sizeX, float sizeZ, loat clothMass, float damping, float gravity, int fixUpIterations, float fixUpPercent) :
	damping(damping),
	gravity(gravity)
{
	std::vector<int> fixedNodes;
	std::vector<float4> obstacles;

	for (int i = 0; i < 3; ++i)
	{
		positionsBuffers[i] = nullptr;
	}
	
	newCloth(gridDim, sizeX, sizeZ, clothMass, fixUpIterations, fixUpPercent, fixedNodes, obstacles);
}

ClothSimulation::~ClothSimulation()
{
	for (int i = 0; i < 3; ++i)
	{
		if (positionsBuffers[i] != nullptr)
		{
			checkCudaError(cudaGraphicsUnregisterResource(positionsBuffers[i]));
		}
	}
}

void ClothSimulation::mapBuffers()
{
	checkCudaError(cudaGraphicsMapResources(3, positionsBuffers));

	size_t numBytes;
	
	for (int i = 0; i < 3; ++i)
	{
		checkCudaError(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&mappedPositions[i]), &numBytes, positionsBuffers[i]));
	}
}

void ClothSimulation::unmapBuffers()
{
	checkCudaError(cudaGraphicsUnmapResources(3, positionsBuffers));
}

void ClothSimulation::registerBuffers(GL::Buffer buffers[3])
{
	for (int i = 0; i < 3; ++i)
	{
		if (positionsBuffers[i] != nullptr)
		{
			checkCudaError(cudaGraphicsUnregisterResource(positionsBuffers[i]));
		}
	}
	
	for (int i = 0; i < 3; ++i)
	{
		checkCudaError(cudaGraphicsGLRegisterBuffer(&positionsBuffers[i], buffers[i], cudaGraphicsMapFlagsNone));
	}
}