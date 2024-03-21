#ifndef CLOTHSIMULATION_H_POW5AKL9

#define CLOTHSIMULATION_H_POW5AKL9

#pragma once

#include "GL/buffer.h"
#include <cuda_runtime.h>
#include <vector>

class ClothSimulation
{
	int gridDim;
	int fixUpIterations;
	float sizeX, sizeZ;
	float invParticleMass;
	float fixUpPercent;
	float damping;
	float gravity;

	float lengthStructural;
	float lengthShear;
	float lengthFlexion;

	cudaGraphicsResource_t positionsBuffers[3];
	float4* mappedPositions[3];

	int* dptrFixedNodes = nullptr;
	int* fixedNodesArray = nullptr;

	float4* dptrObstacles = nullptr;
	float4* obstaclesArray = nullptr;

public:
	ClothSimulation(int gridDim, float sizeX, float sizeZ, float clothMass, float damping, float gravity, int fixUpIterations, float fixUpPercent);
	~ClothSimulation();
	void mapBuffers();
	void unmapBuffers();
	void simulate(float dt, bool windChanged, float windX, float windY, float windZ, int& lastPositionBuffer);
	void newCloth(int gridDim, float sizeX, float sizeZ, float clothMass, int fixUpIterations, float fixUpPercent, std::vector<int> fixedNodes, std::vector<float4> obstacles);
	void registerBuffers(GL::Buffer buffers[3]);
	void copyPositionsToCPU(float4* cpuBuffer, int currentPositionBuffer);
};

#endif 