#include "../ClothSimulation.h"
#include <framework/cuda/CheckError.h>
#include <framework/cuda/helper_math.h>

__constant__ float dGravity;
__constant__ float dDamping;
__constant__ float dFixupPercentage;
__constant__ int dGridDim;

__constant__ float dStickLengthStructural;
__constant__ float dStickLengthShear;
__constant__ float dStickLengthFlexion;

__constant__ float dInvParticleMass;
__constant__ float4 dWind;

__constant__ int dFixedNodesCount;
__constant__ int* dFixedNodes;

__constant__ float4* dObstacles;
__constant__ int dObstacleCount;

__device__ bool isFixed(const int myParticle, const int fixedCount, const int* __restrict fixedNodes)
{
	for (int i = 0; i < fixedCount; i++)
	{
		if (fixedNodes[i] == myParticle)
		{
			return true;
		}
	}
	
	return false;
}

__global__ void simulationStep(float dt, const float4* __restrict pLast, const float4* __restrict pCurrent, float4* pNext)
{
	int gridDim = dGridDim;
	
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	
	if (x >= gridDim || y >= gridDim)
	{
		return;
	}

	int myParticle = y * gridDim + x;
	float4 myParticlePos = pCurrent[myParticle];
	int fixedCount = dFixedNodesCount;

	if (isFixed(myParticle, fixedCount, dFixedNodes))
	{
		pNext[myParticle] = myParticlePos;
		return;
	}

	float damping = dDamping;
	float4 wind = dWind;

	if (length(wind) > 0.0f)
	{
		int horizontalNeighborLeft = -1, horizontalNeighborRight = -1, verticalNeighborUp = -1, verticalNeighborDown = -1;
		
		if (x > 0)
		{
			horizontalNeighborLeft = myParticle - 1;
		}
		
		if (x < gridDim - 1)
		{
			horizontalNeighborRight = myParticle + 1;
		}
		
		if (y > 0)
		{
			verticalNeighborUp = myParticle - gridDim;
		}
		
		if (y < gridDim - 1)
		{
			verticalNeighborDown = myParticle + gridDim;
		}
		
		float3 normal = make_float3(0.0f);
		
		if (horizontalNeighborLeft >= 0)
		{
			float3 horizontalNeighborPosDiff;
			
			{
				float4 horizontalNeighborPos = pCurrent[horizontalNeighborLeft];
				horizontalNeighborPosDiff = make_float3(horizontalNeighborPos.x - myParticlePos.x, horizontalNeighborPos.y - myParticlePos.y, horizontalNeighborPos.z - myParticlePos.z);
			}
			
			if (verticalNeighborUp >= 0)
			{
				float3 verticalNeighborPosDiff;
				
				{
					float4 verticalNeighborPos = pCurrent[verticalNeighborUp];
					verticalNeighborPosDiff = make_float3(verticalNeighborPos.x - myParticlePos.x, verticalNeighborPos.y - myParticlePos.y, verticalNeighborPos.z - myParticlePos.z);
				}
				
				normal -= normalize(cross(horizontalNeighborPosDiff, verticalNeighborPosDiff));
			}
			
			if (verticalNeighborDown >= 0)
			{
				float3 verticalNeighborPosDiff;
				
				{
					float4 verticalNeighborPos = pCurrent[verticalNeighborDown];
					verticalNeighborPosDiff = make_float3(verticalNeighborPos.x - myParticlePos.x, verticalNeighborPos.y - myParticlePos.y, verticalNeighborPos.z - myParticlePos.z);
				}
				
				normal += normalize(cross(horizontalNeighborPosDiff, verticalNeighborPosDiff));
			}
		}
		
		if (horizontalNeighborRight >= 0)
		{
			float3 horizontalNeighborPosDiff;
			
			{
				float4 horizontalNeighborPos = pCurrent[horizontalNeighborRight];
				horizontalNeighborPosDiff = make_float3(horizontalNeighborPos.x - myParticlePos.x, horizontalNeighborPos.y - myParticlePos.y, horizontalNeighborPos.z - myParticlePos.z);
			}
			
			if (verticalNeighborUp >= 0)
			{
				float3 verticalNeighborPosDiff;
				
				{
					float4 verticalNeighborPos = pCurrent[verticalNeighborUp];
					verticalNeighborPosDiff = make_float3(verticalNeighborPos.x - myParticlePos.x, verticalNeighborPos.y - myParticlePos.y, verticalNeighborPos.z - myParticlePos.z);
				}
				
				normal += normalize(cross(horizontalNeighborPosDiff, verticalNeighborPosDiff));
			}
			
			if (verticalNeighborDown >= 0)
			{
				float3 verticalNeighborPosDiff;
				
				{
					float4 verticalNeighborPos = pCurrent[verticalNeighborDown];
					verticalNeighborPosDiff = make_float3(verticalNeighborPos.x - myParticlePos.x, verticalNeighborPos.y - myParticlePos.y, verticalNeighborPos.z - myParticlePos.z);
				}
				
				normal -= normalize(cross(horizontalNeighborPosDiff, verticalNeighborPosDiff));
			}
		}

		normal = normalize(normal);
		
		wind = wind * dot(wind, make_float4(normal.x, normal.y, normal.z, 0.f));

		float invParticleMass = dInvParticleMass;
		
		if (x == 0 || x == gridDim - 1)
		{
			invParticleMass *= 2.0f;
		}
		
		if (y == 0 || y == gridDim - 1)
		{
			invParticleMass *= 2.0f;
		}
		
		wind *= invParticleMass;
	}

	pNext[myParticle] = (1 + damping) * myParticlePos - damping * pLast[myParticle] + make_float4(wind.x, dGravity + wind.y, wind.z, 0.0f) * (dt * dt);
}

__global__ void fixupStep(const float4* __restrict pNext, float4* pCalc)
{
	int gridDim = dGridDim;
	
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	
	if (x >= gridDim || y >= gridDim)
	{
		return;
	}

	int myParticle = y * gridDim + x;
	float4 myParticlePos = pNext[myParticle];

	int fixedCount = dFixedNodesCount;
	
	if (isFixed(myParticle, fixedCount, dFixedNodes))
	{
		pCalc[myParticle] = myParticlePos;
		return;
	}

	float4 delta = make_float4(0.0f);
	{
		float ratio;
		int other;
		float4 deltaPos;
		
		{
			float stickLengthStructural = dStickLengthStructural;
			
			if (y > 0)
			{
				other = myParticle - gridDim;
				ratio = 0.5f;
				
				if (isFixed(other, fixedCount, dFixedNodes))
				{
					ratio = 1.0f;
				}
				else if (y == gridDim - 1)
				{
					ratio = 2.0f / 3.0f;
				}
				else if (other < gridDim)
				{
					ratio = 1.0f / 3.0f;
				}
				
				deltaPos = myParticlePos - pNext[other];
				delta += normalize(deltaPos) * ((stickLengthStructural - length(deltaPos)) * ratio);
			}
			
			if (y < gridDim - 1)
			{
				other = myParticle + gridDim;
				ratio = 0.5f;
				
				if (isFixed(other, fixedCount, dFixedNodes))
				{
					ratio = 1.0f;
				}
				else if (y == 0)
				{
					ratio = 2.0f / 3.0f;
				}
				else if (other >= (gridDim - 1) * gridDim)
				{
					ratio = 1.0f / 3.0f;
				}
				
				deltaPos = myParticlePos - pNext[other];
				delta += normalize(deltaPos) * ((stickLengthStructural - length(deltaPos)) * ratio);
			}
			
			if (x > 0)
			{
				other = myParticle - 1;
				ratio = 0.5f;
				
				if (isFixed(other, fixedCount, dFixedNodes))
				{
					ratio = 1.0f;
				}
				else if (x == gridDim - 1)
				{
					ratio = 2.0f / 3.0f;
				}
				else if (other % gridDim == 0)
				{
					ratio = 1.0f / 3.0f;
				}
				
				deltaPos = myParticlePos - pNext[other];
				delta += normalize(deltaPos) * ((stickLengthStructural - length(deltaPos)) * ratio);
			}
			if (x < gridDim - 1)
			{
				other = myParticle + 1;
				ratio = 0.5f;
				
				if (isFixed(other, fixedCount, dFixedNodes))
				{
					ratio = 1.0f;
				}
				else if (x == 0)
				{
					ratio = 2.0f / 3.0f;
				}
				else if ((other + 1) % gridDim == 0)
				{
					ratio = 1.0f / 3.0f;
				}
				
				deltaPos = myParticlePos - pNext[other];
				delta += normalize(deltaPos) * ((stickLengthStructural - length(deltaPos)) * ratio);
			}
		}

		{
			float stickLengthShear = dStickLengthShear;
			
			if (y > 0 && x > 0)
			{
				other = myParticle - gridDim - 1;
				ratio = 0.5f;
				
				if (isFixed(other, fixedCount, dFixedNodes))
				{
					ratio = 1.0f;
				}
				else if (y == gridDim - 1 && x == gridDim - 1)
				{
					ratio = 4.0f / 5.0f;
				}
				else if (other == 0)
				{
					ratio = 1.0f / 5.0f;
				}
				else if ((other < gridDim && x < gridDim - 1) || (other % gridDim == 0 && y < gridDim - 1))
				{
					ratio = 2.0f / 3.0f;
				}
				else if ((other > gridDim && x == gridDim - 1) || (other % gridDim > 0 && y == gridDim - 1))
				{
					ratio = 1.0f / 3.0f;
				}
				
				deltaPos = myParticlePos - pNext[other];
				delta += normalize(deltaPos) * ((stickLengthShear - length(deltaPos)) * ratio);
			}
			
			if (y > 0 && x < gridDim - 1)
			{
				other = myParticle - gridDim + 1;
				ratio = 0.5f;
				
				if (isFixed(other, fixedCount, dFixedNodes))
				{
					ratio = 1.0f;
				}
				else if (y == gridDim - 1 && x == 0)
				{
					ratio = 4.0f / 5.0f;
				}
				else if (other == gridDim - 1)
				{
					ratio = 1.0f / 5.0f;
				}
				else if ((other < gridDim && x > 0) || ((other + 1) % gridDim == 0 && y < gridDim - 1))
				{
					ratio = 2.0f / 3.0f;
				}
				else if ((other > gridDim && x == 0) || ((other + 1) % gridDim > 0 && y == gridDim - 1))
				{
					ratio = 1.0f / 3.0f;
				}
				
				deltaPos = myParticlePos - pNext[other];
				delta += normalize(deltaPos) * ((stickLengthShear - length(deltaPos)) * ratio);
			}
			
			if (y < gridDim - 1 && x > 0)
			{
				other = myParticle + gridDim - 1;
				ratio = 0.5f;
				
				if (isFixed(other, fixedCount, dFixedNodes))
				{
					ratio = 1.0f;
				}
				else if (y == 0 && x == gridDim - 1)
				{
					ratio = 4.0f / 5.0f;
				}
				else if (other == (gridDim - 1) * gridDim)
				{
					ratio = 1.0f / 5.0f;
				}
				else if ((other % gridDim > 0 && y == 0) || (other < (gridDim - 1) * gridDim && x == gridDim - 1))
				{
					ratio = 2.0f / 3.0f;
				}
				else if ((other % gridDim == 0 && y > 0) || (other >= (gridDim - 1) * gridDim && x < gridDim - 1))
				{
					ratio = 1.0f / 3.0f;
				}
				
				deltaPos = myParticlePos - pNext[other];
				delta += normalize(deltaPos) * ((stickLengthShear - length(deltaPos)) * ratio);
			}
			
			if (y < gridDim - 1 && x < gridDim - 1)
			{
				other = myParticle + gridDim + 1;
				ratio = 0.5f;
				
				if (isFixed(other, fixedCount, dFixedNodes))
				{
					ratio = 1.0f;
				}
				else if (y == 0 && x == 0)
				{
					ratio = 4.0f / 5.0f;
				}
				else if (other == gridDim * gridDim - 1)
				{
					ratio = 1.0f / 5.0f;
				}
				else if (((other + 1) % gridDim > 0 && y == 0) || (other < (gridDim - 1) * gridDim && x == 0))
				{
					ratio = 2.0f / 3.0f;
				}
				else if (((other + 1) % gridDim == 0 && y > 0) || (other >= (gridDim - 1) * gridDim && x > 0))
				{
					ratio = 1.0f / 3.0f;
				}
				
				deltaPos = myParticlePos - pNext[other];
				delta += normalize(deltaPos) * ((stickLengthShear - length(deltaPos)) * ratio);
			}
		}

		{
			float stickLengthFlexion = dStickLengthFlexion;
			
			if (y > 1)
			{
				other = myParticle - (gridDim * 2);
				ratio = 0.5f;
				
				if (isFixed(other, fixedCount, dFixedNodes))
				{
					ratio = 1.0f;
				}
				else if (y == gridDim - 1)
				{
					ratio = 2.0f / 3.0f;
				}
				else if (other < gridDim)
				{
					ratio = 1.0f / 3.0f;
				}
				
				deltaPos = myParticlePos - pNext[other];
				delta += normalize(deltaPos) * ((stickLengthFlexion - length(deltaPos)) * ratio);
			}
			
			if (y < gridDim - 2)
			{
				other = myParticle + (gridDim * 2);
				ratio = 0.5f;
				
				if (isFixed(other, fixedCount, dFixedNodes))
				{
					ratio = 1.0f;
				}
				else if (y == 0)
				{
					ratio = 2.0f / 3.0f;
				}
				else if (other >= (gridDim - 1) * gridDim)
				{
					ratio = 1.0f / 3.0f;
				}
				
				deltaPos = myParticlePos - pNext[other];
				delta += normalize(deltaPos) * ((stickLengthFlexion - length(deltaPos)) * ratio);
			}
			
			if (x > 1)
			{
				other = myParticle - 2;
				ratio = 0.5f;
				
				if (isFixed(other, fixedCount, dFixedNodes))
				{
					ratio = 1.0f;
				}
				else if (x == gridDim - 1)
				{
					ratio = 2.0f / 3.0f;
				}
				else if (other % gridDim == 0)
				{
					ratio = 1.0f / 3.0f;
				}
				
				deltaPos = myParticlePos - pNext[other];
				delta += normalize(deltaPos) * ((stickLengthFlexion - length(deltaPos)) * ratio);
			}
			
			if (x < gridDim - 2)
			{
				other = myParticle + 2;
				ratio = 0.5f;
				
				if (isFixed(other, fixedCount, dFixedNodes))
				{
					ratio = 1.0f;
				}
				else if (x == 0)
				{
					ratio = 2.0f / 3.0f;
				}
				else if ((other + 1) % gridDim == 0)
				{
					ratio = 1.0f / 3.0f;
				}
				
				deltaPos = myParticlePos - pNext[other];
				delta += normalize(deltaPos) * ((stickLengthFlexion - length(deltaPos)) * ratio);
			}
		}
	}

	float4 newpos = myParticlePos + delta * dFixupPercentage;
	float4 obstacleDiff;
	float radius, difference;

	int obstacleCount = dObstacleCount;
	
	for (int i = 0; i < obstacleCount; i++)
	{
		obstacleDiff = dObstacles[i];
		radius = obstacleDiff.w;
		obstacleDiff.w = newpos.w;
		obstacleDiff = newpos - obstacleDiff;
		difference = radius - length(obstacleDiff);
		
		if (difference > 0.0f)
		{
			newpos += normalize(obstacleDiff) * difference;
		}
	}

	pCalc[myParticle] = newpos;
}

void ClothSimulation::simulate(float dt, bool windChanged, float windX, float windY, float windZ, int& currentPositionBuffer)
{
	int lastPosbuffer = (currentPositionBuffer + 2) % 3;
	int nextPosbuffer = (currentPositionBuffer + 1) % 3;
	dim3 block(16, 16);
	dim3 threadGrid((gridDim + block.x - 1) / block.x, (gridDim + block.y - 1) / block.y);

	if (windChanged)
	{
		float4 wind{ windX, windY, windZ, 0.0f };
		cudaMemcpyToSymbol(dWind, &wind, sizeof(float4));
	}

	simulationStep<<<threadGrid, block>>>(dt, mappedPositions[lastPosbuffer], mappedPositions[currentPositionBuffer], mappedPositions[nextPosbuffer]);

	cudaDeviceSynchronize();
	checkCudaError(cudaGetLastError());

	currentPositionBuffer = nextPosbuffer;

	for (int i = 0; i < fixUpIterations; i++)
	{
		int tmp = lastPosbuffer;
		lastPosbuffer = currentPositionBuffer;
		currentPositionBuffer = tmp;

		fixupStep<<<threadGrid, block>>>(mappedPositions[nextPosbuffer], mappedPositions[lastPosbuffer]);
	}
}

void ClothSimulation::newCloth(int gridDim, float sizeX, float sizeZ, float clothMass, int fixUpIterations, float fixUpPercent, std::vector<int> fixedNodes, std::vector<float4> obstacles)
{
	this->gridDim = gridDim;
	this->sizeX = sizeX;
	this->sizeZ = sizeZ;
	this->invParticleMass = gridDim * gridDim / clothMass;
	this->fixUpIterations = fixUpIterations;
	this->fixUpPercent = fixUpPercent;

	lengthStructural = sizeX / (gridDim - 1);
	lengthShear = lengthStructural * sqrt(2.0f);
	lengthFlexion = lengthStructural * 2.0f;

	checkCudaError(cudaMemcpyToSymbol(dGravity, &gravity, sizeof(float), 0, cudaMemcpyHostToDevice));
	checkCudaError(cudaMemcpyToSymbol(dDamping, &damping, sizeof(float), 0, cudaMemcpyHostToDevice));
	checkCudaError(cudaMemcpyToSymbol(dFixupPercentage, &fixUpPercent, sizeof(float), 0, cudaMemcpyHostToDevice));
	checkCudaError(cudaMemcpyToSymbol(dGridDim, &gridDim, sizeof(int), 0, cudaMemcpyHostToDevice));

	checkCudaError(cudaMemcpyToSymbol(dStickLengthStructural, &lengthStructural, sizeof(float), 0, cudaMemcpyHostToDevice));
	checkCudaError(cudaMemcpyToSymbol(dStickLengthShear, &lengthShear, sizeof(float), 0, cudaMemcpyHostToDevice));
	checkCudaError(cudaMemcpyToSymbol(dStickLengthFlexion, &lengthFlexion, sizeof(float), 0, cudaMemcpyHostToDevice));

	checkCudaError(cudaMemcpyToSymbol(dInvParticleMass, &invParticleMass, sizeof(float), 0, cudaMemcpyHostToDevice));

	if (fixedNodesArray != nullptr)
	{
		delete[] fixedNodesArray;
		fixedNodesArray = nullptr;
		checkCudaError(cudaFree(dptrFixedNodes));
	}

	int fixedNodesArraySize = fixedNodes.size();
	checkCudaError(cudaMemcpyToSymbol(dFixedNodesCount, &fixedNodesArraySize, sizeof(int), 0, cudaMemcpyHostToDevice));

	if (fixedNodesArraySize > 0)
	{
		fixedNodesArray = new int[fixedNodesArraySize];

		for (int i = 0; i < fixedNodesArraySize; i++)
		{
			fixedNodesArray[i] = fixedNodes[i];
		}

		checkCudaError(cudaMalloc(&dptrFixedNodes, fixedNodesArraySize * sizeof(int)));
		checkCudaError(cudaMemcpy(dptrFixedNodes, &fixedNodesArray[0], fixedNodesArraySize * sizeof(int), cudaMemcpyHostToDevice));
		checkCudaError(cudaMemcpyToSymbol(dFixedNodes, &dptrFixedNodes, sizeof(int*), 0, cudaMemcpyHostToDevice));
	}

	if (obstaclesArray != nullptr)
	{
		delete[] obstaclesArray;
		obstaclesArray = nullptr;
		checkCudaError(cudaFree(dptrObstacles));
	}

	int obstacleArraySize = obstacles.size();
	checkCudaError(cudaMemcpyToSymbol(dObstacleCount, &obstacleArraySize, sizeof(int), 0, cudaMemcpyHostToDevice));

	if (obstacleArraySize > 0)
	{
		obstaclesArray = new float4[obstacleArraySize];

		for (int i = 0; i < obstacleArraySize; i++)
		{
			obstaclesArray[i] = obstacles[i];
		}

		checkCudaError(cudaMalloc(&dptrObstacles, obstacleArraySize * sizeof(float4)));
		checkCudaError(cudaMemcpy(dptrObstacles, &obstaclesArray[0], obstacleArraySize * sizeof(float4), cudaMemcpyHostToDevice));
		checkCudaError(cudaMemcpyToSymbol(dObstacles, &dptrObstacles, sizeof(float4*), 0, cudaMemcpyHostToDevice));
	}
}

void ClothSimulation::copyPositionsToCPU(float4 * cpuBuffer, int currentPositionBuffer)
{
	checkCudaError(cudaMemcpy(cpuBuffer, mappedPositions[currentPositionBuffer], gridDim*gridDim * 4 * 4, cudaMemcpyDeviceToHost));
}