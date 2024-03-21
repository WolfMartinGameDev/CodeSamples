#include "ParticleSystem.h"
#include "utils.cuh"

#include "framework/cuda/CheckError.h"
#include "framework/cuda/helper_math.h"

#include <algorithm>
#include <float.h>
#include <iostream>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>

#include <cuda.h>
#include <math_constants.h>

__constant__ ParticleSystem::SimulationParameters cParameters;

__constant__ int* dCellSizes;
__constant__ int* dCellStartIndices;
__constant__ int* dSpatialGrid;
__constant__ int dSpatialGridSize;
__constant__ int3 dGridDimensions;

surface<void, cudaSurfaceType2D> dRenderTarget;

template <typename F>
__device__ void iterateParticles(float4 position, F f)
{
	float h = cParameters.particleRadius * 4.0f;
	int cellX = (position.x - cParameters.bbMin.x) / h;
	int cellY = (position.y - cParameters.bbMin.y) / h;
	int cellZ = (position.z - cParameters.bbMin.z) / h;

	int currentCell = 0;
	int currentCellStartIdx = 0;
	int currentCellSize = 0;

#pragma unroll
	for (int x = -1; x <= 1; ++x)
	{
		for (int y = -1; y <= 1; ++y)
		{
			for (int z = -1; z <= 1; ++z)
			{
				if (cellX + x >= 0 && cellX + x < dGridDimensions.x && cellY + y >= 0 && cellY + y < dGridDimensions.y && cellZ + z >= 0 && cellZ + z < dGridDimensions.z)
				{
					currentCell = (int)((cellX + x) + ((cellY + y) * dGridDimensions.x) + ((cellZ + z) * dGridDimensions.x * dGridDimensions.y));

					currentCellStartIdx = dCellStartIndices[currentCell];
					currentCellSize = dCellSizes[currentCell];

					for (int i = currentCellStartIdx; i < currentCellStartIdx + currentCellSize; i++)
					{
						f(dSpatialGrid[i]);
					}
				}
			}
		}
	}
}

__device__ inline int computeParticleIdx(uint gridDim)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int z = threadIdx.z + blockIdx.z * blockDim.z;

	if (x >= gridDim || y >= gridDim || z >= gridDim)
	{
		return -1;
	}

	return x + y * gridDim + z * gridDim * gridDim;
}

__device__ inline int computeDestinationCell(float4 position)
{
	float h = cParameters.particleRadius * 4.0f;
	int cellX = (position.x - cParameters.bbMin.x) / h;
	int cellY = (position.y - cParameters.bbMin.y) / h;
	int cellZ = (position.z - cParameters.bbMin.z) / h;

	if (cellX >= dGridDimensions.x || cellY >= dGridDimensions.y || cellZ >= dGridDimensions.z)
	{
		return;
	}

	return (int)((cellX)+(cellY * dGridDimensions.x) + (cellZ * dGridDimensions.x * dGridDimensions.y));
}

__global__ void computeGridCells(int gridDim, float4* positions)
{
	int myParticle = computeParticleIdx(gridDim);
	
	if (myParticle < 0)
	{
		return;
	}

	float4 position = positions[myParticle];
	int destinationCell = computeDestinationCell(position);
	
	if (destinationCell < 0)
	{
		return;
	}

	atomicAdd(&dCellSizes[destinationCell], 1);
}

__global__ void insertIntoSpatialGrid(int gridDim, float4* positions)
{
	int myParticle = computeParticleIdx(gridDim);
	if (myParticle < 0)
	{
		return;
	}

	float4 position = positions[myParticle];
	int destinationCell = computeDestinationCell(position);
	
	if (destinationCell < 0)
	{
		return;
	}

	int cellAddress = dCellStartIndices[destinationCell];
	int offset = atomicAdd(&dCellSizes[destinationCell], 1);

	if (cellAddress + offset >= dSpatialGridSize)
	{
		return;
	}

	dSpatialGrid[cellAddress + offset] = myParticle;
}

__device__ inline float4 matMul(float4 vec, math::float4x4 mat)
{
	float4 res;

	res.x = vec.x * mat.column1().x + vec.y * mat.column2().x + vec.z * mat.column3().x + vec.w * mat.column4().x;
	res.y = vec.x * mat.column1().y + vec.y * mat.column2().y + vec.z * mat.column3().y + vec.w * mat.column4().y;
	res.z = vec.x * mat.column1().z + vec.y * mat.column2().z + vec.z * mat.column3().z + vec.w * mat.column4().z;
	res.w = vec.x * mat.column1().w + vec.y * mat.column2().w + vec.z * mat.column3().w + vec.w * mat.column4().w;

	return res;
}

__device__ inline float2 getTNearFar(math::float3 minPoint, math::float3 maxPoint, float4 checkPoint, float4 direction)
{
	float tx1 = (minPoint.x - checkPoint.x) / direction.x;
	float tx2 = (maxPoint.x - checkPoint.x) / direction.x;
	float ty1 = (minPoint.y - checkPoint.y) / direction.y;
	float ty2 = (maxPoint.y - checkPoint.y) / direction.y;
	float tz1 = (minPoint.z - checkPoint.z) / direction.z;
	float tz2 = (maxPoint.z - checkPoint.z) / direction.z;

	return make_float2(math::max(
			math::max(
				math::min(tz1, tz2),
				math::min(ty1, ty2)), 
			math::min(tx1, tx2)),
		math::min(
			math::min(
				math::max(tz1, tz2),
				math::max(ty1, ty2)), 
			math::max(tx1, tx2)));
}

__global__ void renderRay(int width, int height, math::float4x4 inverseMvpMat, const float4* __restrict positions, const float4* __restrict colors)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x >= width || y >= height)
	{
			return;
	}

	float4 rayNear = make_float4(2.0f * (x + 0.5f) / width - 1.0f, 2.0f * (y + 0.5f) / height - 1.0f, -1.0f, 1.0f);
	float4 rayFar = rayNear;
	rayFar.z = 1.0f;

	rayNear = matMul(rayNear, inverseMvpMat);
	rayFar = matMul(rayFar, inverseMvpMat);

	rayNear /= rayNear.w;
	rayFar /= rayFar.w;

	uchar4 color = cParameters.backgroundColor;
	float4 direction = rayFar - rayNear;
	float rayLength = length(direction);
	float minDistance = rayLength + 0.00001f;
	direction = normalize(direction);

	float2 tNearFar = getTNearFar(cParameters.bbMin, cParameters.bbMax, rayNear, direction);
	math::float3 bbDim = cParameters.bbMax - cParameters.bbMin;
	bbDim = math::float3(bbDim.x / dGridDimensions.x, bbDim.y / dGridDimensions.y, bbDim.z / dGridDimensions.z);
	bool found = false;

	if (tNearFar.x <= tNearFar.y)
	{
		float h = cParameters.particleRadius * 4.0f;
		float tCurr = tNearFar.x;
		
		while (tCurr < tNearFar.y)
		{
			found = false;
			float4 currPos = rayNear + (tCurr + 0.00001f) * direction;
			
			iterateParticles(currPos, [&](uint i)
			{
				float4 particlePos = positions[i];
				float tParticleNear, tParticleFar;
				float4 distance = currPos - particlePos;

				float a = dot(direction, direction);
				float b = 2.0f * dot(direction, distance);
				float c = dot(distance, distance) - cParameters.particleRadius2;

				float discr = b * b - 4.0f * a * c;
				
				if (discr >= 0.0f)
				{
					if (discr == 0.0f)
					{
						tParticleNear = tParticleFar = -0.5f * b / a;
					}
					else
					{
						float q = (b > 0.0f) ? -0.5f * (b + sqrt(discr)) : -0.5f * (b - sqrt(discr));
						tParticleNear = q / a;
						tParticleFar = c / q;
					}

					if (tParticleNear > tParticleFar)
					{
						float tmp = tParticleNear;
						tParticleNear = tParticleFar;
						tParticleFar = tmp;
					}

					if (tParticleNear < minDistance)
					{
						minDistance = tParticleNear;
						found = true;
						float4 intersectionPoint = currPos + direction * tParticleNear;
						float4 normal = intersectionPoint - particlePos;
						normal = normalize(normal);
						float reflectedLight = max(dot(-normal, direction), 0.0f);
						color = make_uchar4(0, 0, reflectedLight * 255, 255);
					}
				}
			});

			if (found)
			{
				surf2Dwrite(color, dRenderTarget, x * 4, y);
				return;
			}
			else
			{
				tCurr += length(bbDim);
			}
		}
	}
	
	surf2Dwrite(color, dRenderTarget, x * 4, y);
}

struct SmoothingWendlandC2_3D
{
	static constexpr float C = 21.0f / (2.0f * CUDART_PI_F);

	__device__ static float smoothing(float q)
	{
		if (q >= 1.0f)
		{
			return 0.0f;
		}

		float v = 1.0f - q;
		v = v * v * v * v;
		v *= (1.0f + 4.0f * q);
		return v * C;
	}
	
	__device__ static float dsmoothing(float q)
	{
		if (q >= 1.0f)
		{
			return 0.0f;
		}
		
		float v = (q - 1.0f);
		v = 20.0f * v * v * v * q;
		return v * C;
	}

	__device__ static math::float3 grad(const math::float3& r_ij)
	{
		float rinv = 1.0f / cParameters.particleRadius;

		math::float3 rabs = abs(r_ij) * rinv;

		math::float3 vs(dsmoothing(rabs.x), dsmoothing(rabs.y), dsmoothing(rabs.z));

		return math::float3(r_ij.x > 0 ? vs.x : -vs.x, r_ij.y > 0 ? vs.y : -vs.y, r_ij.z > 0 ? vs.z : -vs.z);
	}
};

template <typename T>
__device__ inline math::float3 toM3(const T& v)
{
	return math::float3(v.x, v.y, v.z);
}

template <typename T>
__device__ inline float3 toF3(const T& v)
{
	return make_float3(v.x, v.y, v.z);
}

template <typename T>
__device__ inline T estimatorSymmetric(T Aj, T Ai, float densityj, float densityi)
{
	return (Ai / (densityi * densityi) + Aj / (densityj * densityj));
}

__device__ float pressure(float density)
{
	return cParameters.springConstant * (density - cParameters.rho0);
}

__global__ void gMoveParticleToTemp(int gridDim, const float4* __restrict position, float4* tempPosition, int numParticles)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;
	int tid = x + y * gridDim + z * gridDim * gridDim;
	
	if (tid >= numParticles)
	{
		return;
	}
	
	float4 pos = position[tid];
	tempPosition[tid] = pos;
}

__global__ void gComputeState(int gridDim, const float4* __restrict positions, const float4* __restrict velocity, float* __restrict dDensity, int numParticles)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;

	int tid = x + y * gridDim + z * gridDim * gridDim;

	if (tid >= numParticles)
	{
		return;
	}

	uint particleId = tid;
	math::float3 pos = toM3(positions[particleId]);
	math::float3 vel = toM3(velocity[particleId]);

	float density = SmoothingWendlandC2_3D::smoothing(0.0f);

	float4 position = positions[particleId];
	
	iterateParticles(position, [&](uint otherId)
	{
		float dist2 = length2(pos - toM3(positions[otherId]));
		
		if (dist2 < cParameters.particleRadius2)
		{
			float dist = sqrt(dist2);
			density += SmoothingWendlandC2_3D::smoothing(dist / cParameters.particleRadius);
		}
	});
	
	dDensity[particleId] = density;
}

__global__ void gUpdate(int gridDim, const float4* __restrict positions, float4* newPosition, const float4* __restrict velocity, float4* newVelocity, const float* __restrict dDensity, int numParticles, float dt)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;

	int tid = x + y * gridDim + z * gridDim * gridDim;

	if (tid >= numParticles)
	{
		return;
	}

	uint particleId = tid;
	math::float3 pos = toM3(positions[particleId]);
	math::float3 vel = toM3(velocity[particleId]);
	float density = dDensity[particleId];

	math::float3 dPressure(0);
	math::float3 dVelocityViscosity(0);
	math::float3 dVelocityAdjustedViscosity(0);

	float surroundingDensity = 0;
	int surrounding = 0;

	float4 position = positions[particleId];
	
	iterateParticles(position, [&](uint otherId)
	{
		math::float3 otherPos = toM3(positions[otherId]);
		math::float3 r_ij = otherPos - pos;
		float d2 = length2(r_ij);

		if (d2 < cParameters.particleRadius2)
		{
			math::float3 nablaW_ij = SmoothingWendlandC2_3D::grad(r_ij);

			float otherDensity = dDensity[otherId];

			dPressure += nablaW_ij * estimatorSymmetric(pressure(density), pressure(otherDensity), density, otherDensity);

			if (otherId != particleId && d2 > 0.0001f)
			{
				surroundingDensity += otherDensity;
				++surrounding;
			}

			math::float3 otherVel = toM3(velocity[otherId]);
			math::float3 v_ij = otherVel - vel;

			dVelocityViscosity += cParameters.viscosity * v_ij * SmoothingWendlandC2_3D::smoothing(sqrt(d2) / cParameters.particleRadius);

			dVelocityAdjustedViscosity -= nablaW_ij * cParameters.adjustedViscosity * (dot(v_ij, r_ij) / (d2 + 0.0001f * cParameters.particleRadius2));
		}
	});

	if (surrounding < 2)
	{
		surroundingDensity = 0;
	}
	else
	{
		surroundingDensity /= surrounding;
	}

	math::float3 dVelocityPress = -dPressure;
	math::float3 dVelocityGravity = math::float3(0, -cParameters.gravity * (density - surroundingDensity) / density, 0);

	math::float3 dVel = dVelocityPress + dVelocityGravity + dVelocityViscosity + dVelocityAdjustedViscosity;

	dVel -= cParameters.dissipation * vel;

	math::float3 velNew = vel + dt * dVel;
	pos = pos + 0.5f * dt * (vel + velNew);
	vel = velNew;

	math::float3 bounceVel(0);
	math::float3 bbmin = cParameters.bbMin + cParameters.particleRadius;
	math::float3 bbmax = cParameters.bbMax - cParameters.particleRadius;
	
	if (pos.x < bbmin.x)
	{
		bounceVel.x = vel.x;
		pos.x = bbmin.x;
	}
	else if (pos.x > bbmax.x)
	{
		bounceVel.x = vel.x;
		pos.x = bbmax.x;
	}
	
	if (pos.y < bbmin.y)
	{
		bounceVel.y = vel.y;
		pos.y = bbmin.y;
	}
	else if (pos.y > bbmax.y)
	{
		bounceVel.y = vel.y;
		pos.y = bbmax.y;
	}
	
	if (pos.z < bbmin.z)
	{
		bounceVel.z = vel.z;
		pos.z = bbmin.z;
	}
	else if (pos.z > bbmax.z)
	{
		bounceVel.z = vel.z;
		pos.z = bbmax.z;
	}

	vel -= bounceVel * (1.0f + cParameters.contactElasticity);
	float l = length(vel);
	
	if (l > cParameters.maxVelocity)
	{
		vel = cParameters.maxVelocity * vel / l;
		l = cParameters.maxVelocity;
	}

	newPosition[particleId] = make_float4(pos.x, pos.y, pos.z, 1);
	newVelocity[particleId] = make_float4(vel.x, vel.y, vel.z, 0);
}

template <typename T>
T ceilDiv(T a, T b)
{
	return (a + b - 1) / b;
}

void ParticleSystem::runParticleSimulation(float dt)
{
	params.position = dPositions;

	checkCudaError(cudaMemcpyToSymbol(cParameters, &params, sizeof(SimulationParameters)));
	
	dim3 threadNum(8, 8, 8);
	int gridDim = std::cbrt(numParticles);
	dim3 threadGrid((gridDim + threadNum.x - 1) / threadNum.x, (gridDim + threadNum.y - 1) / threadNum.y, (gridDim + threadNum.z - 1) / threadNum.z);

	checkCudaError(cudaMemset((void*)dptrCellSize, 0, numCells * sizeof(int)));
	computeGridCells<<<threadGrid, threadNum>>>(gridDim, dPositions);
	thrust::exclusive_scan(thrust::device, dptrCellSize, dptrCellSize + numCells, dptrCellStartIdx);
	checkCudaError(cudaMemset((void*)dptrCellSize, 0, numCells * sizeof(int)));
	insertIntoSpatialGrid<<<threadGrid, threadNum>>>(gridDim, dPositions);

	gMoveParticleToTemp<<<threadGrid, threadNum>>>(gridDim, dPositions, dTempPositions, numParticles);
	gComputeState<<<threadGrid, threadNum>>>(gridDim, dTempPositions, dVelocity, dDensity, numParticles);
	gUpdate<<<threadGrid, threadNum>>>(gridDim, dTempPositions, dPositions, dVelocity, dNewVelocity, dDensity, numParticles, dt);

	float4* tmp = dVelocity;
	dVelocity = dNewVelocity;
	dNewVelocity = tmp;
}

void ParticleSystem::setupFixedSizeData()
{
	checkCudaError(cudaMalloc((void**)&dVelocity, sizeof(float4) * maxNumParticles));
	checkCudaError(cudaMemset((void*)dVelocity, 0, sizeof(float4) * maxNumParticles));

	if (dNewVelocity)
	{
		cudaFree(dNewVelocity);
		dNewVelocity = 0;
	}

	checkCudaError(cudaMalloc((void**)&dNewVelocity, sizeof(float4) * maxNumParticles));
	checkCudaError(cudaMemset((void*)dNewVelocity, 0, sizeof(float4) * maxNumParticles));

	if (dTempPositions)
	{
		cudaFree(dTempPositions);
		cudaFree(dDensity);
		dTempPositions = 0;
	}

	checkCudaError(cudaMalloc((void**)&dDensity, sizeof(float) * maxNumParticles));
	checkCudaError(cudaMalloc((void**)&dTempPositions, sizeof(float4) * maxNumParticles));
	checkCudaError(cudaMemset((void*)dTempPositions, 0, sizeof(float4) * maxNumParticles));
}

void ParticleSystem::setupSceneData()
{
	math::float3 bbdim = params.bbMax - params.bbMin;
	printf("BB dim: %6.2f %6.2f %6.2f\n", bbdim.x, bbdim.y, bbdim.z);
	printf("BB min: %6.2f %6.2f %6.2f\n", params.bbMin.x, params.bbMin.y, params.bbMin.z);
	printf("BB max: %6.2f %6.2f %6.2f\n", params.bbMax.x, params.bbMax.y, params.bbMax.z);

	params.position = dPositions;
	params.numParticles = numParticles;
	params.particleRadius2 = params.particleRadius * params.particleRadius;

	float h = params.particleRadius * 4.0f;
	int3 gridDims = {static_cast<int>(ceilf((bbdim.x) / h)), static_cast<int>(ceilf((bbdim.y) / h)), static_cast<int>(ceilf((bbdim.z) / h))};

	numCells = gridDims.x * gridDims.y * gridDims.z;

	if (dptrSpatialGrid)
	{
		checkCudaError(cudaFree(dptrSpatialGrid));
	}
	
	if (dptrCellSize)
	{
		checkCudaError(cudaFree(dptrCellSize));
	}
	
	if (dptrCellStartIdx)
	{
		checkCudaError(cudaFree(dptrCellStartIdx));
	}

	checkCudaError(cudaMalloc((void**)&this->dptrSpatialGrid, maxNumParticles * sizeof(int)));
	checkCudaError(cudaMalloc((void**)&this->dptrCellSize, numCells * sizeof(int)));
	checkCudaError(cudaMalloc((void**)&this->dptrCellStartIdx, numCells * sizeof(int)));

	checkCudaError(cudaMemset(this->dptrSpatialGrid, 0, maxNumParticles * sizeof(int)));
	checkCudaError(cudaMemset(this->dptrCellSize, 0, numCells * sizeof(int)));
	checkCudaError(cudaMemset(this->dptrCellStartIdx, 0, numCells * sizeof(int)));

	checkCudaError(cudaMemcpyToSymbol(cParameters, &params, sizeof(SimulationParameters)));
	checkCudaError(cudaMemcpyToSymbol(dSpatialGrid, &this->dptrSpatialGrid, sizeof(int*)));
	checkCudaError(cudaMemcpyToSymbol(dSpatialGridSize, &maxNumParticles, sizeof(maxNumParticles)));
	checkCudaError(cudaMemcpyToSymbol(dGridDimensions, &gridDims, sizeof(gridDims)));
	checkCudaError(cudaMemcpyToSymbol(dCellSizes, &this->dptrCellSize, sizeof(int*)));
	checkCudaError(cudaMemcpyToSymbol(dCellStartIndices, &this->dptrCellStartIdx, sizeof(int*)));
}

void ParticleSystem::cudaRender(RenderMode renderMode, const math::float4x4& viewProjectionInverse)
{
	checkCudaError(cudaGraphicsMapResources(1, (cudaGraphicsResource_t*)&renderTargetCudaResource, 0));
	checkCudaError(cudaGraphicsSubResourceGetMappedArray(&renderTargetCudaArray, renderTargetCudaResource, 0, 0));
	checkCudaError(cudaBindSurfaceToArray(dRenderTarget, renderTargetCudaArray));

	dim3 blockSize(16, 16);

	dim3 threadGrid((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

	renderRay<<<threadGrid, blockSize>>>(width, height, viewProjectionInverse, dPositions, dColor);

	checkCudaError(cudaGraphicsUnmapResources(1, &renderTargetCudaResource, 0));
}