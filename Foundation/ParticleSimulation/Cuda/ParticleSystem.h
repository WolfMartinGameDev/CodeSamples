#pragma once

#include "../RenderMode.h"
#include "framework/math/matrix.h"
#include "framework/math/vector.h"
#include <cuda_runtime.h>
#include <vector>

#ifndef HELPER_MATH_H
typedef unsigned int uint;
#endif

class ParticleSystem
{
public:
	struct SimulationParameters
	{
		// bounding box limits
		math::float3 bbMin;
		math::float3 bbMax;

		float particleRadius;

		// target density of the fluid
		float rho0;
		// stiffness of the fluid
		float springConstant;
		float viscosity;
		// rate at which approaching particles are slowed down
		float adjustedViscosity;
		// how much velocity is converted to heat(lost) each frame
		float dissipation;
		// how bouncy collisions are
		float contactElasticity;
		float maxVelocity;
		float gravity;

		uchar4 backgroundColor;

		float particleRadius2;
		uint numParticles;

		const float4* position;
	};

	ParticleSystem(uint maxNumParticles);
	~ParticleSystem();

	void registerBuffers(uint positionBuffer, uint colorBuffer);

	void clearSimulation(const SimulationParameters& newParams);

	void addParticles(uint num, const math::float4* positions, const math::float4* velocities, const math::float4* colors);
	float update(float dt);
	float render(RenderMode renderMode, const math::float4x4& viewProjectionInverse);

	void resizeRenderTarget(uint image, int width, int height);

	void mapSharedBuffers();
	void unmapSharedBuffers();

	inline int getParticleCount() const;

protected:
	void runParticleSimulation(float dt);
	void cudaRender(RenderMode renderMode, const math::float4x4& viewProjectionInverse);
	void setupSceneData();
	void setupFixedSizeData();
	
	SimulationParameters params;
	
	void* positionBufferCudaResource{nullptr};
	float4* dPositions{nullptr};
	void* colorBufferCudaResource{nullptr};
	float4* dColor{nullptr};

	float4* dVelocity{nullptr};

	int width{0};
	int height{0};
	uint maxNumParticles{0};
	uint numParticles{0};

	cudaGraphicsResource_t renderTargetCudaResource{nullptr};
	cudaArray_t renderTargetCudaArray{nullptr};

	cudaEvent_t start;
	cudaEvent_t stop;
	
	int* dptrSpatialGrid = nullptr;
	int* dptrCellSize = nullptr;
	int numCells = 0;
	int* dptrCellStartIdx = nullptr;
	
	float4* dNewVelocity{0};
	float4* dTempPositions{0};
	float* dDensity{0};
};

int ParticleSystem::getParticleCount() const
{
	return numParticles;
}