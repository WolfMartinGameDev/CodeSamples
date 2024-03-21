#include "ParticleSystem.h"
#include "framework/cuda/CheckError.h"
#include <GL/gl.h>
#include <cuda.h>
#include <cuda_gl_interop.h>

ParticleSystem::ParticleSystem(uint maxNumParticles) : maxNumParticles(maxNumParticles)
{
	checkCudaError(cudaEventCreate(&start));
	checkCudaError(cudaEventCreate(&stop));
	setupFixedSizeData();
}

ParticleSystem::~ParticleSystem()
{
	checkCudaError(cudaEventDestroy(start));
	checkCudaError(cudaEventDestroy(stop));
	checkCudaError(cudaFree(dVelocity));

	checkCudaError(cudaFree(dNewVelocity));
	checkCudaError(cudaFree(dDensity));
	checkCudaError(cudaFree(dTempPositions));
}

void ParticleSystem::registerBuffers(unsigned int positionBuffer, unsigned int colorBuffer)
{
	checkCudaError(cudaGraphicsGLRegisterBuffer((struct cudaGraphicsResource**)&positionBufferCudaResource, positionBuffer, cudaGraphicsMapFlagsNone));
	checkCudaError(cudaGraphicsGLRegisterBuffer((struct cudaGraphicsResource**)&colorBufferCudaResource, colorBuffer, cudaGraphicsMapFlagsNone));
}

void ParticleSystem::mapSharedBuffers()
{
	checkCudaError(cudaGraphicsMapResources(1, (struct cudaGraphicsResource**)&positionBufferCudaResource, 0));
	checkCudaError(cudaGraphicsMapResources(1, (struct cudaGraphicsResource**)&colorBufferCudaResource, 0));

	size_t numBytes;
	checkCudaError(cudaGraphicsResourceGetMappedPointer((void**)&dPositions, &numBytes, (struct cudaGraphicsResource*)positionBufferCudaResource));

	checkCudaError(cudaGraphicsResourceGetMappedPointer((void**)&dColor, &numBytes, (struct cudaGraphicsResource*)colorBufferCudaResource));
}

void ParticleSystem::unmapSharedBuffers()
{
	checkCudaError(cudaGraphicsUnmapResources(1, (struct cudaGraphicsResource**)&positionBufferCudaResource, 0));
	checkCudaError(cudaGraphicsUnmapResources(1, (struct cudaGraphicsResource**)&colorBufferCudaResource, 0));
}

float ParticleSystem::update(float dt)
{
	checkCudaError(cudaEventRecord(start, 0));

	runParticleSimulation(dt);

	checkCudaError(cudaEventRecord(stop, 0));

	checkCudaError(cudaEventSynchronize(stop));
	float time;
	checkCudaError(cudaEventElapsedTime(&time, start, stop));
	return time / 1000.0f;
}

float ParticleSystem::render(RenderMode renderMode, const math::float4x4& viewProjectionInverse)
{
	checkCudaError(cudaEventRecord(start, 0));

	cudaRender(renderMode, viewProjectionInverse);

	checkCudaError(cudaEventRecord(stop, 0));

	checkCudaError(cudaEventSynchronize(stop));
	
	float time;
	checkCudaError(cudaEventElapsedTime(&time, start, stop));
	
	return time / 1000.0f;
}

void ParticleSystem::addParticles(uint num, const math::float4* positions, const math::float4* velocities, const math::float4* colors)
{
	checkCudaError(cudaMemcpy(dPositions + numParticles, positions, num * sizeof(math::float4), cudaMemcpyHostToDevice));
	checkCudaError(cudaMemcpy(dVelocity + numParticles, velocities, num * sizeof(math::float4), cudaMemcpyHostToDevice));
	checkCudaError(cudaMemcpy(dColor + numParticles, colors, num * sizeof(math::float4), cudaMemcpyHostToDevice));
	numParticles += num;
}

void ParticleSystem::clearSimulation(const SimulationParameters& newParams)
{
	params = newParams;
	numParticles = 0;

	setupSceneData();
}

void ParticleSystem::resizeRenderTarget(unsigned int image, int width, int height)
{
	this->width = width;
	this->height = height;
	
	if (renderTargetCudaResource)
	{
		checkCudaError(cudaGraphicsUnregisterResource(renderTargetCudaResource));
		renderTargetCudaResource = nullptr;
	}
	
	checkCudaError(cudaGraphicsGLRegisterImage(&renderTargetCudaResource, image, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore));
}