#include "Renderer.h"

#include "GL/error.h"
#include "GL/platform/Application.h"
#include "GL/platform/Window.h"

#include "framework/math/vector.h"

#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>

// generated from the shaders by the build system
extern const char particle_sprite_vs[];
extern const char particle_sprite_gs[];
extern const char particle_sprite_fs[];

extern const char vs_fullscreen[];
extern const char fs_color[];

Renderer::Renderer(GL::platform::Window& window, Camera& camera, OrbitalNavigator& navigator, int openGLVersionMajor, int openGLVersionMinor, int numParticles) :
	GL::platform::Renderer(),
	window(window),
	camera(camera),
	navigator(navigator),
	context(window.createContext(openGLVersionMajor, openGLVersionMinor, true)),
	contextScope(context, window),
	particleSystem(numParticles),
	maxNumParticles(numParticles)
{
	glBindVertexArray(particleVertexArray);
	glActiveTexture(GL_TEXTURE0);

	glEnableVertexAttribArray(0);
	glDisableVertexAttribArray(1);
	glDisableVertexAttribArray(2);
	glDisableVertexAttribArray(3);

	glClearColor(0.1f, 0.3f, 1.0f, 1.0f);
	glClearDepth(1.0f);
	glEnable(GL_DEPTH_TEST);

	auto fullscreenVs = GL::compileVertexShader(vs_fullscreen);
	auto fullscreenFs = GL::compileFragmentShader(fs_color);
	glAttachShader(fullscreenProgram, fullscreenVs);
	glAttachShader(fullscreenProgram, fullscreenFs);
	GL::linkProgram(fullscreenProgram);
	glUseProgram(fullscreenProgram);

	fullscreenProgramTextureLocation = glGetUniformLocation(fullscreenProgram, "tex");
	glUniform1i(fullscreenProgramTextureLocation, raycastingTextureUnit);

	auto particleVs = GL::compileVertexShader(particle_sprite_vs);
	auto particleFs = GL::compileFragmentShader(particle_sprite_fs);
	auto particleGs = GL::compileGeometryShader(particle_sprite_gs);
	glAttachShader(particleProgram, particleVs);
	glAttachShader(particleProgram, particleFs);
	glAttachShader(particleProgram, particleGs);
	GL::linkProgram(particleProgram);

	GLuint cameraParameters = glGetUniformBlockIndex(particleProgram, "CameraParameters");
	glUniformBlockBinding(particleProgram, cameraParameters, 0);
	particleRadiusLocation = glGetUniformLocation(particleProgram, "radius");

	glBindBuffer(GL_UNIFORM_BUFFER, cameraUniformBuffer);
	glBufferData(GL_UNIFORM_BUFFER, sizeof(Camera::UniformBuffer), nullptr, GL_STATIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, particleVertexBuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(math::float4) * numParticles, nullptr, GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindBuffer(GL_ARRAY_BUFFER, particleColorBuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(math::float4) * numParticles, nullptr, GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(particleVertexArray);
	glBindBuffer(GL_ARRAY_BUFFER, particleVertexBuffer);
	glBindVertexBuffer(0U, particleVertexBuffer, 0U, sizeof(float) * 4);
	glEnableVertexAttribArray(0U);

	glBindBuffer(GL_ARRAY_BUFFER, particleColorBuffer);
	glBindVertexBuffer(1U, particleColorBuffer, 0U, sizeof(float) * 4);
	glEnableVertexAttribArray(1U);

	glVertexAttribFormat(0U, 4, GL_FLOAT, GL_FALSE, 0U);
	glVertexAttribBinding(0U, 0U);

	glVertexAttribFormat(1U, 4, GL_FLOAT, GL_FALSE, 0U);
	glVertexAttribBinding(1U, 1U);

	GL::checkError();

	glEnable(GL_CULL_FACE);

	particleSystem.registerBuffers(particleVertexBuffer, particleColorBuffer);
	particleSystem.mapSharedBuffers();

	switchScene(0);

	window.attach(this);
}

void Renderer::resize(int width, int height)
{
	if (width == viewportWidth && height == viewportHeight)
	{
		return;
	}
	
	viewportWidth = width;
	viewportHeight = height;
	std::cout << "Resizing to " << width << "x" << height << std::endl;

	glActiveTexture(GL_TEXTURE0 + raycastingTextureUnit);

	raycastingTexture = GL::createTexture2D(width, height, 1, GL_RGBA8);

	float clearColor[] = {0.2f, 0.2f, 0.2f};
	glClearTexImage(raycastingTexture, 0, GL_RGBA, GL_FLOAT, &clearColor);

	glActiveTexture(GL_TEXTURE0);

	particleSystem.resizeRenderTarget(raycastingTexture, width, height);
}

void Renderer::resetParticles()
{
	switchScene(scene);
}

namespace
{
	math::float4 colorRamp(float t)
	{
		const int ncolors = 7;
		math::float3 c[ncolors] = {
			math::float3(1.0, 0.0, 0.0), math::float3(1.0, 0.5, 0.0),
			math::float3(1.0, 1.0, 0.0), math::float3(0.0, 1.0, 0.0),
			math::float3(0.0, 1.0, 1.0), math::float3(0.0, 0.0, 1.0),
			math::float3(1.0, 0.0, 1.0)};
		t = t * (ncolors - 1);
		int i = (int)t;
		float u = t - floorf(t);
		
		return math::float4(c[i].x + u * (c[i + 1].x - c[i].x),
							c[i].y + u * (c[i + 1].y - c[i].y),
							c[i].z + u * (c[i + 1].z - c[i].z), 1);
	}

	math::float4 particleColor(int i, int numParticles)
	{
		return colorRamp(static_cast<float>(i) / static_cast<float>(numParticles - 1));
	}

	void scene0(ParticleSystem& particleSystem, int targetNumParticles, ParticleSystem::SimulationParameters& params)
	{
		std::cout << "Using scene 0" << std::endl;
		int particlesPerDim = static_cast<int>(std::cbrt(static_cast<double>(targetNumParticles)));

		int numParticles = particlesPerDim * particlesPerDim * particlesPerDim;

		const float bbSize = 2.0f;
		const float bbHalfSize = bbSize * 0.5f;
		params.bbMin = math::float3(-bbHalfSize, 0, -bbHalfSize);
		params.bbMax = math::float3(+bbHalfSize, bbSize, +bbHalfSize);

		particleSystem.clearSimulation(params);
		std::vector<math::float4> position;
		std::vector<math::float4> velocity;
		std::vector<math::float4> color;
		position.resize(numParticles);
		color.resize(numParticles);
		velocity.resize(numParticles);

		math::float3 start(params.bbMin);
		start = start + params.particleRadius;
		float placementDist = 0.9f * params.particleRadius;

		int i = 0;
		
		for (int pz = 0; pz < particlesPerDim; ++pz)
		{
			for (int py = 0; py < particlesPerDim; ++py)
			{
				for (int px = 0; px < particlesPerDim; ++px)
				{
					math::float4 pos(start.x + (px + 0.5f) * placementDist,
									start.y + (py + 0.5f) * placementDist,
									start.z + (pz + 0.5f) * placementDist, 1);
									
					position[i] = pos;
					velocity[i] = math::float4(0);
					color[i] = particleColor(i, numParticles);
					++i;
				}
			}
		}
		
		std::cout << "Generated " << i << " particles" << std::endl;
		particleSystem.addParticles(numParticles, &position[0], &velocity[0], &color[0]);
	}

	void scene1(ParticleSystem& particleSystem, int targetNumParticles, ParticleSystem::SimulationParameters& params)
	{
		std::cout << "Using scene 1" << std::endl;

		const int numParticles = targetNumParticles;

		const float bbSize = 2.0f;
		const float bbHalfSize = bbSize * 0.5f;
		params.bbMin = math::float3(-bbHalfSize, 0, -bbHalfSize);
		params.bbMax = math::float3(+bbHalfSize, bbSize, +bbHalfSize);

		const float cylinderRadius = bbSize * 0.11f;

		particleSystem.clearSimulation(params);
		std::vector<math::float4> position;
		std::vector<math::float4> velocity;
		std::vector<math::float4> color;
		position.resize(numParticles);
		color.resize(numParticles);
		velocity.resize(numParticles);

		math::float3 center = 0.5f * (params.bbMax + params.bbMin);
		math::float4 start(center, 1.0f);
		start.y = params.bbMin.y + params.particleRadius;

		const float placementDist = params.particleRadius * 0.99f;
		const float numRings = floorf(cylinderRadius / placementDist);

		int i = 0;

		int numParticlesPerLevel = 1;
		
		for (int ring = 1; ring < numRings; ++ring)
		{
			numParticlesPerLevel += floorf((placementDist * ring * 2 * M_PI) / placementDist);
		}
		
		const int numLevels = targetNumParticles / numParticlesPerLevel;

		auto generateFunction = [&]() {
			for (int level = 0; level < numLevels; ++level)
			{
				// center particle is fixed
				position[i] = start + math::float4(0, level * placementDist, 0, 0);
				velocity[i] = math::float4(0);
				color[i] = particleColor(i, numParticles);
				++i;

				for (int ring = 1; ring < numRings; ++ring)
				{
					const float ringR = placementDist * ring;
					const float ringC = 2 * M_PI * ringR;
					const int numRingParticles = floorf(ringC / placementDist);
					const float ringPlacementAngle = (2.0f * M_PI) / numRingParticles;

					for (int ringParticle = 0; ringParticle < numRingParticles;
							 ++ringParticle)
					{
						float theta = ringParticle * ringPlacementAngle;
						auto p =
								start + math::float4(ringR * cosf(theta), level * placementDist,
																		 ringR * sinf(theta), 0);

						if (p.x < params.bbMin.x || p.x > params.bbMax.x ||
								p.y < params.bbMin.y || p.y > params.bbMax.y ||
								p.z < params.bbMin.z || p.z > params.bbMax.z)
						{
							std::cout << "Pos out of bounds: " << p << " ring " << ring
												<< " ringParticle " << ringParticle << std::endl;
							std::cout << "Stopping generation" << std::endl;
							return;
						}

						position[i] = p;
						velocity[i] = math::float4(0);
						color[i] = particleColor(i, numParticles);
						++i;
					}
				}
			}
		};

		generateFunction();

		std::cout << "Generated " << i << " particles" << std::endl;
		particleSystem.addParticles(i, &position[0], &velocity[0], &color[0]);
	}
}

void Renderer::switchScene(int scene)
{
	particleRadius = 2.0f / (static_cast<int>(std::cbrt((double)maxNumParticles)) * 4.0f);

	ParticleSystem::SimulationParameters params;
	memset(&params, 0, sizeof(ParticleSystem::SimulationParameters));

	params.backgroundColor = make_uchar4(50, 50, 50, 255);
	params.maxVelocity = 2.5f;
	params.particleRadius = particleRadius;
	params.gravity = 9.81f;

	params.viscosity = 0.01f;
	params.adjustedViscosity = 0.02f;
	params.dissipation = 0.05f;
	params.contactElasticity = 0.1f;
	params.rho0 = 35.0f;
	params.springConstant = 0.7f;

	switch (scene)
	{
	case 1:
		scene1(particleSystem, maxNumParticles, params);
		break;
	default:
	case 0:
		scene0(particleSystem, maxNumParticles, params);
		break;
	}
	lastDisplayUpdate = -1.0f;
}

void Renderer::freeze()
{
	frozen = !frozen;
	
	if (frozen)
	{
		std::cout << "Freezing simulation" << std::endl;
	}
	else
	{
		std::cout << "Resuming simulation" << std::endl;
	}
	
	lastDisplayUpdate = -1.0f;
}

void Renderer::swapBuffers()
{
	contextScope.swapBuffers();
}

void Renderer::renderOpenGL()
{
	particleSystem.unmapSharedBuffers();

	Camera::UniformBuffer camParams;
	camera.writeUniformBuffer(&camParams, navigator, static_cast<float>(viewportWidth) / viewportHeight);

	glBindBuffer(GL_UNIFORM_BUFFER, cameraUniformBuffer);
	glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(Camera::UniformBuffer), &camParams);
	glBindBufferRange(GL_UNIFORM_BUFFER, 0, cameraUniformBuffer, 0, sizeof(Camera::UniformBuffer));

	glViewport(0, 0, viewportWidth, viewportHeight);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glUseProgram(particleProgram);
	glUniform1f(particleRadiusLocation, particleRadius);

	glBindVertexArray(particleVertexArray);

	glDrawArrays(GL_POINTS, 0, particleSystem.getParticleCount());

	particleSystem.mapSharedBuffers();
}

void Renderer::renderCUDA()
{
	Camera::UniformBuffer camParams;
	camera.writeUniformBuffer(&camParams, navigator, static_cast<float>(viewportWidth) / viewportHeight);
	renderTime = particleSystem.render(renderMode, camParams.PV_inv);

	particleSystem.unmapSharedBuffers();
	glViewport(0, 0, viewportWidth, viewportHeight);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glUseProgram(fullscreenProgram);
	glBindVertexArray(fullscreenVertexArray);
	glUniform1i(fullscreenProgramTextureLocation, raycastingTextureUnit);
	glDrawArrays(GL_TRIANGLES, 0, 3);
	particleSystem.mapSharedBuffers();
}

void Renderer::updateSimulation(double now)
{
	int numSimulation;
	
	if (lastSimulation < 0)
	{
		lastSimulation = now;
		numSimulation = 1;
	}
	else
	{
		numSimulation = static_cast<int>((now - lastSimulation) / (double)simulationStep * simulationSpeed);
		lastSimulation += numSimulation * simulationStep;
	}

	if (numSimulation > maxSimulationsPerFrame)
	{
		numSimulation = maxSimulationsPerFrame;
	}

	if (benchmark)
	{
		numSimulation = 1;
	}

	for (int i = 0; i < numSimulation; ++i)
	{
		simulationTime = particleSystem.update(simulationStep);
	}
}

void Renderer::adjustSimulationSpeed(double factor)
{
	simulationSpeed *= factor;
}

void Renderer::render()
{
	double now = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - referenceTime).count();
	
	if (!frozen)
	{
		updateSimulation(now);
	}

	if (renderMode == RENDER_MODE_OPENGL)
	{
		renderOpenGL();
	}
	else
	{
		renderCUDA();
	}

	frameCounter++;

	if (now - lastDisplayUpdate > 1.0)
	{
		lastDisplayUpdate = now;
		std::stringstream info;
		
		if (renderMode != RENDER_MODE_OPENGL)
		{
			info << "RM: " << renderMode;
			info << " RT: " << std::setprecision(5) << (renderTime * 1000) << "ms	";
		}
		
		if (!frozen)
		{
			info << "sim: " << std::setprecision(5) << (simulationTime * 1000) << "ms	";
			info << "sim-speed: " << std::setprecision(3) << simulationSpeed << "x	";
		}
		else
		{
			info << "frozen	";
		}
		
		info << defaultTitle;
		window.title(info.str().c_str());
	}

	GL::checkError();
	swapBuffers();
}

void Renderer::close()
{
	GL::platform::quit();
}

void Renderer::destroy()
{
	GL::platform::quit();
}

void Renderer::move(int, int)
{
}

void Renderer::cycleRenderMode()
{
	renderMode = static_cast<RenderMode>((renderMode + 1) % RENDER_MODE_END_MARKER);
	lastDisplayUpdate = -1.0f;
}