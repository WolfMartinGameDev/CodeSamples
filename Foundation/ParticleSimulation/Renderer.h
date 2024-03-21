#pragma once

#include "RenderMode.h"
#include "cuda/ParticleSystem.h"

#include "GL/buffer.h"
#include "GL/gl.h"
#include "GL/platform/Context.h"
#include "GL/platform/DefaultDisplayHandler.h"
#include "GL/platform/Window.h"
#include "GL/shader.h"
#include "GL/texture.h"
#include "GL/vertex_array.h"

#include "framework/BasicRenderer.h"
#include "framework/utils/Camera.h"
#include "framework/utils/OrbitalNavigator.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <fstream>

class Renderer : public GL::platform::Renderer, public GL::platform::DisplayHandler
{
public:
	Renderer(const Renderer&) = delete;
	Renderer& operator=(const Renderer&) = delete;

	Renderer(GL::platform::Window& window, Camera& camera, OrbitalNavigator& navigator, int openGLVersionMajor, int openGLVersionMinor, int numParticles);

	virtual void render() override;

	virtual void move(int x, int y) override;
	virtual void resize(int width, int height) override;
	virtual void close() override;
	virtual void destroy() override;

	void resetParticles();
	void cycleRenderMode();
	void freeze();
	void adjustSimulationSpeed(double factor);
	void switchScene(int i);

protected:
	GL::platform::Window& window;
	Camera& camera;
	OrbitalNavigator& navigator;

	GL::platform::Context context;
	GL::platform::context_scope<GL::platform::Window> contextScope;

	int viewportWidth{0};
	int viewportHeight{0};
	RenderMode renderMode{RENDER_MODE_OPENGL};

	GL::Texture raycastingTexture{0};
	int raycastingTextureUnit{0};

	GL::Program fullscreenProgram;
	GLint fullscreenProgramTextureLocation{0};
	GL::VertexArray fullscreenVertexArray;

	GL::Program particleProgram;
	GL::Buffer cameraUniformBuffer;
	GL::Buffer particleVertexBuffer;
	GL::Buffer particleColorBuffer;
	GL::VertexArray particleVertexArray;
	GLint particleRadiusLocation;

	ParticleSystem particleSystem;

	std::chrono::high_resolution_clock::time_point referenceTime;
	float simulationTime;
	float renderTime;

	int maxNumParticles;
	float particleRadius;
	int scene{0};
	int frameCounter{0};
	const char* defaultTitle{"RTG2 SPH"};
	const float simulationStep = 0.01f;
	double lastDisplayUpdate{0.0};
	double lastSimulation{-1.0};
	double simulationSpeed{1.0};
	int maxSimulationsPerFrame{1};
	bool benchmark{false};
	bool frozen{false};

	void renderOpenGL();
	void renderCUDA();
	void updateSimulation(double now);
	void swapBuffers();
};