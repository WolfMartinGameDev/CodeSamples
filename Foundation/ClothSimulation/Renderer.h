#ifndef RENDERER_H_OIBQH5XL
#define RENDERER_H_OIBQH5XL

#pragma once

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
#include <vector>

#include "ClothSimulation.h"

class Renderer : public GL::platform::Renderer, public GL::platform::DisplayHandler
{
public:
	Renderer(const Renderer&) = delete;
	
	Renderer& operator=(const Renderer&) = delete;

	Renderer(GL::platform::Window& window, Camera& camera, OrbitalNavigator& navigator, int openglVersionMajor, int openglVersionMinor,
		int gridSize, float clothDim, float clothHeight, float clothMass, int fixUpIts, float fixUpStep, int scene);
			 
	~Renderer();

	virtual void render() override;

	virtual void move(int x, int y) override;
	virtual void resize(int width, int height) override;
	virtual void close() override;
	virtual void destroy() override;

	enum class WindType : int {
		NoWind = 0,
		LowSide = 1,
		StrongSide = 2,
		LowBottom = 3,
		StrongBottom = 4,
		MovingSide = 5};

	void initGrid();
	void freeze();
	void setBenchmark(int benchmark);
	void setWriteResult(int write);
	void adjustSimulationSpeed(double factor);
	void switchRendermode();
	void switchScene(int i);
	void toggleWind();
	void setWind(WindType type);

protected:
	GL::platform::Window& window;
	Camera& camera;
	OrbitalNavigator& navigator;

	GL::platform::Context context;
	GL::platform::context_scope<GL::platform::Window> contextScope;

	int viewportWidth{0};
	int viewportHeight{0};

	GL::VertexShader view_vs;
	GL::FragmentShader phong_fs;
	GL::Program phongProg;

	GL::VertexShader clothes_vs;
	GL::GeometryShader clothes_gs;
	GL::Program clothesProg;

	GL::Buffer cameraUniformBuffer;

	GL::Buffer clothesVertexBuffer[4];
	GL::Buffer clothesVertexColor;
	GL::VertexArray clothesVertexArray;

	GL::Buffer sphereVertexBuffer;
	GL::Buffer sphereIndexBuffer;
	uint32_t sphereTriangles;
	GLint spherePos, sphereRadius;
	GL::VertexArray sphereVertexArray;

	std::chrono::high_resolution_clock::time_point refTime;
	float simulationTime;
	float renderTime;

	int gridSize{256};
	int renderMode{0};
	int frameCounter{0};
	int currentVertices{0};

	float clothsizeX{2.f};
	float clothsizeZ{2.f};
	float clothHeight{1.5f};
	float clothMass{0.05f};
	int fixupIterations;
	float fixupPercent;

	std::vector<float4> obstacles;

	WindType currentWind{WindType::NoWind};
	int currentScene{0};

	float windX{0}, windY{0}, windZ{0};
	bool windChanged = false;

	const char* deftitle{"RTG Task 1: Cloth Simulation"};
	const float simulationStep{0.001f};
	double lastDisplayUpdate;
	double lastSimulation{-1.0};
	double simulationSpeed{1.0};
	int maxSimulationsPerFrame{50};
	int benchmark{0};
	int writeResult{0};
	bool frozen{false};

	cudaEvent_t start, stop;
	
	void createGrid(int scene);
	void updateWind();
	
	void drawObstacles();
	void drawClothes();
	void updateSimulation(double now);
	void swapBuffers();

	ClothSimulation sim;
};

#endif