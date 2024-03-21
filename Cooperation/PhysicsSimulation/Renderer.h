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

class Scene;

class Renderer : public GL::platform::Renderer, public GL::platform::DisplayHandler
{
public:
	Renderer(const Renderer&) = delete;
	Renderer& operator=(const Renderer&) = delete;

	Renderer(Scene& scene, GL::platform::Window& window, Camera& camera, OrbitalNavigator& navigator, int openglVersionMajor, int openglVersionMinor);
	~Renderer();

	virtual void render() override;

	virtual void move(int x, int y) override;
	virtual void resize(int width, int height) override;
	virtual void close() override;
	virtual void destroy() override;

	void resetTime();

protected:
	Scene & scene;
	GL::platform::Window& window;
	Camera& camera;
	OrbitalNavigator& navigator;

	GL::platform::Context context;
	GL::platform::context_scope<GL::platform::Window> contextScope;

	int viewportWidth{0};
	int viewportHeight{0};

	GL::VertexShader box_vs;
	GL::FragmentShader phong_fs;
	GL::Program boxProg;

	GL::Buffer cameraUniformBuffer;

	GL::Buffer boxesVertexBuffer;

	size_t boxVertices;
	size_t boxesBuffersize = 1024;
	std::vector<glm::mat4x4> boxTransforms;
	std::vector<glm::vec4> boxColors;
	GL::Buffer boxesTransformationBuffer;
	GL::Buffer boxesColorBuffer;
	GL::VertexArray boxesVertexArray;

	std::chrono::high_resolution_clock::time_point refTime;
	std::chrono::high_resolution_clock::time_point lastDisplayUpdate;

	const char* deftitle{"Simple Physics"};

	void drawBoxes();
	void swapBuffers();
};

#endif