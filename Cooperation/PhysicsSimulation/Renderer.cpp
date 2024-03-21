

#include "Renderer.h"

#include "Scene.h"

#include "GL/error.h"
#include "GL/platform/Application.h"
#include "GL/platform/Window.h"

#include "framework/utils/Box.h"
#include <glm/gtc/matrix_access.hpp>
#include <glm/gtx/matrix_operation.hpp>
#include <glm/gtx/quaternion.hpp>

#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>

// generated from the shaders by the build system
extern const char box_vs[];
extern const char view_vs[];
extern const char phong_fs[];

Renderer::Renderer(Scene& scene, GL::platform::Window& window, Camera& camera, OrbitalNavigator& navigator, int openglVersionMajor, int openglVersionMinor) :
	scene(scene),
	GL::platform::Renderer(),
	window(window),
	camera(camera),
	navigator(navigator),
	context(window.createContext(openglVersionMajor, openglVersionMinor, true)),
	contextScope(context, window)
{
	glClearColor(0.1f, 0.3f, 1.0f, 1.0f);
	glClearDepth(1.0f);
	glEnable(GL_DEPTH_TEST);
	GL::checkError();

	box_vs = GL::compileVertexShader(::box_vs);
	phong_fs = GL::compileFragmentShader(::phong_fs);

	glAttachShader(boxProg, box_vs);
	glAttachShader(boxProg, phong_fs);
	GL::linkProgram(boxProg);

	glBindBuffer(GL_UNIFORM_BUFFER, cameraUniformBuffer);
	glBufferData(GL_UNIFORM_BUFFER, sizeof(Camera::UniformBuffer), nullptr, GL_STATIC_DRAW);

	GL::checkError();

	glBindVertexArray(boxesVertexArray);
	glUseProgram(boxProg);

	GLuint boxesCameraParameters = glGetUniformBlockIndex(boxProg, "CameraParameters");
	glUniformBlockBinding(boxProg, boxesCameraParameters, 0);

	glBindBuffer(GL_UNIFORM_BUFFER, cameraUniformBuffer);

	GL::checkError();

	BoxVertices box;
	std::vector<float> boxVertexNormals;
	boxVertices = box.numVertices();
	boxVertexNormals.reserve(boxVertices * 8);
	const float* v = box.getVertices();
	const float* n = box.getNormals();
	for (size_t i = 0; i < boxVertices; ++i)
	{
		boxVertexNormals.push_back(v[4 * i + 0]);
		boxVertexNormals.push_back(v[4 * i + 1]);
		boxVertexNormals.push_back(v[4 * i + 2]);
		boxVertexNormals.push_back(v[4 * i + 3]);

		boxVertexNormals.push_back(n[4 * i + 0]);
		boxVertexNormals.push_back(n[4 * i + 1]);
		boxVertexNormals.push_back(n[4 * i + 2]);
		boxVertexNormals.push_back(n[4 * i + 3]);
	}

	
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, boxesVertexBuffer);
	glBufferData(GL_SHADER_STORAGE_BUFFER, boxVertexNormals.size() * 4, &boxVertexNormals[0], GL_STATIC_DRAW);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, boxesVertexBuffer);

	boxTransforms.resize(boxesBuffersize);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, boxesTransformationBuffer);
	glBufferData(GL_SHADER_STORAGE_BUFFER, boxesBuffersize * 16 * 4, nullptr, GL_DYNAMIC_DRAW);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, boxesTransformationBuffer);

	boxColors.resize(boxesBuffersize);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, boxesColorBuffer);
	glBufferData(GL_SHADER_STORAGE_BUFFER, boxesBuffersize * 4 * 4, nullptr, GL_DYNAMIC_DRAW);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, boxesColorBuffer);

	GL::checkError();
	glBindVertexArray(0);

	glDisable(GL_CULL_FACE);

	window.attach(this);
	resetTime();
}

Renderer::~Renderer()
{
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
}

void Renderer::drawBoxes()
{
	boxTransforms.clear();
	boxColors.clear();
	scene.visitBoxes([&, this](const Box& b)
	{
		glm::mat4x4 scale = glm::diagonal4x4(glm::vec4(b.size, 1.0f));
		glm::mat4x4 rotation = glm::toMat4(b.rotation);
		glm::mat4x4 position = column(glm::diagonal4x4(glm::vec4(1.0f)), 3, glm::vec4(b.position, 1.0f));
		boxTransforms.push_back(position * (rotation * scale));
		boxColors.push_back(glm::vec4(b.color, 1.0f));
	});
	
	if (boxTransforms.size() == 0)
	{
		return;
	}

	if (boxTransforms.size() < boxesBuffersize)
	{
		glBindBuffer(GL_UNIFORM_BUFFER, boxesTransformationBuffer);
		glBufferSubData(GL_UNIFORM_BUFFER, 0, boxesBuffersize * 16 * 4, &boxTransforms[0]);
		glBindBuffer(GL_UNIFORM_BUFFER, boxesColorBuffer);
		glBufferSubData(GL_UNIFORM_BUFFER, 0, boxesBuffersize * 4 * 4, &boxColors[0]);
	}
	else
	{
		while(boxTransforms.size() < boxesBuffersize)
		{
			boxesBuffersize = 2 * boxesBuffersize;
		}
		
		boxTransforms.reserve(boxesBuffersize);
		boxColors.reserve(boxesBuffersize);
		glBindBuffer(GL_UNIFORM_BUFFER, boxesTransformationBuffer);
		glBufferData(GL_UNIFORM_BUFFER, boxesBuffersize * 16 * 4, &boxTransforms[0], GL_DYNAMIC_DRAW);
		glBindBuffer(GL_UNIFORM_BUFFER, boxesColorBuffer);
		glBufferData(GL_UNIFORM_BUFFER, boxesBuffersize * 4 * 4, &boxColors[0], GL_DYNAMIC_DRAW);
	}

	glUseProgram(boxProg);
	glBindVertexArray(boxesVertexArray);
	glDrawArrays(GL_TRIANGLES, 0, boxVertices * boxTransforms.size());
}

void Renderer::swapBuffers()
{
	contextScope.swapBuffers();
}

void Renderer::render()
{
	auto now = std::chrono::high_resolution_clock::now();

	double dt = std::chrono::duration_cast<std::chrono::duration<double>>(now - refTime).count();
	scene.update(static_cast<float>(dt));
	auto afterUpdate = std::chrono::high_resolution_clock::now();
	

	Camera::UniformBuffer camParams;
	camera.writeUniformBuffer(&camParams, navigator, static_cast<float>(viewportWidth) / viewportHeight);

	GL::checkError();
	glBindBuffer(GL_UNIFORM_BUFFER, cameraUniformBuffer);
	glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(Camera::UniformBuffer), &camParams);
	glBindBufferRange(GL_UNIFORM_BUFFER, 0, cameraUniformBuffer, 0, sizeof(Camera::UniformBuffer));

	GL::checkError();
	glViewport(0, 0, viewportWidth, viewportHeight);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	drawBoxes();
	glFinish();
	auto afterDraw = std::chrono::high_resolution_clock::now();

	if (std::chrono::duration_cast<std::chrono::duration<double>>(now - lastDisplayUpdate).count() > 1.0)
	{
		lastDisplayUpdate = now;
		std::stringstream info;

		info << deftitle;
		if (!scene.frozen())
		{
			double simulationTime = std::chrono::duration_cast<std::chrono::duration<double>>(afterUpdate - now).count();
			double drawTime = std::chrono::duration_cast<std::chrono::duration<double>>(afterDraw - afterUpdate).count();
			info << "  sim: " << std::setprecision(5) << (simulationTime * 1000) << "ms";
			info << "  draw: " << std::setprecision(3) << drawTime << "ms";
		}
		else
		{
			info << "frozen ";
		}
		
		window.title(info.str().c_str());
	}

	GL::checkError();
	swapBuffers();
	refTime = now;
}

void Renderer::close()
{
	GL::platform::quit();
}

void Renderer::destroy()
{
	GL::platform::quit();
}

void Renderer::resetTime()
{
	refTime = std::chrono::high_resolution_clock::now();
}

void Renderer::move(int x, int y)
{
}