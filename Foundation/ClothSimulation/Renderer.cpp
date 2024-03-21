#include "Renderer.h"

#include "GL/error.h"
#include "GL/platform/Application.h"
#include "GL/platform/Window.h"

#include "framework/math/vector.h"

#include "framework/utils/Sphere.h"

#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>

// generated from the shaders by the build system
extern const char clothes_vs[];
extern const char clothes_gs[];
extern const char view_vs[];
extern const char phong_fs[];

Renderer::Renderer(GL::platform::Window& window, Camera& camera, OrbitalNavigator& navigator, int openglVersionMajor, int openglVersionMinor,
	int gridSize, float clothDim, float clothHeight, float clothMass, int fixUpIts, float fixUpStep, int scene) :
	GL::platform::Renderer(),
	window(window),
	camera(camera),
	navigator(navigator),
	context(window.createContext(openglVersionMajor, openglVersionMinor, true)),
	contextScope(context, window),
	gridSize(gridSize),
	clothsizeX(clothDim),
	clothsizeZ(clothDim),
	clothHeight(clothHeight),
	clothMass(clothMass),
	fixupIterations(fixUpIts),
	fixupPercent(fixUpStep),
	sim(gridSize, clothsizeX, clothsizeZ, clothMass, 0.995f, -9.81f, fixUpIts, fixUpStep)
{
	glClearColor(0.1f, 0.3f, 1.0f, 1.0f);
	glClearDepth(1.0f);
	glEnable(GL_DEPTH_TEST);
	GL::checkError();

	clothes_vs = GL::compileVertexShader(::clothes_vs);
	clothes_gs = GL::compileGeometryShader(::clothes_gs);
	phong_fs = GL::compileFragmentShader(::phong_fs);
	view_vs = GL::compileVertexShader(::view_vs);

	glAttachShader(clothesProg, clothes_vs);
	glAttachShader(clothesProg, clothes_gs);
	glAttachShader(clothesProg, phong_fs);
	GL::linkProgram(clothesProg);

	glAttachShader(phongProg, view_vs);
	glAttachShader(phongProg, phong_fs);
	GL::linkProgram(phongProg);

	glBindBuffer(GL_UNIFORM_BUFFER, cameraUniformBuffer);
	glBufferData(GL_UNIFORM_BUFFER, sizeof(Camera::UniformBuffer), nullptr, GL_STATIC_DRAW);

	GL::checkError();

	glBindVertexArray(clothesVertexArray);
	glUseProgram(clothesProg);
	GLuint clothesCameraParameters = glGetUniformBlockIndex(clothesProg, "CameraParameters");
	glUniformBlockBinding(clothesProg, clothesCameraParameters, 0);

	glBindBuffer(GL_UNIFORM_BUFFER, cameraUniformBuffer);

	GL::checkError();

	IcoSphere sphere(6);

	glBindVertexArray(sphereVertexArray);
	glUseProgram(phongProg);
	glBindBuffer(GL_ARRAY_BUFFER, sphereVertexBuffer);
	glBindVertexBuffer(0U, sphereVertexBuffer, 0U, sizeof(float) * 4);
	glEnableVertexAttribArray(0U);

	glVertexAttribFormat(0U, 4, GL_FLOAT, GL_FALSE, 0U);
	glVertexAttribBinding(0U, 0U);

	glBindBuffer(GL_ARRAY_BUFFER, sphereVertexBuffer);
	glBufferData(GL_ARRAY_BUFFER, sphere.numVertices() * 4 * 4, sphere.getVertices(), GL_STATIC_DRAW);

	sphereTriangles = static_cast<uint32_t>(sphere.numTriangles());
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, sphereIndexBuffer);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sphere.numTriangles() * 4 * 3, sphere.getInidices(), GL_STATIC_DRAW);

	spherePos = glGetUniformLocation(phongProg, "pos");
	sphereRadius = glGetUniformLocation(phongProg, "radius");

	glBindVertexArray(0);

	GL::checkError();

	createGrid(scene);
	GL::checkError();

	glDisable(GL_CULL_FACE);

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	window.attach(this);
}

Renderer::~Renderer()
{
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}

void Renderer::setBenchmark(int benchmark)
{
	this->benchmark = benchmark;
}

void Renderer::setWriteResult(int write)
{
	writeResult = write;
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
}

void Renderer::swapBuffers()
{
	contextScope.swapBuffers();
}

void Renderer::drawObstacles()
{
	glUseProgram(phongProg);

	glBindVertexArray(sphereVertexArray);

	for (auto& info : obstacles)
	{
		glUniform3f(spherePos, info.x, info.y, info.z);
		glUniform1f(sphereRadius, 0.96f * info.w);
		glDrawElements(GL_TRIANGLES, 3 * sphereTriangles, GL_UNSIGNED_INT, nullptr);
	}

	GL::checkError();
}

void Renderer::drawClothes()
{
	glUseProgram(clothesProg);

	if (renderMode)
	{
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	}

	glBindVertexArray(clothesVertexArray);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, clothesVertexBuffer[currentVertices]);
	glDrawArrays(GL_POINTS, 0, (gridSize - 1) * (gridSize - 1));
	GL::checkError();

	if (renderMode)
	{
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	}
}

void Renderer::createGrid(int scene)
{
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, clothesVertexBuffer[0]);
	glBufferData(GL_SHADER_STORAGE_BUFFER, gridSize * gridSize * 4 * 4, nullptr, GL_STATIC_DRAW);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, clothesVertexBuffer[1]);
	glBufferData(GL_SHADER_STORAGE_BUFFER, gridSize * gridSize * 4 * 4, nullptr, GL_STATIC_DRAW);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, clothesVertexBuffer[2]);
	glBufferData(GL_SHADER_STORAGE_BUFFER, gridSize * gridSize * 4 * 4, nullptr, GL_STATIC_DRAW);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, clothesVertexBuffer[3]);
	glBufferData(GL_SHADER_STORAGE_BUFFER, gridSize * gridSize * 4 * 4, nullptr, GL_STATIC_DRAW);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, clothesVertexColor);
	glBufferData(GL_SHADER_STORAGE_BUFFER, gridSize * gridSize * 4 * 4, nullptr, GL_STATIC_DRAW);

	GL::checkError();

	glBindVertexArray(clothesVertexArray);
	glUseProgram(clothesProg);
	GL::checkError();
	GLuint clothesColorBuffer = glGetProgramResourceIndex(clothesProg, GL_SHADER_STORAGE_BLOCK, "ColorBuffer");
	glShaderStorageBlockBinding(clothesProg, clothesColorBuffer, 1);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, clothesVertexColor);
	GL::checkError();

	GLuint vertexBuffer = glGetProgramResourceIndex(clothesProg, GL_SHADER_STORAGE_BLOCK, "VertexBuffer");
	glShaderStorageBlockBinding(clothesProg, vertexBuffer, 2);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, clothesVertexBuffer[0]);
	GL::checkError();

	GLint gridsize = glGetUniformLocation(clothesProg, "gridSize");
	glUniform2i(gridsize, gridSize, gridSize);
	GL::checkError();

	initGrid();

	sim.registerBuffers(clothesVertexBuffer);

	switchScene(scene);
}

void Renderer::initGrid()
{
	currentVertices = 0;

	std::vector<math::float4> initPositions;
	initPositions.reserve(gridSize * gridSize);
	std::vector<math::float4> colors;
	colors.reserve(gridSize * gridSize);
	float xstep = clothsizeX / (gridSize - 1);
	float zstep = clothsizeZ / (gridSize - 1);
	float inv_gridSize = 1.0f / (gridSize - 1);
	
	for (size_t j = 0; j < gridSize; ++j)
	{
		for (size_t i = 0; i < gridSize; ++i)
		{
			initPositions.emplace_back(math::float4(-0.5f * clothsizeX + i * xstep, clothHeight, -0.5f * clothsizeZ + j * zstep, 1));
			colors.emplace_back(math::float4(i * inv_gridSize, j * inv_gridSize, 0.5f, 1.0f));
		}
	}

	glBindBuffer(GL_SHADER_STORAGE_BUFFER, clothesVertexColor);
	glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, gridSize * gridSize * 4 * 4, &colors[0]);
	
	for (int i = 0; i < 4; ++i)
	{
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, clothesVertexBuffer[i]);
		glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, gridSize * gridSize * 4 * 4, &initPositions[0]);
	}
	
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void Renderer::updateSimulation(double now)
{
	sim.mapBuffers();
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

	if (benchmark > 0 || writeResult > 0)
	{
		numSimulation = 1;
	}

	for (int i = 0; i < numSimulation; ++i)
	{
		if (currentWind == WindType::MovingSide)
		{
			static std::random_device rd;
			static std::mt19937 gen(rd());
			float perParticleFactor = clothMass / (128 * gridSize);
			std::uniform_real_distribution<float> dist(-20.0f * perParticleFactor, 20.0f * perParticleFactor);
			windX += dist(gen);
			windZ += dist(gen);
			float maxwind = 5000.0f * perParticleFactor;
			windX = std::min(maxwind, std::max(-maxwind, windX));
			windZ = std::min(maxwind, std::max(-maxwind, windZ));
			windChanged = true;
		}
		
		cudaEventRecord(start);
		sim.simulate(simulationStep, windChanged, windX, windY, windZ, currentVertices);
		windChanged = false;
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float ms;
		cudaEventElapsedTime(&ms, start, stop);
		simulationTime = ms / 1000.0f;
		
		if (writeResult > 0 && frameCounter % writeResult == 0)
		{
			static std::ofstream fsim("frame_data.bin", std::fstream::binary);
			if (!fsim)
			{
				throw std::runtime_error("could not open frame_data output file");
			}

			std::vector<float4> tempPosBuffer(gridSize * gridSize);
			sim.copyPositionsToCPU(&tempPosBuffer[0], currentVertices);
			fsim.write(reinterpret_cast<const char*>(&tempPosBuffer[0]), gridSize * gridSize * 4 * 4);
			fsim.flush();
		}
	}
	sim.unmapBuffers();
}

void Renderer::adjustSimulationSpeed(double factor)
{
	simulationSpeed *= factor;
}

void Renderer::switchRendermode()
{
	renderMode = (renderMode + 1) % 2;
}

void Renderer::switchScene(int i)
{
	initGrid();
	std::vector<int> fixedNodes;

	obstacles.clear();

	currentScene = i;

	switch (i)
	{
		default:
		case 0:
			std::cout << "Scene 0: 2 fixed vertices\n";
			fixedNodes.push_back(0);
			fixedNodes.push_back(gridSize - 1);
			break;
		case 1:
			std::cout << "Scene 1: 1 fixed vertices\n";
			fixedNodes.push_back(0);
			break;
		case 2:
			std::cout << "Scene 2: 3 fixed vertices\n";
			fixedNodes.push_back(0);
			fixedNodes.push_back(gridSize - 1);
			fixedNodes.push_back(gridSize * gridSize - 1);
			break;
		case 3:
			std::cout << "Scene 3: 2 fixed vertices and an obstacle\n";
			fixedNodes.push_back(0);
			fixedNodes.push_back(gridSize - 1);
			obstacles.push_back(make_float4(0, 0.5f, -0.1f, 0.2f));
			break;
		case 4:
			std::cout << "Scene 4: 2 fixed vertices and 2 obstacles\n";
			fixedNodes.push_back(0);
			fixedNodes.push_back(gridSize - 1);
			obstacles.push_back(make_float4(-0.3f, 0.5f, -0.1f, 0.2f));
			obstacles.push_back(make_float4(0.3f, 0.5f, -0.1f, 0.2f));
			break;
		case 5:
			std::cout << "Scene 5: Free falling and an obstacle\n";
			obstacles.push_back(make_float4(0, 0, 0, 0.5f));
			break;
		case 6:
			std::cout << "Scene 6: Free falling and 4 obstacles\n";
			obstacles.push_back(make_float4(-0.3f, 0, -0.3f, 0.25f));
			obstacles.push_back(make_float4(-0.3f, 0, 0.3f, 0.25f));
			obstacles.push_back(make_float4(0.3f, 0, -0.3f, 0.25f));
			obstacles.push_back(make_float4(0.3f, 0, 0.3f, 0.25f));
			break;
		case 7:
			std::cout << "Scene 7: Free falling and 3 obstaclse\n";
			obstacles.push_back(make_float4(-0.3f, 0.2f, -0.3f, 0.25f));
			obstacles.push_back(make_float4(0.2f, -0.3f, 0.4f, 0.25f));
			obstacles.push_back(make_float4(0.4f, -0.8f, -0.1f, 0.25f));
			break;
	}

	sim.newCloth(gridSize, clothsizeX, clothsizeZ, clothMass, fixupIterations, fixupPercent, fixedNodes, obstacles);
}

void Renderer::setWind(WindType type)
{
	currentWind = type;
	updateWind();
}

void Renderer::toggleWind()
{
	float perParticleFactor = clothMass / (128 * gridSize);
	
	switch (currentWind)
	{
		case WindType::NoWind:
			currentWind = WindType::LowSide;
			break;
		case WindType::LowSide:
			currentWind = WindType::StrongSide;
			break;
		case WindType::StrongSide:
			currentWind = WindType::LowBottom;
			break;
		case WindType::LowBottom:
			currentWind = WindType::StrongBottom;
			break;
		case WindType::StrongBottom:
			currentWind = WindType::MovingSide;
			break;
		case WindType::MovingSide:
			currentWind = WindType::NoWind;
			break;
	}
	
	updateWind();
}

void Renderer::updateWind()
{
	windChanged = true;
	float perParticleFactor = clothMass / (128 * gridSize);
	
	switch (currentWind)
	{
		case WindType::LowSide:
			std::cout << "low side wind\n";
			windY = 0;
			windX = 25.0f * perParticleFactor;
			windZ = 1500.0f * perParticleFactor;
			break;
		case WindType::StrongSide:
			std::cout << "strong side wind\n";
			windY = 0;
			windX = 70.0f * perParticleFactor;
			windZ = 2800.0f * perParticleFactor;
			break;
		case WindType::LowBottom:
			std::cout << "low bottom wind\n";
			windY = 1600.0f * perParticleFactor;
			windX = 0;
			windZ = 0;
			break;
		case WindType::StrongBottom:
			std::cout << "strong bottom wind\n";
			windY = 3600.0f * perParticleFactor;
			windX = 0;
			windZ = 0;
			break;
		case WindType::MovingSide:
			std::cout << "moving side wind\n";
			windY = 0;
			windX = 20.0f * perParticleFactor;
			windZ = 2800.0f * perParticleFactor;
			break;
		case WindType::NoWind:
			std::cout << "no wind\n";
			windX = windY = windZ = 0;
			break;
	}
}

void Renderer::render()
{
	double now = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - refTime).count();
	
	if (!frozen)
	{
		updateSimulation(now);
	}

	Camera::UniformBuffer camParams;
	camera.writeUniformBuffer(&camParams, navigator, static_cast<float>(viewportWidth) / viewportHeight);

	glBindBuffer(GL_UNIFORM_BUFFER, cameraUniformBuffer);
	glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(Camera::UniformBuffer), &camParams);
	glBindBufferRange(GL_UNIFORM_BUFFER, 0, cameraUniformBuffer, 0, sizeof(Camera::UniformBuffer));

	glViewport(0, 0, viewportWidth, viewportHeight);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	drawClothes();
	drawObstacles();
	frameCounter++;

	if (now - lastDisplayUpdate > 1.0)
	{
		lastDisplayUpdate = now;
		std::stringstream info;
		
		if (!frozen)
		{
			info << "sim: " << std::setprecision(5) << (simulationTime * 1000) << "ms	";
			info << "sim-speed: " << std::setprecision(3) << simulationSpeed << "x	";
		}
		else
		{
			info << "frozen	";
		}
		
		info << "scene: " << currentScene << "	";
		info << deftitle;
		window.title(info.str().c_str());
	}

	GL::checkError();
	swapBuffers();

	if(benchmark > 0 && frameCounter >= benchmark)
	GL::platform::quit();
}

void Renderer::close()
{
	GL::platform::quit();
}

void Renderer::destroy()
{
	GL::platform::quit();
}

void Renderer::move(int x, int y)
{
}