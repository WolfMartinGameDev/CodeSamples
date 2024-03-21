#include "InputHandler.h"
#include "Renderer.h"
#include "Scene.h"

#include "framework/argparse.h"
#include "framework/utils/OrbitalNavigator.h"
#include "framework/utils/PerspectiveCamera.h"

#include "GL/platform/Application.h"
#include "GL/platform/Window.h"

#include <cstring>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <glm/gtx/euler_angles.hpp>

void printUsage(const char* argv0)
{
	std::cout << "usage: " << argv0 << " [options]\n"
	          << "options:\n"
	          << "  --help         (print help)\n"
	          << "  --boxes        (number of boxes)\n"
	          << "  --timestep     (timestep)\n"
	          << "  --frozen       (start frozen)\n";
}

int main(int argc, char* argv[])
{
	try
	{
		bool startFrozen = false;
		int numBoxes = 3;
		float timeStep = 1.0f / 60.0f;

		const char boxesToken[] = "--boxes";
		const char timeStepToken[] = "--timestep";
		const char frozenToken[] = "--frozen";

		for (char** a = &argv[1]; *a; ++a)
		{
			if (std::strcmp("--help", *a) == 0)
			{
				printUsage(argv[0]);
				return 0;
			}
			
			if (!argparse::checkArgument(frozenToken, a, startFrozen))
			{
				if (!argparse::checkArgument(timeStepToken, a, timeStep))
				{
					if (!argparse::checkArgument(boxesToken, a, numBoxes))
					{
						std::cout << "warning: unknown option " << *a << " will be ignored" << std::endl;
					}
				}
			}
		}

		if (startFrozen)
		{
			std::cout << "starting with frozen simulation" << std::endl;
		}

		std::cout << "Simulation setup: \n"
		          << "  " << numBoxes << " boxes\n"
		          << "  simulation step " << timeStep * 1000 << "ms\n";

		GL::platform::Window window("Simple Physics", 1024, 768, 0, 0, false, false);

		PerspectiveCamera camera(60.0f * glm::pi<float>() / 180.0f, 0.1f, 1000.0f);

		OrbitalNavigator navigator( 0.5f * glm::pi<float>(),
									0.12f * glm::pi<float>(), 20.0f);

		Scene scene(timeStep, startFrozen);

		srand(10000);

		for (int i = 0; i < numBoxes; ++i)
		{
			scene.addBox(Box("Box", glm::vec3(1, 1, 1), glm::vec3(-numBoxes + 1 + 3 * i, 0, 0)));
		}
		
		scene.addBox(Box("Ground", glm::vec3(20, 1, 20), glm::vec3(0.0f, -5.0f, 0.0f), false, true, 0.0f, glm::quat(1.0, 0, 0, 0), glm::vec3(0, 0, 0), glm::vec3(0, 0 ,0)));
		scene.addBox(Box("tilted Box", glm::vec3(1, 1, 1), glm::vec3(2.0f, 2.0f, 0.0f), true, false, 1.0f, glm::quat(vec3(0.0f, 0.0f, 0.0f)), glm::vec3(0, 0, 0), glm::vec3(0.0f, 0 ,0)));
		scene.addBox(Box("tilted Box", glm::vec3(1, 1, 1), glm::vec3(0.0f, 2.0f, 0.0f), true, false, 1.0f, glm::quat(vec3(0.0f, 0.0f, 0.0f)), glm::vec3(0, 0, 0), glm::vec3(0.0f, 0 ,0)));
		scene.addBox(Box("tilted Box", glm::vec3(1, 1, 1), glm::vec3(-2.0f, 2.0f, 0.0f), true, false, 1.0f, glm::quat(vec3(0.0f, 0.0f, 0.0f)), glm::vec3(0, 0, 0), glm::vec3(0.0f, 0 ,0)));
		scene.addBox(Box("tilted Box", glm::vec3(1, 1, 1), glm::vec3(-1.4f, 5.0f, 0.0f), true, false, 1.0f, glm::quat(vec3(20.0f, 0.0f, 0.0f)), glm::vec3(0, 0, 0), glm::vec3(0.0f, 0 ,0)));

		Renderer renderer(scene, window, camera, navigator, 4, 4);

		InputHandler inputHandler(navigator, renderer, scene);

		window.attach(static_cast<GL::platform::KeyboardInputHandler*>(&inputHandler));
		window.attach(static_cast<GL::platform::MouseInputHandler*>(&inputHandler));
		window.attach(static_cast<GL::platform::DisplayHandler*>(&renderer));
		window.attach(static_cast<GL::platform::MouseInputHandler*>(&navigator));

		renderer.resetTime();
		GL::platform::run(renderer);
	}
	catch (const argparse::usage_error& e)
	{
		std::cout << "error: " << e.what() << std::endl;
		printUsage(argv[0]);
		return 1;
	}
	catch (const std::exception& e)
	{
		std::cout << "error: " << e.what() << std::endl;
		return -1;
	}
	catch (...)
	{
		std::cout << "unknown exception" << std::endl;
		return -128;
	}

	return 0;
}