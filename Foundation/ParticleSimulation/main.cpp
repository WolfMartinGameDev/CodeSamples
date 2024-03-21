#include "InputHandler.h"
#include "Renderer.h"

#include "framework/argparse.h"
#include "framework/cuda/CheckError.h"
#include "framework/math/math.h"
#include "framework/utils/OrbitalNavigator.h"
#include "framework/utils/PerspectiveCamera.h"

#include "GL/platform/Application.h"
#include "GL/platform/Window.h"

#include <cstring>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <type_traits>

struct InvalidDevice : std::runtime_error
{
	explicit InvalidDevice(const std::string& msg) : runtime_error(msg)
	{
	}
};

void printUsage(const char* argv0)
{
	std::cout << "usage: " << argv0
			<< " [options]\n"
				 "options:\n"
				 "	--device N			(use cuda device with index N)\n"
				 "	--frozen			(start with simulation frozen)\n"
				 "	--particles N		(spawn N particles)\n"
				 "	--render-mode N		(start with render mode N		(0=OpenGL, 1=CUDA))\n"
				 "	--camera-theta F	(start with camera angle theta	(in radians))\n"
				 "	--camera-phi F		(start with camera angle phi	(in radians))\n"
				 "	--camera-radius F	(start with camera radius)\n"
			<< std::endl;
}

int main(int argc, char* argv[])
{
	try
	{
		int cudaDevice = 0;
		int numParticles = 32 * 32 * 32;
		bool startFrozen = false;
		int renderMode = 0;
		float cameraTheta = -0.75f * math::constants<float>::pi();
		float cameraPhi = 0.16f * math::constants<float>::pi();
		float cameraRadius = 4.2f;

		const char frozenToken[] = "--frozen";
		const char deviceToken[] = "--device";
		const char numParticlesToken[] = "--particles";
		const char renderModeToken[] = "--render-mode";
		const char cameraThetaToken[] = "--camera-theta";
		const char cameraPhiToken[] = "--camera-phi";
		const char cameraRadiusToken[] = "--camera-radius";

		for (char** a = &argv[1]; *a; ++a)
		{
			if (std::strcmp("--help", *a) == 0)
			{
				printUsage(argv[0]);
				return 0;
			}
			
			if (!argparse::checkArgument(frozenToken, a, startFrozen))
			{
				if (!argparse::checkArgument(deviceToken, a, cudaDevice))
				{
					if (!argparse::checkArgument(numParticlesToken, a, numParticles))
					{
						if (!argparse::checkArgument(renderModeToken, a, renderMode))
						{
							if (!argparse::checkArgument(cameraRadiusToken, a, cameraRadius))
							{
								if (!argparse::checkArgument(cameraPhiToken, a, cameraPhi))
								{
									if (!argparse::checkArgument(cameraThetaToken, a, cameraTheta))
									{
										std::cout << "warning: unknown option " << *a << " will be ignored" << std::endl;
									}
								}
							}
						}
					}
				}
			}
		}

		std::cout << "Starting with frozen simulation: " << startFrozen << std::endl;
		std::cout << "Using cuda device " << cudaDevice << std::endl;

		int numCudaDevices = 0;

		checkCudaError(cudaGetDeviceCount(&numCudaDevices));

		if (cudaDevice < 0 || cudaDevice >= numCudaDevices)
		{
			std::ostringstream msg;
			msg << "Specified cuda device index (" << cudaDevice << ") is not valid!" << std::endl
				<< "			 available cuda devices: " << numCudaDevices << std::endl;
				
			throw InvalidDevice(msg.str());
		}

		checkCudaError(cudaSetDevice(cudaDevice));

		GL::platform::Window window("ParticleSimulation", 800, 600, 0, 0, false);

		PerspectiveCamera camera;

		OrbitalNavigator navigator(cameraTheta, cameraPhi, cameraRadius);

		Renderer renderer(window, camera, navigator, 4, 4, numParticles);

		if (startFrozen)
		{
			renderer.freeze();
		}

		for (int i = 0; i < renderMode; ++i)
		{
			renderer.cycleRenderMode();
		}

		InputHandler inputHandler(navigator, renderer);

		window.attach(static_cast<GL::platform::KeyboardInputHandler*>(&inputHandler));
		window.attach(static_cast<GL::platform::MouseInputHandler*>(&inputHandler));
		window.attach(static_cast<GL::platform::DisplayHandler*>(&renderer));
		window.attach(static_cast<GL::platform::MouseInputHandler*>(&navigator));

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