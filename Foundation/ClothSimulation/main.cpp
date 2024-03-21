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
				 "	--device N		(use cuda device with index N)\n"
				 "	--frozen		(start with simulation frozen)\n"
				 "	--gridsize N	(use an NxN particle grid)\n"
				 "	--size N		(set the size of the cloth to NxN)\n"
				 "	--height N		(starting height of the cloth)\n"
				 "	--mass N		(mass of the cloth)\n"
				 "	--steps N		(fix up steps in each iteration)\n"
				 "	--fixratio N	(fix up step fix ratio)\n"
				 "	--scene N		(selected scene (1-8)\n"
				 "	--wind N		(selected wind (0-5)\n"
				 "		 0 = NoWind\n"
				 "		 1 = LowSide\n"
				 "		 2 = StrongSide\n"
				 "		 3 = LowBottom\n"
				 "		 4 = StrongBottom\n"
				 "		 5 = MovingSide\n"
				 "	--benchmark N	(number of frames run benchmark)\n"
				 "	--writeframe N	(frame modulo to write out)\n";
}

int main(int argc, char* argv[])
{
	try
	{
		int cudaDevice = 0;
		int gridSize = 256;
		bool startFrozen = false;
		float clothDim = 2.0f;
		float height = 1.5f;
		float mass = 0.05f;
		int steps = 30;
		float fixupPerc = 0.3f;
		int scene = 0;
		int wind = 0;
		int benchmark = 0;
		int writeframe = 0;

		const char frozenToken[] = "--frozen";
		const char deviceToken[] = "--device";
		const char gridSizeToken[] = "--gridsize";
		const char clothDimToken[] = "--size";
		const char heightToken[] = "--height";
		const char massToken[] = "--mass";
		const char stepsToken[] = "--steps";
		const char fixupToken[] = "--fixratio";
		const char sceneToken[] = "--scene";
		const char windToken[] = "--wind";
		const char benchmarkToken[] = "--benchmark";
		const char writeframeToken[] = "--writeframe";

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
					if (!argparse::checkArgument(gridSizeToken, a, gridSize))
					{
						if (!argparse::checkArgument(clothDimToken, a, clothDim))
						{
							if (!argparse::checkArgument(heightToken, a, height))
							{
								if (!argparse::checkArgument(massToken, a, mass))
								{
									if (!argparse::checkArgument(stepsToken, a, steps))
									{
										if (!argparse::checkArgument(fixupToken, a, fixupPerc))
										{
											if (!argparse::checkArgument(sceneToken, a, scene))
											{
												if (!argparse::checkArgument(windToken, a, wind))
												{
													if (!argparse::checkArgument(benchmarkToken, a, benchmark))
													{
														if (!argparse::checkArgument(writeframeToken, a, writeframe))
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
						}
					}
				}
			}
		}

		if (startFrozen)
		{
			std::cout << "starting with frozen simulation" << std::endl;
		}

		int numCudaDevices = 0;

		checkCudaError(cudaGetDeviceCount(&numCudaDevices));

		if (cudaDevice < 0 || cudaDevice >= numCudaDevices)
		{
			std::ostringstream msg;
			msg << "specified cuda device index (" << cudaDevice << ") is not valid!" << std::endl
				<< "		 available cuda devices: " << numCudaDevices << std::endl;
				
			throw InvalidDevice(msg.str());
		}

		cudaDeviceProp props;
		cudaGetDeviceProperties(&props, cudaDevice);
		std::cout << "using cuda device " << cudaDevice << ":" << std::endl;
		std::cout << "	" << props.name << std::endl;
		std::cout << "	with cc " << props.major << "." << props.minor << " @" << props.clockRate / 1000 << "MHz" << std::endl;

		checkCudaError(cudaSetDevice(cudaDevice));

		std::cout << "Simulation setup: \n	" << gridSize << "x" << gridSize
					<< " simulation grid\n	" << clothDim << "x" << clothDim
					<< " cloth at " << height << " height with a mass of " << mass
					<< "\n	" << steps << " fixup iterations with a fixup of "
					<< fixupPerc << " each\n";

		GL::platform::Window window("ClothSimulation", 1024, 768, 0, 0, false);

		PerspectiveCamera camera;

		OrbitalNavigator navigator(-0.75f * math::constants<float>::pi(), 0.22f * math::constants<float>::pi(), 4.2f);

		Renderer renderer(window, camera, navigator, 4, 4, gridSize, clothDim, height, mass, steps, fixupPerc, scene);

		if (wind != 0)
		{
			renderer.setWind(static_cast<Renderer::WindType>(wind));
		}

		if (startFrozen)
		{
			renderer.freeze();
		}

		if (benchmark != 0)
		{
			renderer.setBenchmark(benchmark);
		}

		if(writeframe != 0)
		{
			renderer.setWriteResult(writeframe);
		}

		InputHandler input_handler(navigator, renderer);

		window.attach(static_cast<GL::platform::KeyboardInputHandler*>(&input_handler));
		window.attach(static_cast<GL::platform::MouseInputHandler*>(&input_handler));
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