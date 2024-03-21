#include "InputHandler.h"

#include "framework/utils/OrbitalNavigator.h"
#include "framework/utils/PerspectiveCamera.h"

#include "GL/platform/Application.h"
#include "GL/platform/Window.h"

#include "Renderer.h"

#include <iostream>

InputHandler::InputHandler(OrbitalNavigator& navigator, Renderer& renderer) :
	navigator(navigator),
	renderer(renderer)
{
	std::cout << std::endl;
	std::cout << "Hotkeys:" << std::endl
				<< "	<R>		 Cycle render mode." << std::endl
				<< "	<F>		 Freeze simulation." << std::endl
				<< "	<Space> Reset simulation." << std::endl
				<< "	<1-8>	 Switch Testscenes." << std::endl;
	std::cout << std::endl;
}

void InputHandler::keyDown(GL::platform::Key key)
{
	switch (key)
	{
		case GL::platform::Key::PLUS:
			renderer.adjustSimulationSpeed(1.1);
			break;
		case GL::platform::Key::MINUS:
			renderer.adjustSimulationSpeed(1. / 1.1);
			break;
		default:
			break;
	}
}

void InputHandler::keyUp(GL::platform::Key key)
{
	if (key >= GL::platform::Key::C_1 && key < GL::platform::Key::C_9)
	{
		renderer.switchScene(static_cast<int>(key) - static_cast<int>(GL::platform::Key::C_1));
		return;
	}
	
	switch (key)
	{
		case GL::platform::Key::BACKSPACE:
			navigator.reset();
			break;
		case GL::platform::Key::SPACE:
			renderer.resetParticles();
			break;
		case GL::platform::Key::C_F:
			renderer.freeze();
			break;
		case GL::platform::Key::C_R:
			renderer.cycleRenderMode();
			break;
		case GL::platform::Key::ESCAPE:
			GL::platform::quit();
			break;
		default:
			break;
	}
}

void InputHandler::buttonDown(GL::platform::Button button, int x, int y)
{
}

void InputHandler::buttonUp(GL::platform::Button button, int x, int y)
{
}

void InputHandler::mouseMove(int x, int y)
{
}

void InputHandler::mouseWheel(int delta)
{
}