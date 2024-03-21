#include "InputHandler.h"

#include "framework/utils/OrbitalNavigator.h"
#include "framework/utils/PerspectiveCamera.h"

#include "GL/platform/Application.h"
#include "GL/platform/Window.h"

#include "Renderer.h"
#include "Scene.h"

#include <iostream>

InputHandler::InputHandler(OrbitalNavigator& navigator, Renderer& renderer, Scene& scene) :
	navigator(navigator),
	renderer(renderer),
	scene(scene)
{
	std::cout << std::endl;
	std::cout << "Hotkeys:" << std::endl
			<< "  <F>     Freeze simulation." << std::endl
		;
	std::cout << std::endl;
}

void InputHandler::keyDown(GL::platform::Key key)
{
}

void InputHandler::keyUp(GL::platform::Key key)
{
	switch (key)
	{
		case GL::platform::Key::BACKSPACE:
			navigator.reset();
			break;
		case GL::platform::Key::C_F:
			scene.freeze(!scene.frozen());
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