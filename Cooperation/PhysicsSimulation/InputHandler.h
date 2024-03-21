#ifndef INPUTHANDLER_H_EKRTQGRC
#define INPUTHANDLER_H_EKRTQGRC

#pragma once

#include "GL/platform/InputHandler.h"

class OrbitalNavigator;
class Renderer;
class Scene;

class InputHandler : public virtual GL::platform::KeyboardInputHandler, public virtual GL::platform::MouseInputHandler
{
public:
	InputHandler(OrbitalNavigator& navigator, Renderer& renderer, Scene& scene);
	InputHandler() = default;

	InputHandler(const InputHandler&) = delete;
	InputHandler& operator=(const InputHandler&) = delete;

	void keyDown(GL::platform::Key key);
	void keyUp(GL::platform::Key key);
	void buttonDown(GL::platform::Button button, int x, int y);
	void buttonUp(GL::platform::Button button, int x, int y);
	void mouseMove(int x, int y);
	void mouseWheel(int delta);

private:
	OrbitalNavigator& navigator;
	Renderer& renderer;
	Scene& scene;
};

#endif