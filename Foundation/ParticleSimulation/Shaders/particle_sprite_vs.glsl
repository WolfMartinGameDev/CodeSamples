#version 430

#include <framework/utils/GLSL/camera>

layout(location = 0) in vec3 vPosition;
layout(location = 1) in vec3 vColor;

out vec4 position;
out vec3 color;

void main()
{
	position = camera.PV * vec4(vPosition, 1.0f);
	color = vColor;
}