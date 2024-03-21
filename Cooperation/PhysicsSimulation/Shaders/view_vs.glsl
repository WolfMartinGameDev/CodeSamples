#version 430

#include <framework/utils/GLSL/camera>

layout(location = 0) in vec4 vPosition;

uniform vec3 pos;
uniform float radius;

out vec3 c;
out vec3 normal;

void main()
{
	gl_Position = camera.PV * vec4(pos + radius * vPosition.xyz, 1.0);
	normal = camera.V * vec4(vPosition.xyz, 0.0);
	c = vec3(0.7, 0.7, 0.8);
}