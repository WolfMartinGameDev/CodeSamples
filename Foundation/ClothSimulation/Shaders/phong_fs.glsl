#version 430

#include <framework/utils/GLSL/camera>

in vec3 c;
in vec3 normal;

layout(location = 0) out vec4 color;

void main()
{
	vec3 lightDir = vec3(0, 0, 1.0);
	float diffuse = abs(dot(lightDir, normalize(normal)));

	color = vec4(c * diffuse, 1.0f);
}