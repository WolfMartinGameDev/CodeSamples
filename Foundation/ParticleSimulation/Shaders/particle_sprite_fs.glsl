#version 430

#include <framework/utils/GLSL/camera>

in vec3 c;
in vec2 n;

layout(location = 0) out vec4 color;

void main()
{
	float mag = dot(n, n);

	if (mag > 1.0f)
	{
		discard;
	}

	vec3 lightDir = vec3(0, 0, 1.0);
	float diffuse = max(dot(lightDir, vec3(n, sqrt(1.0f - mag))), 0.0f);

	color = vec4(c * diffuse, 1.0f);
}