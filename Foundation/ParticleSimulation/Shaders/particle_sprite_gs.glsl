#version 430

#include <framework/utils/GLSL/camera>

layout(points) in;
layout(triangle_strip, max_vertices = 5) out;

uniform float radius;

in vec4 position[];
in vec3 color[];

out vec2 n;
out vec3 c;

void main()
{
	c = color[0];

	vec4 p = position[0];
	float xps = radius * camera.P[0][0];
	float yps = radius * camera.P[1][1];

	n = vec2(-1, -1);
	gl_Position = p + vec4(-xps, -yps, 0, 0); // left bottom
	EmitVertex();
	
	n = vec2(1, -1);
	gl_Position = p + vec4(xps, -yps, 0, 0); // right bottom
	EmitVertex();
	
	n = vec2(-1, 1);
	gl_Position = p + vec4(-xps, yps, 0, 0); // left top
	EmitVertex();
	
	n = vec2(1, 1);
	gl_Position = p + vec4(xps, yps, 0, 0); // right top
	EmitVertex();
	
	EndPrimitive();
}