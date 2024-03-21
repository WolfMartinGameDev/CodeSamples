#version 430

#include <framework/utils/GLSL/camera>

uniform ivec2 gridSize;

layout(std430, binding = 3) buffer VertexBuffer
{
	vec4 positions[];
};
layout(std430, binding = 4) buffer ColorBuffer
{
	vec4 colors[];
};

layout(points) in;
layout(triangle_strip, max_vertices = 4) out;

in ivec2 gridId[];

out vec3 normal;
out vec3 c;

int arrayAccess(ivec2 location)
{
	return location.x + location.y * gridSize.x;
}

vec3 colorAccess(ivec2 location)
{
	return colors[arrayAccess(location)].xyz;
}

vec3 vertexAccess(ivec2 location)
{
	return positions[arrayAccess(location)].xyz;
}

void computeVertex(ivec2 p)
{
	vec3 pos = vertexAccess(p);
	
	c = colorAccess(p);

	vec3 left = vertexAccess(ivec2(max(0, p.x - 1), p.y)) - pos;
	vec3 up = vertexAccess(ivec2(p.x, min(gridSize.y - 1, p.y + 1))) - pos;
	vec3 right = vertexAccess(ivec2(min(gridSize.x - 1, p.x + 1), p.y)) - pos;
	vec3 down = vertexAccess(ivec2(p.x, max(0, p.y - 1))) - pos;

	normal = cross(left, up) + cross(up, right) + cross(right, down) + cross(down, left);
	normal = normalize(normal);

	gl_Position = camera.PV * vec4(pos, 1);
	normal = (camera.V * vec4(normal,0)).xyz;
	EmitVertex();
}

void main()
{
	ivec2 p = gridId[0];

	computeVertex(p);
 
	p.x = gridId[0].x + 1;
	computeVertex(p);

	p.x = gridId[0].x;
	p.y = gridId[0].y + 1;
	computeVertex(p);

	p.x = gridId[0].x + 1;
	computeVertex(p);
 
	EndPrimitive();
}