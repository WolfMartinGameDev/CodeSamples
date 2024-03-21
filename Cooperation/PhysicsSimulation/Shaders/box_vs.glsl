#version 430

#include <framework/utils/GLSL/camera>

layout(std430, column_major, binding = 0) buffer Transformation
{
	mat4x4 transformation[];
};

layout(std430, binding = 1) buffer Colors
{
	vec4 colors[];
};

struct VertexData
{
	vec4 position;
	vec4 normal;
};

layout(std430, binding = 2) buffer VertexDataBlock
{
	VertexData vertices[36];
};

out vec3 c;
out vec3 normal;

void main()
{
	uint localVertex = uint(gl_VertexID) % 36;
	uint boxId = uint(gl_VertexID) / 36;
	VertexData myVertex = vertices[localVertex];
	vec4 p = transformation[boxId] * myVertex.position;
	
	gl_Position = camera.PV * (transformation[boxId] * myVertex.position);
	normal = (camera.V * (transformation[boxId] * myVertex.normal)).xyz;
	c = colors[boxId].xyz;
}