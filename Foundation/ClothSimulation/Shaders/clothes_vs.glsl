#version 430

layout(location = 2) uniform ivec2 gridSize;

out ivec2 gridId;

void main()
{
	gridId = ivec2(gl_VertexID % (gridSize.x - 1), gl_VertexID / (gridSize.x - 1));
}