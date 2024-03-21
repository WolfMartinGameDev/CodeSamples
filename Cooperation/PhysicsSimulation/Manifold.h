#pragma once
#include <glm/glm.hpp>

enum ContactType {
	VertexFace,
	EdgeEdge,
	FaceFace,
	EdgeFace,
	VertexEdge,
	VertexVertex
};

struct Manifold
{
	Moveable* object;
	Moveable* otherObject;
	glm::vec3 collisionPosition;
	glm::vec3 normal = glm::vec3(0, 0, 0);
	float penetrationDepth  = 0.0f;
	ContactType contactType;

	Manifold() {}
};