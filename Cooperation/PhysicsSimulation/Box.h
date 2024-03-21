#pragma once

#include "Moveable.h"
#include <vector>

struct Box : public Moveable
{
	glm::vec3 size;
	glm::vec3 color;
	glm::vec3 halfSize;

	Box(std::string objName = "",
		const glm::vec3& size = glm::vec3(1, 1, 1),
		const glm::vec3& position = glm::vec3(0, 0, 0),
		bool isGravityEnabled = true,
		bool isStatic = false,
		float mass = 1.0f,
		const glm::quat& rotation = glm::quat(1, 0, 0, 0),
		const glm::vec3& linearVelocity = glm::vec3(0, 0, 0),
		const glm::vec3& angularVelocity = glm::vec3(0, 0, 0));

	std::vector<glm::vec3> getVertices();
	std::vector<glm::vec3> getProjectedVertices(glm::vec3 axis);
	glm::vec3 getAxis(int dimension);
	bool tryAxis(Box &other, glm::vec3 axis,
		const glm::vec3& toCentre, unsigned index,
		float& smallestPenetration, unsigned& smallestCase);
	void calcTransformMatrix() override;

};