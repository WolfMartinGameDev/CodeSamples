#include "Box.h"
#include <glm/gtx/projection.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtc/matrix_access.hpp>
#include <glm/gtx/matrix_operation.hpp>
#include <iostream>

Box::Box(
	std::string objName,
	const glm::vec3 & size,
	const glm::vec3 & position,
	bool isGravityEnabled,
	bool isStatic,
	float mass,
	const glm::quat & rotation,
	const glm::vec3 & linearVelocity,
	const glm::vec3 & angularVelocity) :
	Moveable(objName, position, isGravityEnabled, isStatic, mass, rotation, linearVelocity, angularVelocity),
	size(size),
	color(0.7f, 0.9f, 0.7f)
{
	if (mass != 0)
	{
		invMass = 1 / mass;
	} else
	{
		invMass = 0;
	}
		
	halfSize = size / 2.0f;

	if (mass == 0)
	{
		inertiaTensor = glm::mat3x3(0.0f, 0.0f, 0.0f,
									0.0f, 0.0f, 0.0f,
									0.0f, 0.0f, 0.0f);
		inverseInertiaTensor = inertiaTensor;
	}
	else
	{
		inertiaTensor = glm::mat3x3((mass * (size.y * size.y + size.z * size.z) / 12.0f), 0.0f, 0.0f,
									0.0f, (mass * (size.x * size.x + size.z * size.z) / 12.0f), 0.0f,
									0.0f, 0.0f, (mass * (size.x * size.x + size.y * size.y) / 12.0f));
		inverseInertiaTensor = glm::inverse(inertiaTensor);
	}
	
}

std::vector<glm::vec3> Box::getVertices()
{
	std::vector<glm::vec3> vertices;
	vertices.push_back(glm::vec3(size.x / 2.0f, size.y / 2.0f, size.z / 2.0f));
	vertices.push_back(glm::vec3(-size.x / 2.0f, size.y / 2.0f, size.z / 2.0f));
	vertices.push_back(glm::vec3(size.x / 2.0f, -size.y / 2.0f, size.z / 2.0f));
	vertices.push_back(glm::vec3(-size.x / 2.0f, -size.y / 2.0f, size.z / 2.0f));
	vertices.push_back(glm::vec3(size.x / 2.0f, size.y / 2.0f, -size.z / 2.0f));
	vertices.push_back(glm::vec3(-size.x / 2.0f, size.y / 2.0f, -size.z / 2.0f));
	vertices.push_back(glm::vec3(size.x / 2.0f, -size.y / 2.0f, -size.z / 2.0f));
	vertices.push_back(glm::vec3(-size.x / 2.0f, -size.y / 2.0f, -size.z / 2.0f));

	for (auto& vertex : vertices)
	{
		glm::rotate(rotation, vertex);
		vertex += position;
	}

	return vertices;
}

std::vector<glm::vec3> Box::getProjectedVertices(glm::vec3 axis)
{
	std::vector<glm::vec3> vertices = getVertices();
	std::vector<glm::vec3> projectedVertices;

	for (auto& vertex : vertices)
	{
		projectedVertices.push_back(glm::proj(vertex, axis));
	}
	
	return projectedVertices;
}

glm::vec3 Box::getAxis(int dimension)
{
	return glm::vec3(transformMatrix[dimension][0], transformMatrix[dimension][1], transformMatrix[dimension][2]);
}

bool Box::tryAxis(Box &other, glm::vec3 axis, const glm::vec3& toCentre, unsigned index, float& smallestPenetration, unsigned& smallestCase)
{
	float oneProject = this->halfSize.x * abs(glm::dot(axis, this->getAxis(0))) +
					this->halfSize.y * abs(glm::dot(axis, this->getAxis(1))) +
					this->halfSize.z * abs(glm::dot(axis, this->getAxis(2)));

	float twoProject = other.halfSize.x * abs(glm::dot(axis, other.getAxis(0))) +
				    other.halfSize.y * abs(glm::dot(axis, other.getAxis(1))) +
				    other.halfSize.z * abs(glm::dot(axis, other.getAxis(2)));
	
    float distance = glm::abs(glm::dot(toCentre, axis));

    // return the overlap (positive indicates overlap, negative indicates separation)
    float penetration = oneProject + twoProject - distance;

    if (penetration < 0.0f)
	{
		return false;
	}
	
    if (penetration < smallestPenetration && axis != glm::vec3(0.0f, 0.0f, 0.0f))
	{
        smallestPenetration = penetration;
        smallestCase = index;
    }
	
    return true;
}

void Box::calcTransformMatrix()
{
	scaleMatrix = glm::diagonal4x4(glm::vec4(size, 1.0f));
	rotationMatrix = glm::toMat4(this->rotation);
	positionMatrix = column(glm::diagonal4x4(glm::vec4(1.0f)), 3, glm::vec4(this->position, 1.0f));
	transformMatrix = positionMatrix * (rotationMatrix);
}