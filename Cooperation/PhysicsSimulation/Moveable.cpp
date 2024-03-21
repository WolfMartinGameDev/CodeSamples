#include "Moveable.h"

Moveable::Moveable(
	std::string objName,
	const glm::vec3& position,
	bool isGravityEnabled,
	bool isStatic,
	float mass,
	const glm::quat& rotation,
	const glm::vec3& linearVelocity,
	const glm::vec3& angularVelocity
	) :
	objectName(objName),
	position(position),
	rotation(rotation),
	linearVelocity(glm::vec3(0.0f)),
	angularVelocity(glm::vec3(0.0f)),
	mass(mass),
	linearVelocityUpdate(0),
	angularVelocityUpdate(0),
	isGravityEnabled(isGravityEnabled),
	isStatic(isStatic)
{
	calcTransformMatrix();
}

void Moveable::update(float dt)
{
	calcTransformMatrix();

	if (isStatic)
	{
		return;
	}

	if (isAsleep)
	{
		return;
	}
	
	if (isGravityEnabled)
	{
		linearVelocity += gravity * dt;
	}

	position += linearVelocity * dt;
	
	glm::vec3 angularVelocityDt = angularVelocity * dt;
	float angularVelocityDtLength = glm::length(angularVelocityDt);
	
	if (angularVelocityDtLength != 0)
	{
		angularVelocityDt = glm::normalize(angularVelocityDt);
		rotation = glm::angleAxis(angularVelocityDtLength, angularVelocityDt) * rotation;
		rotation = glm::normalize(rotation);
	}

	linearVelocityUpdate = glm::vec3(0);
	angularVelocityUpdate = glm::vec3(0);

	angularVelocity *= 0.8f;
	linearVelocity *= 0.985f;
}

Moveable::~Moveable()
{
}

void Moveable::calcTransformMatrix()
{
}