#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/quaternion.hpp>
#include <string>
#include <iostream>

static const glm::vec3 gravity = glm::vec3(0.0f, -9.81f, 0.0f );

struct Moveable
{
	std::string objectName = "";
	
	glm::vec3 position;
	glm::quat rotation;

	glm::vec3 linearVelocity = glm::vec3(0.0f);
	glm::vec3 angularVelocity = glm::vec3(0.0f);

	float mass;
	float invMass;

	glm::vec3 linearVelocityUpdate;
	glm::vec3 angularVelocityUpdate;

	bool isGravityEnabled;
	bool isStatic;
	bool isAsleep = false;

	float bounciness = 0.0f;

	glm::mat4 transformMatrix;
	glm::mat4 scaleMatrix = glm::mat4(0.0f);
	glm::mat4 rotationMatrix = glm::mat4(0.0f);
	glm::mat4 positionMatrix = glm::mat4(0.0f);

	glm::mat3x3 inertiaTensor;
	glm::mat3x3 inverseInertiaTensor;

	Moveable(std::string objName,
		const glm::vec3& position = glm::vec3(0, 0, 0),
		bool isGravityEnabled = true,
		bool isStatic = false,
		float mass = 1.0f,
		const glm::quat& rotation = glm::quat(1, 0, 0, 0),
		const glm::vec3& linearVelocity = glm::vec3(0, 0, 0),
		const glm::vec3& angularVelocity = glm::vec3(0, 0, 0));
	virtual ~Moveable();

	virtual void update(float dt);
	virtual void calcTransformMatrix();
};