#include "Scene.h"

Scene::Scene(float timeStep, bool frozen) :
	timeStep(timeStep),
	isFrozen(frozen)
{
	debugBox = &addBox(Box("Debug", glm::vec3(0.1f, 0.1f, 0.1f), glm::vec3(0, 0, 0), false, false, 0.0f, glm::quat(1, 0, 0, 0), glm::vec3(0, 0, 0), glm::vec3(0, 0, 0)));
}

Box & Scene::addBox()
{
	boxes.emplace_back(Box());
	return boxes.back();
}

Box & Scene::addBox(Box && box)
{
	boxes.emplace_back(std::move(box));
	return boxes.back();
}

void Scene::update(float dt)
{
	performStep(timeStep);
}

bool Scene::frozen() const
{
	return isFrozen;
}
void Scene::freeze(bool frozen)
{
	isFrozen = frozen;
}

float Scene::timestep() const
{
	return timeStep;
}

void Scene::timestep(float dt)
{
	timeStep = dt;
}

void Scene::performStep(float dt)
{
	if (isFrozen)
	{
		return;
	}

	move(dt);
	collission(dt);
	resolveCollisions(dt);

	spawnBoxes(dt);
}

void Scene::move(float dt)
{
	for (auto& b : boxes)
	{
		b.update(dt);
	}
}

void Scene::collission(float dt)
{
	std::list<Box>::iterator ita = boxes.begin();
	std::list<Box>::iterator itb;
	
	for (int i = 0; i < boxes.size(); i++)
	{
		itb = ita;
		itb++;

		for (int j = i + 1; j < boxes.size(); j++)
		{			
			Box& a = *ita;
			Box& b = *itb;

			if (a.objectName == "Debug" || b.objectName == "Debug")
			{
				continue;
			}
			
			Manifold* manifold = new Manifold();
			bool intersect = BoxToBox(a, b, manifold);
			
			if (!intersect)
			{
				delete manifold;
			}
			
			itb++;
		}
		ita++;
	}
}

void Scene::resolveCollisions(float dt)
{
	while (!collisions.empty())
	{
		for (std::pair<const float, Manifold*> collision : collisions)
		{
			float c = 0.6f;
			float linSleepSpeed = 1.0f;
			float angSleepSpeed = 1.6f;

			Manifold* manifold = collision.second;
			Moveable* A = manifold->object;
			Moveable* B = manifold->otherObject;
			A->isAsleep = false;
			B->isAsleep = false;
	
			float totalInvMass = A->invMass + B->invMass;

			vec3 n = manifold->normal;
			n = normalize(n);

			A->position -= n * manifold->penetrationDepth * (A->invMass / totalInvMass);
			B->position += n * manifold->penetrationDepth * (B->invMass / totalInvMass);
			
			vec3 qARelative = A->position - (B->position + manifold->collisionPosition);
			vec3 qBRelative = manifold->collisionPosition;

			vec3 qLinA = A->linearVelocity;
			vec3 qLinB = B->linearVelocity;

			vec3 uA = cross(qARelative, n);
			vec3 uB = cross(qBRelative, n);
			vec3 qAngA = cross(A->inverseInertiaTensor * uA, qARelative);
			vec3 qAngB = cross(B->inverseInertiaTensor * uB, qBRelative);
			vec3 qA = qLinA + cross(A->angularVelocity, qARelative);
			vec3 qB = qLinB + cross(B->angularVelocity, qBRelative);
			
			float vs = dot((qA - qB), n);
			
			if (vs <= 0.0f)
			{
				delete collision.second;
				collision.second = nullptr;
				continue;
			}
			
			float vdest = -(1.0f + c) * vs;
			float pAll = A->invMass + dot(qAngA, n) + B->invMass + dot(qAngB, n);
			float gContact = vdest / pAll;
			vec3 gWorld = gContact * n;

			A->linearVelocity += gWorld * A->invMass;
			A->angularVelocity += A->inverseInertiaTensor * cross(qARelative, gWorld);
			
			B->linearVelocity -= gWorld * B->invMass;
			B->angularVelocity -= B->inverseInertiaTensor * cross(qBRelative, gWorld);

			if (length(A->linearVelocity) < linSleepSpeed && length(A->angularVelocity) < angSleepSpeed)
			{
				A->isAsleep = true;
			}
			
			if (length(B->linearVelocity) < linSleepSpeed && length(B->angularVelocity) < angSleepSpeed)
			{
				B->isAsleep = true;
			}
			
			debugBox->position = B->position + manifold->collisionPosition;
			debugBox->color = vec3(1.0f, 0.0f, 0.0f);
			
			delete collision.second;
			collision.second = nullptr;
		}
		
		collisions.clear();
		collission(dt);
	}
}

void Scene::spawnBoxes(float dt)
{
	currentTime += dt;
	
	while (currentTime >= spawnTime)
	{
		addBox(Box("Random Box", glm::vec3(1.0f, 1.0, 1.0), glm::vec3(9.0f * getRandom() - 4.5f, 1.0f + 5.0f * getRandom(), 9.0f * getRandom() - 4.5f), true, false, 1.0f, glm::normalize(glm::quat(1.0f, getRandom(), getRandom(), getRandom())), glm::vec3(getRandom(), getRandom(), getRandom()), glm::vec3(getRandom(), getRandom(), getRandom())));
		currentTime -= spawnTime;
	}
}

float Scene::getRandom()
{
	return static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
}