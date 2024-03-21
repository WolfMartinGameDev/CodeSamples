#ifndef SCENE_H
#define SCENE_H

#pragma once

#include "Box.h"
#include <vector>
#include <list>
#include <glm/gtx/projection.hpp>
#include "Manifold.h"
#include <iostream>
#include <map>
#include <queue>
#include <string>

using namespace glm;

class Scene
{
public:
	Box* debugBox;
	
	Scene(const Scene&) = delete;
	Scene& operator=(const Scene&) = delete;

	Scene(float timeStep, bool frozen = false);
	Box& addBox();
	Box& addBox(Box&& box);

	virtual void update(float dt);
	void resolveCollisions(float dt);
	void freeze(bool frozen);
	bool frozen() const;
	float timestep() const;
	void timestep(float dt);
	void spawnBoxes(float dt);
	float getRandom();

	template<typename V>
	void visitBoxes(V v) const
	{
		for (auto& b : boxes)
		{
			v(b);
		}
	}

	template<typename V>
	void visitMovables(V v) const
	{
		for (auto& b : boxes)
		{
			v(b);
		}
	}

	virtual ~Scene() {}

protected:
	bool isFrozen;
	float timeStep;

	std::list<Box> boxes;

	virtual void performStep(float dt);
	virtual void move(float dt);
	virtual void collission(float dt);
	std::map<float, Manifold*> collisions;

	const float spawnTime = 2.0f;
	float currentTime = 0.0f;

	bool BoxToBox(Box& a, Box& b, Manifold* manifold)
	{
		vec3 toCentre = b.position - a.position;
		float penetration = std::numeric_limits<float>::max();
	    unsigned best = 0xffffff;

		if (!a.tryAxis(b, a.getAxis(0), toCentre, 0, penetration, best))
		{
			return false;
		}
		
		if (!a.tryAxis(b, a.getAxis(1), toCentre, 1, penetration, best))
		{
			return false;
		}
		
		if (!a.tryAxis(b, a.getAxis(2), toCentre, 2, penetration, best))
		{
			return false;
		}

		if (!b.tryAxis(a, b.getAxis(0), toCentre, 3, penetration, best))
		{
			return false;
		}
		
		if (!b.tryAxis(a, b.getAxis(1), toCentre, 4, penetration, best))
		{
			return false;
		}
		
		if (!b.tryAxis(a, b.getAxis(2), toCentre, 5, penetration, best))
		{
			return false;
		}

		unsigned bestSingleAxis = best;

		if (!a.tryAxis(b, cross(a.getAxis(0), b.getAxis(0)), toCentre, 6, penetration, best))
		{
			return false;
		}
		
		if (!a.tryAxis(b, cross(a.getAxis(0), b.getAxis(1)), toCentre, 7, penetration, best))
		{
			return false;
		}
		
		if (!a.tryAxis(b, cross(a.getAxis(0), b.getAxis(2)), toCentre, 8, penetration, best))
		{
			return false;
		}
		if (!a.tryAxis(b, cross(a.getAxis(1), b.getAxis(0)), toCentre, 9, penetration, best))
		{
			return false;
		}
		
		if (!a.tryAxis(b, cross(a.getAxis(1), b.getAxis(1)), toCentre, 10, penetration, best))
		{
			return false;
		}
		
		if (!a.tryAxis(b, cross(a.getAxis(1), b.getAxis(2)), toCentre, 11, penetration, best))
		{
			return false;
		}
		
		if (!a.tryAxis(b, cross(a.getAxis(2), b.getAxis(0)), toCentre, 12, penetration, best))
		{
			return false;
		}
		
		if (!a.tryAxis(b, cross(a.getAxis(2), b.getAxis(1)), toCentre, 13, penetration, best))
		{
			return false;
		}
		
		if (!a.tryAxis(b, cross(a.getAxis(2), b.getAxis(2)), toCentre, 14, penetration, best))
		{
			return false;
		}

		if (best < 3)
	    {
			// a vertex of box two is on a face of box one
			manifold->contactType = ContactType::VertexFace;
			fillCollisionData(a, b, manifold, penetration, best, toCentre);
	    }
		else if (best < 6)
		{
			// a vertex of box one is on a face of box two
			manifold->contactType = ContactType::VertexFace;
			fillCollisionData(b, a, manifold, penetration, best - 3, toCentre * -1.0f);
		}
		else
		{
			// edge to edge contact -> find out which axes
			best -= 6;
			unsigned oneAxisIndex = best / 3;
			unsigned twoAxisIndex = best % 3;
			vec3 oneAxis = a.getAxis(oneAxisIndex);
			vec3 twoAxis = b.getAxis(twoAxisIndex);
			vec3 axis;
			
			if (oneAxis == twoAxis)
			{
				axis = oneAxis;
			}
			else
			{
				axis = cross(oneAxis, twoAxis);
				axis = normalize(axis);
			}
			
			// the axis should point from box one to box two
			if (dot(axis, toCentre) < 0.0f)
			{
				axis = axis * -1.0f;
			}

			// each axis has 4 edges parallel to it -> find out which edge for each object
			vec3 ptOnOneEdge = a.halfSize;
			vec3 ptOnTwoEdge = b.halfSize;
			
			for (unsigned i = 0; i < 3; i++)
			{
				if (i == oneAxisIndex)
				{
					ptOnOneEdge[i] = 0;
				}
				else if (dot(a.getAxis(i), axis) > 0.0f)
				{
					ptOnOneEdge[i] = -ptOnOneEdge[i];
				}

				if (i == twoAxisIndex)
				{
					ptOnTwoEdge[i] = 0;
				}
				else if (dot(b.getAxis(i), axis) < 0.0f)
				{
					ptOnTwoEdge[i] = -ptOnTwoEdge[i];
				}
			}

			// move the edges into world coordinates
			ptOnOneEdge = a.transformMatrix * vec4(ptOnOneEdge, 0);
			ptOnTwoEdge = b.transformMatrix * vec4(ptOnTwoEdge, 0);

			// find out the point of closest approach of the two axes.
			vec3 vertex = contactPoint(ptOnOneEdge, oneAxis, a.halfSize[oneAxisIndex], ptOnTwoEdge, twoAxis, b.halfSize[twoAxisIndex], bestSingleAxis > 2);

			manifold->contactType = ContactType::EdgeEdge;
			manifold->penetrationDepth = penetration;
			manifold->normal = axis;
			manifold->object = &a;
			manifold->otherObject = &b;
			manifold->collisionPosition = -vertex;
		}

		if (penetration < 0.001f)
		{
			return false;
		}

		collisions.insert(std::make_pair(penetration, manifold));
		return true;
	}

	static void fillCollisionData(Box& a, Box& b, Manifold* manifold, float penetration, unsigned best, vec3 toCentre)
	{
		vec3 normal = -a.getAxis(best);

		// find out which vertex of box two is colliding
		vec3 vertex = b.halfSize;
		if (dot(b.getAxis(0), normal) < 0.0f)
		{
			vertex.x = -vertex.x;
		}
		
		if (dot(b.getAxis(1), normal) < 0.0f)
		{
			vertex.y = -vertex.y;
		}
		
		if (dot(b.getAxis(2), normal) < 0.0f)
		{
			vertex.z = -vertex.z;
		}

		if (dot(a.getAxis(best), toCentre) > 0.0f)
		{
			normal = normal * -1.0f;
		}
		
		manifold->penetrationDepth = penetration;
		manifold->object = &a;
		manifold->otherObject = &b;
		manifold->normal = normal;
		manifold->collisionPosition = b.transformMatrix * vec4(vertex, 0);
	}

	static inline vec3 contactPoint(
		const vec3& pOne,
		const vec3& dOne,
		float oneSize,
		const vec3& pTwo,
		const vec3& dTwo,
		float twoSize,
		bool useOne)
	{
		vec3 toSt, cOne, cTwo;
		float dpStaOne, dpStaTwo, dpOneTwo, smOne, smTwo;
		float denom, mua, mub;

		smOne = length2(dOne);
		smTwo = length2(dTwo);
		dpOneTwo = dot(dTwo, dOne);

		toSt = pOne - pTwo;
		dpStaOne = dot(dOne, toSt);
		dpStaTwo = dot(dTwo, toSt);

		denom = smOne * smTwo - dpOneTwo * dpOneTwo;

		// zero denominator indicates parrallel lines
		if (abs(denom) < 0.0001f)
		{
			return useOne ? pOne : pTwo;
		}

		mua = (dpOneTwo * dpStaTwo - smTwo * dpStaOne) / denom;
		mub = (smOne * dpStaTwo - dpOneTwo * dpStaOne) / denom;

		// if either of the edges has the nearest point out of bounds we have an edge to face contact
		if (mua > oneSize || mua < -oneSize || mub > twoSize || mub < -twoSize)
		{
			return useOne ? pOne : pTwo;
		}
		else
		{
			cOne = pOne + dOne * mua;
			cTwo = pTwo + dTwo * mub;

			return cOne * 0.5f + cTwo * 0.5f;
		}
	}

	static vec3 Normalize(vec3 vec)
	{
		if (vec == vec3(0.0f))
		{
			return vec;
		}
		
		return normalize(vec);
	}

	static void printVector(vec3 vec, std::string name)
	{
		std::cout << name << ": " << (vec.x) << "," << (vec.y) << "," << (vec.z) << std::endl;
	}
};
#endif