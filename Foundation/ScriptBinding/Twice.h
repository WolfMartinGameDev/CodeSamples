#pragma once
#include "Element.h"

class Twice : public Element
{
public:
	Twice(Element* parent);
	std::string getValue();

private:
	Element* m_parent;
};