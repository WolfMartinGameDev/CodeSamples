#include "pch.h"

Twice::Twice(Element* parent)
{
	m_parent = parent;
}

std::string Twice::getValue()
{
	return m_parent->getValue() + " " + m_parent->getValue();
}