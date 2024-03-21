#pragma once

template <typename T>
__device__ void vectorToColor(const T& vector, uchar4& outColor)
{
	outColor.x = (unsigned char)(vector.x * 255.999f);
	outColor.y = (unsigned char)(vector.y * 255.999f);
	outColor.z = (unsigned char)(vector.z * 255.999f);
}