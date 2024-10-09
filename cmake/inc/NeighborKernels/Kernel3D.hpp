//
// Created by ruben.laso on 13/10/22.
//

#ifndef KERNEL3D_HPP
#define KERNEL3D_HPP

#include "KernelAbstract.hpp"

class Kernel3D : public KernelAbstract
{
	public:
	Kernel3D(const Point& center, const double radius) : KernelAbstract(center, radius) {}
	Kernel3D(const Point& center, const Vector& radii) : KernelAbstract(center, radii) {}

	[[nodiscard]] bool boxOverlap(const Point& center, const double radius) const override
	{
		if (center.getX() + radius < boxMin().getX() || center.getY() + radius < boxMin().getY() ||
		    center.getZ() + radius < boxMin().getZ())
		{
			return false;
		}

		if (center.getX() - radius > boxMax().getX() || center.getY() - radius > boxMax().getY() ||
		    center.getZ() - radius > boxMax().getZ())
		{
			return false;
		}

		return true;
	}
};

#endif /* end of include guard: KERNEL3D_HPP */
