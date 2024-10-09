//
// Created by ruben.laso on 13/10/22.
//

#ifndef KERNEL2D_HPP
#define KERNEL2D_HPP

#include "KernelAbstract.hpp"

class Kernel2D : public KernelAbstract
{
	public:
	Kernel2D(const Point& center, const double radius) : KernelAbstract(center, radius) {}
	Kernel2D(const Point& center, const Point& radii) : KernelAbstract(center, radii) {}

	[[nodiscard]] bool boxOverlap(const Point& center, const double radius) const override
	/**
 * @brief Checks if a given octant overlaps with the given kernel in 2 dimensions
 * @param octant
 * @return
 */
	{
		if (center.getX() + radius < boxMin().getX() || center.getY() + radius < boxMin().getY()) { return false; }

		if (center.getX() - radius > boxMax().getX() || center.getY() - radius > boxMax().getY()) { return false; }

		return true;
	}
};

#endif /* end of include guard: KERNEL2D_HPP */
