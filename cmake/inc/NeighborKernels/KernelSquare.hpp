//
// Created by ruben.laso on 13/10/22.
//

#ifndef KERNELSQUARE_HPP
#define KERNELSQUARE_HPP

#include "Kernel2D.hpp"
#include "util.hpp"

class KernelSquare : public Kernel2D
{
	public:
	KernelSquare(const Point& center, const double radius) : Kernel2D(center, radius) {}
	KernelSquare(const Point& center, const Vector& radii) : Kernel2D(center, radii) {}

	[[nodiscard]] bool isInside(const Point& p) const override
	/**
 * @brief Checks if a given point lies inside the kernel
 * @param p
 * @return
 */
	{
		return onInterval(p.getX(), boxMin().getX(), boxMax().getX()) &&
		       onInterval(p.getY(), boxMin().getY(), boxMax().getY());
	}
};

#endif /* end of include guard: KERNELSQUARE_HPP */
