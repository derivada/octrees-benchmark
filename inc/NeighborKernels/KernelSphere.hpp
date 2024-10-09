//
// Created by ruben.laso on 13/10/22.
//

#ifndef KERNELSPHERE_HPP
#define KERNELSPHERE_HPP

#include "Kernel3D.hpp"

class KernelSphere : public Kernel3D
{
	double radius_;

	public:
	KernelSphere(const Point& center, const double radius) : Kernel3D(center, radius), radius_(radius) {}

	[[nodiscard]] inline auto radius() const { return radius_; }

	[[nodiscard]] bool isInside(const Point& p) const override
	/**
 * @brief Checks if a given point lies inside the kernel
 * @param p
 * @return
 */
	{
		return square(p.getX() - center().getX()) + square(p.getY() - center().getY()) +
		           square(p.getZ() - center().getZ()) <
		       square(radius());
	}
};

#endif /* end of include guard: KERNELSPHERE_HPP */
