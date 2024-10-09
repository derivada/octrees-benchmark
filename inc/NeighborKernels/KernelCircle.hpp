//
// Created by ruben.laso on 13/10/22.
//

#ifndef KERNELCIRCLE_HPP
#define KERNELCIRCLE_HPP

#include "Kernel2D.hpp"

#include "util.hpp"

class KernelCircle : public Kernel2D
{
	double radius_;

	public:
	KernelCircle(const Point& center, const double radius) : Kernel2D(center, radius), radius_(radius) {}

	[[nodiscard]] inline auto radius() const { return radius_; }

	[[nodiscard]] inline bool isInside(const Point& p) const override
	/**
 * @brief Checks if a given point lies inside the kernel
 * @param p
 * @return
 */
	{
		return square(p.getX() - center().getX()) + square(p.getY() - center().getY()) < square(radius());
	}
};

#endif /* end of include guard: KERNELCIRCLE_HPP */
