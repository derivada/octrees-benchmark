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

	/**
	 * @brief For the boxInside functions, we find the furthest corner of the passed box
	 * from the kernel center and check if it is inside. We don't have to test the other corners, since
	 * they will always be inside because of the circle definition.
	*/
	[[nodiscard]] bool boxInside(const Point& c, const double radius) const override
	{
		Point furthest = Point(
			center().getX() + ( c.getX() > center().getX() ? radius : -radius ),
			center().getY() + ( c.getY() > center().getY() ? radius : -radius ),
			0.0f);

		return isInside(furthest);
	}

	[[nodiscard]] bool boxInside(const Point& c, const Vector& radii) const override
	{
		Point furthest = Point(
			center().getX() + ( c.getX() > center().getX() ? radii.getX() : -radii.getX() ),
			center().getY() + ( c.getY() > center().getY() ? radii.getY() : -radii.getY() ),
			0.0f);

		return isInside(furthest);
	}
};

#endif /* end of include guard: KERNELCIRCLE_HPP */
