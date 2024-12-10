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

	/**
	 * @brief For the boxInside functions, we find the furthest corner of the passed box
	 * from the kernel center and check if it is inside. We don't have to test the other corners, since
	 * they will always be inside because of the sphere definition.
	*/
	[[nodiscard]] bool boxInside(const Point& c, const double radius) const override
	{
		Point furthest = Point(
			center().getX() + ( c.getX() > center().getX() ? radius : -radius ),
			center().getY() + ( c.getY() > center().getY() ? radius : -radius ),
			center().getZ() + ( c.getZ() > center().getZ() ? radius : -radius ));

		return isInside(furthest);
	}

	[[nodiscard]] bool boxInside(const Point& c, const Vector& radii) const override
	{
		Point furthest = Point(
			center().getX() + ( c.getX() > center().getX() ? radii.getX() : -radii.getX() ),
			center().getY() + ( c.getY() > center().getY() ? radii.getY() : -radii.getY() ),
			center().getZ() + ( c.getZ() > center().getZ() ? radii.getZ() : -radii.getZ() ));

		return isInside(furthest);
	}
};

#endif /* end of include guard: KERNELSPHERE_HPP */
