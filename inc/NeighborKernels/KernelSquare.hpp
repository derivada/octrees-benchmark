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

	/**
	 * @brief For the boxInside functions, we check if the volume passed is inside the bounding box of the square
	*/
	[[nodiscard]] bool boxInside(const Point& center, const double radius) const override
	{
		// Passed box xy boundaries
		auto maxX = center.getX() + radius, minX = center.getX() - radius;
		auto maxY = center.getY() + radius, minY = center.getY() - radius;
		// Check everything is inside
		return maxX <= boxMax().getX() && minX >= boxMin().getX() &&
			   maxY <= boxMax().getY() && minY >= boxMin().getY(); 
	}

	[[nodiscard]] bool boxInside(const Point& center, const Vector& radii) const override
	{
		// Passed box xy boundaries
		auto maxX = center.getX() + radii.getX(), minX = center.getX() - radii.getX();
		auto maxY = center.getY() + radii.getY(), minY = center.getY() - radii.getY();
		// Check everything is inside
		return maxX < boxMax().getX() && minX > boxMin().getX() &&
			   maxY < boxMax().getY() && minY > boxMin().getY(); 
	}
};

#endif /* end of include guard: KERNELSQUARE_HPP */
