//
// Created by ruben.laso on 13/10/22.
//

#ifndef KERNELCUBE_HPP
#define KERNELCUBE_HPP

#include "Kernel3D.hpp"
#include "util.hpp"

class KernelCube : public Kernel3D
{
	public:
	KernelCube(const Point& center, const double radius) : Kernel3D(center, radius) {}
	KernelCube(const Point& center, const Vector& radii) : Kernel3D(center, radii) {}

	[[nodiscard]] bool isInside(const Point& p) const override
	/**
 * @brief Checks if a given point lies inside the kernel
 * @param p
 * @return
 */
	{
		return onInterval(p.getX(), boxMin().getX(), boxMax().getX()) &&
		       onInterval(p.getY(), boxMin().getY(), boxMax().getY()) &&
		       onInterval(p.getZ(), boxMin().getZ(), boxMax().getZ());
	};

	/**
	 * @brief For the boxInside functions, we check if the volume passed is inside the bounding box of the square
	*/
	[[nodiscard]] bool boxInside(const Point& center, const double radius) const override
	{
		// Passed box xy boundaries
		auto maxX = center.getX() + radius, minX = center.getX() - radius;
		auto maxY = center.getY() + radius, minY = center.getY() - radius;
		auto maxZ = center.getZ() + radius, minZ = center.getZ() - radius;
		// Check everything is inside
		return maxX < boxMax().getX() && minX > boxMin().getX() &&
			   maxY < boxMax().getY() && minY > boxMin().getY() &&
			   maxZ < boxMax().getZ() && minZ > boxMin().getZ();
	}

	[[nodiscard]] bool boxInside(const Point& center, const Vector& radii) const override
	{
		// Passed box xy boundaries
		auto maxX = center.getX() + radii.getX(), minX = center.getX() - radii.getX();
		auto maxY = center.getY() + radii.getY(), minY = center.getY() - radii.getY();
		auto maxZ = center.getZ() + radii.getZ(), minZ = center.getZ() - radii.getZ();
		// Check everything is inside
		return maxX < boxMax().getX() && minX > boxMin().getX() &&
			   maxY < boxMax().getY() && minY > boxMin().getY() &&
			   maxZ < boxMax().getZ() && minZ > boxMin().getZ();
	}
};


#endif /* end of include guard: KERNELCUBE_HPP */
