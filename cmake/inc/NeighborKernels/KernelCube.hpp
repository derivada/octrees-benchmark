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
};


#endif /* end of include guard: KERNELCUBE_HPP */
