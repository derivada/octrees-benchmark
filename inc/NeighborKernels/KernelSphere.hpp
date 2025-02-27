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

	[[nodiscard]] inline bool isInside(const Point& p) const override
	/**
	 * @brief Checks if a given point lies inside the kernel
	 * @param p
	 * @return
	 */
	{
		const double dx = p.getX() - center().getX();
		const double dy = p.getY() - center().getY();
		const double dz = p.getY() - center().getY();
		const double r = radius();
		return (dx * dx + dy * dy + dz*dz) < (r * r);
	}

	/**
	 * @brief For the boxOverlap functions, we find the furthest corner of the passed box
	 * from the kernel center and check if it is inside. We don't have to test the other corners, since
	 * they will always be inside because of the sphere definition.
	*/
	[[nodiscard]] inline IntersectionJudgement boxIntersect(const Point& center, const double radius) const override
	{
		// Box bounds
		const double highX = center.getX() + radius, lowX = center.getX() - radius;
		const double highY = center.getY() + radius, lowY = center.getY() - radius;
		const double highZ = center.getZ() + radius, lowZ = center.getZ() - radius;

		// Kernel bounds
		const double boxMaxX = boxMax().getX(), boxMinX = boxMin().getX(); 
		const double boxMaxY = boxMax().getY(), boxMinY = boxMin().getY();
		const double boxMaxZ = boxMax().getZ(), boxMinZ = boxMin().getZ();

		// Check if box is definitely outside the kernel (like in boxOverlap)
		if (highX < boxMinX || highY < boxMinY 	|| highZ < boxMinZ || 
			lowX > boxMaxX 	|| lowY > boxMaxY 	|| lowZ > boxMaxZ) { 
			return KernelAbstract::IntersectionJudgement::OUTSIDE; 
		}

		// Get the coordinates of the furthest point in the bbox of the sphere
		const double furthestX = (this->center().getX() > center.getX()) ? highX : lowX;
		const double furthestY = (this->center().getY() > center.getY()) ? highY : lowY;
		const double furthestZ = (this->center().getZ() > center.getZ()) ? highZ : lowZ;

		return isInside(Point(furthestX, furthestY, furthestZ)) ? 
			KernelAbstract::IntersectionJudgement::INSIDE : 
			KernelAbstract::IntersectionJudgement::OVERLAP;
	}
	
	[[nodiscard]] inline IntersectionJudgement boxIntersect(const Point& center, const Vector &radii) const override
	{
		// Box bounds
		const double highX = center.getX() + radii.getX(), lowX = center.getX() - radii.getX();
		const double highY = center.getY() + radii.getY(), lowY = center.getY() - radii.getY();
		const double highZ = center.getZ() + radii.getZ(), lowZ = center.getZ() - radii.getZ();

		// Kernel bounds
		const double boxMaxX = boxMax().getX(), boxMinX = boxMin().getX(); 
		const double boxMaxY = boxMax().getY(), boxMinY = boxMin().getY();
		const double boxMaxZ = boxMax().getZ(), boxMinZ = boxMin().getZ();

		// Check if box is definitely outside the kernel (like in boxOverlap)
		if (highX < boxMinX || highY < boxMinY 	|| highZ < boxMinZ || 
			lowX > boxMaxX 	|| lowY > boxMaxY 	|| lowZ > boxMaxZ) { 
			return KernelAbstract::IntersectionJudgement::OUTSIDE; 
		}
		
		// Get the coordinates of the furthest point in the bbox of the sphere
		const double furthestX = (this->center().getX() > center.getX()) ? highX : lowX;
		const double furthestY = (this->center().getY() > center.getY()) ? highY : lowY;
		const double furthestZ = (this->center().getZ() > center.getZ()) ? highZ : lowZ;

		return isInside(Point(furthestX, furthestY, furthestZ)) ? 
			KernelAbstract::IntersectionJudgement::INSIDE : 
			KernelAbstract::IntersectionJudgement::OVERLAP;
	}
};

#endif /* end of include guard: KERNELSPHERE_HPP */
