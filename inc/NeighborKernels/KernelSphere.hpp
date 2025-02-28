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
	 * @brief For the boxOverlap functions, we find the furthest corner of the passed box
	 * from the kernel center and check if it is inside. We don't have to test the other corners, since
	 * they will always be inside because of the sphere definition.
	*/
	[[nodiscard]] IntersectionJudgement boxIntersect(const Point& center, const double radius) const override
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
	
		// Check if the furthest point from the center of the box is inside sphere -> the box is inside the
		// sphere	
		Point furthest = Point(
			( this->center().getX() > center.getX() ? highX : lowX ),
			( this->center().getY() > center.getY() ? highY : lowY ),
			( this->center().getZ() > center.getZ() ? highZ : lowZ ));
		if(isInside(furthest)) {
			return KernelAbstract::IntersectionJudgement::INSIDE;
		}

		// Otherwise, the box may overlap the sphere 
		// (this can give false positives but that is ok for octree traversal purposes) 
		return KernelAbstract::IntersectionJudgement::OVERLAP;
	}

	[[nodiscard]] IntersectionJudgement boxIntersect(const Point& center, const Vector &radii) const override
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
		
		// Check if the furthest point from the center of the box is inside sphere -> the box is inside the
		// sphere
		Point furthest = Point(
			( this->center().getX() < center.getX() ? highX : lowX ),
			( this->center().getY() < center.getY() ? highY : lowY ),
			( this->center().getZ() < center.getZ() ? highZ : lowZ ));
		if(isInside(furthest)) {
			return KernelAbstract::IntersectionJudgement::INSIDE;
		}

		// Otherwise, the box may overlap the sphere 
		// (this can give false positives but that is ok for octree traversal purposes) 
		return KernelAbstract::IntersectionJudgement::OVERLAP;
	}
};

#endif /* end of include guard: KERNELSPHERE_HPP */
