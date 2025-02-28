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
	 * @brief For the boxOverlap functions, we find the furthest corner of the passed box
	 * from the kernel center and check if it is inside. We don't have to test the other corners, since
	 * they will always be inside because of the circle definition.
	*/
	[[nodiscard]] IntersectionJudgement boxIntersect(const Point& center, const double radius) const override
	{
		// Box bounds
		const double highX = center.getX() + radius, lowX = center.getX() - radius;
		const double highY = center.getY() + radius, lowY = center.getY() - radius;

		// Kernel bounds
		const double boxMaxX = boxMax().getX(), boxMinX = boxMin().getX(); 
		const double boxMaxY = boxMax().getY(), boxMinY = boxMin().getY();

		// Check if box is definitely outside the kernel (like in boxOverlap)
		if (highX < boxMinX || highY < boxMinY ||
			lowX > boxMaxX 	|| lowY > boxMaxY) { 
			return KernelAbstract::IntersectionJudgement::OUTSIDE; 
		}
	
		// Check if the furthest point from the center of the box is inside sphere -> the box is inside the
		// sphere	
		Point furthest = Point(
			( this->center().getX() > center.getX() ? highX : lowX ),
			( this->center().getY() > center.getY() ? highY : lowY ),
			0.0f);
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

		// Kernel bounds
		const double boxMaxX = boxMax().getX(), boxMinX = boxMin().getX(); 
		const double boxMaxY = boxMax().getY(), boxMinY = boxMin().getY();

		// Check if box is definitely outside the kernel (like in boxOverlap)
		if (highX < boxMinX || highY < boxMinY ||
			lowX > boxMaxX 	|| lowY > boxMaxY) { 
			return KernelAbstract::IntersectionJudgement::OUTSIDE; 
		}
		
		// Check if the furthest point from the center of the box is inside sphere -> the box is inside the
		// sphere
		Point furthest = Point(
			( this->center().getX() < center.getX() ? highX : lowX ),
			( this->center().getY() < center.getY() ? highY : lowY ),
			0.0f);
		if(isInside(furthest)) {
			return KernelAbstract::IntersectionJudgement::INSIDE;
		}

		// Otherwise, the box may overlap the sphere 
		// (this can give false positives but that is ok for octree traversal purposes) 
		return KernelAbstract::IntersectionJudgement::OVERLAP;
	}
};

#endif /* end of include guard: KERNELCIRCLE_HPP */

