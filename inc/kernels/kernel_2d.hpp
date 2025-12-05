//
// Created by ruben.laso on 13/10/22.
//

#pragma once

#include <array>
#include "kernel_abstract.hpp"

class Kernel2D : public KernelAbstract
{
	public:
	Kernel2D(const Point& center, const double radius) : KernelAbstract(center, radius) {}
	Kernel2D(const Point& center, const Point& radii) : KernelAbstract(center, radii) {}

	[[nodiscard]] bool boxOverlap(const Point& center, const double radius) const override
	/**
 * @brief Checks if a given octant overlaps with the given kernel in 2 dimensions
 * @param octant
 * @return
 */
	{
		if (center.getX() + radius < boxMin().getX() || center.getY() + radius < boxMin().getY()) { return false; }

		if (center.getX() - radius > boxMax().getX() || center.getY() - radius > boxMax().getY()) { return false; }

		return true;
	}
	[[nodiscard]] bool boxOverlap(const Point& center, const Vector& radii) const override
	/**
 * @brief Checks if a given octant overlaps with the given kernel in 2 dimensions
 * @param octant
 * @return
 */
	{
		if (center.getX() + radii.getX() < boxMin().getX() || center.getY() + radii.getY() < boxMin().getY()) { return false; }

		if (center.getX() - radii.getX() > boxMax().getX() || center.getY() - radii.getY() > boxMax().getY()) { return false; }

		return true;
	}
};

class KernelCircle : public Kernel2D
{
	double radius_;
	double radiusSq_;
	public:
	KernelCircle(const Point& center, const double radius) : Kernel2D(center, radius), radius_(radius), radiusSq_(radius*radius) {}

	[[nodiscard]] inline auto radius() const { return radius_; }

	[[nodiscard]] bool isInside(const Point& p) const override
	/**
	 * @brief Checks if a given point lies inside the kernel
	 * @param p
	 * @return
	*/
	{
		double d1 = p.getX() - center().getX();
		double d2 = p.getY() - center().getY();
		return d1*d1 + d2*d2 < radiusSq_;
	}

	/**
	 * @brief For the boxOverlap functions, we find the furthest corner of the passed box
	 * from the kernel center and check if it is inside. We don't have to test the other corners, since
	 * they will always be inside because of the circle definition.
	*/
	[[nodiscard]] IntersectionJudgement boxIntersect(const Point& octantCenter, const double octantRadius) const override
	{
		const Point& kernelCenter = this->center();
	
		// Symmetry trick: operate in first octant by taking abs difference.
		double x = std::abs(kernelCenter.getX() - octantCenter.getX());
		double y = std::abs(kernelCenter.getY() - octantCenter.getY());
	
		double maxDist = radius_ + octantRadius;
	
		// === OUTSIDE ===
		if (x > maxDist || y > maxDist)
			return IntersectionJudgement::OUTSIDE;
	
		// === CONTAINS ===
		// Translate box corner to farthest corner from center, like in Octree::contains
		double cx = x + octantRadius;
		double cy = y + octantRadius;
	
		if ((cx * cx + cy * cy) < radiusSq_)
			return IntersectionJudgement::INSIDE;
	
		// === OVERLAPS ===
		// Mirror of Octree::overlaps
		int32_t numLessExtent = (x < octantRadius) + (y < octantRadius);
		if (numLessExtent > 1)
			return IntersectionJudgement::OVERLAP;
	
		x = std::max(x - octantRadius, 0.0);
		y = std::max(y - octantRadius, 0.0);
	
		if ((x * x + y * y) < radiusSq_)
			return IntersectionJudgement::OVERLAP;
	
		return IntersectionJudgement::OUTSIDE;
	}

	[[nodiscard]] IntersectionJudgement boxIntersect(const Point& octantCenter, const Vector& octantRadii) const override
	{
		const Point& kernelCenter = this->center();

		// Symmetric test: reflect into positive octant
		double dx = std::abs(kernelCenter.getX() - octantCenter.getX());
		double dy = std::abs(kernelCenter.getY() - octantCenter.getY());

		const double rx = octantRadii.getX();
		const double ry = octantRadii.getY();

		const double maxDx = radius_ + rx;
		const double maxDy = radius_ + ry;

		// === OUTSIDE === (no overlap at all)
		if (dx > maxDx || dy > maxDy)
			return IntersectionJudgement::OUTSIDE;

		// === CONTAINS ===
		// Farthest corner from center (Minkowski sum check)
		double cx = dx + rx;
		double cy = dy + ry;
		if ((cx * cx + cy * cy) < radiusSq_)
			return IntersectionJudgement::INSIDE;

		// === OVERLAPS ===
		int32_t numInside = (dx < rx) + (dy < ry);
		if (numInside > 1)
			return IntersectionJudgement::OVERLAP;

		dx = std::max(dx - rx, 0.0);
		dy = std::max(dy - ry, 0.0);

		if ((dx * dx + dy * dy) < radiusSq_)
			return IntersectionJudgement::OVERLAP;

		return IntersectionJudgement::OUTSIDE;
	}
};

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
	 * @brief For the boxIntersect functions, we check if the box passed is inside,
	 * overlaps or is outside the bounding box of the square kernel
	 * 
	 * @returns IntersectionJudgement, a enum value signaling each of the three conditions
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
		
		// Check if everything is inside
		if(highX <= boxMaxX && highY <= boxMaxY &&
		   lowX >= boxMinX 	&& lowY >= boxMinY) {
			return KernelAbstract::IntersectionJudgement::INSIDE;
		}

		return KernelAbstract::IntersectionJudgement::OVERLAP;
	}

	[[nodiscard]] IntersectionJudgement boxIntersect(const Point& center, const Vector& radii) const override
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
		
		// Check if everything is inside
		if(highX <= boxMaxX && highY <= boxMaxY &&
		   lowX >= boxMinX 	&& lowY >= boxMinY) {
			return KernelAbstract::IntersectionJudgement::INSIDE;
		}

		return KernelAbstract::IntersectionJudgement::OVERLAP;
	}
};
