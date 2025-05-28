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

#endif /* end of include guard: KERNELCIRCLE_HPP */

