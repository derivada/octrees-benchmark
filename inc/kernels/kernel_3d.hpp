//
// Created by ruben.laso on 13/10/22.
//

#pragma once

#include <array>
#include "kernel_abstract.hpp"

class Kernel3D : public KernelAbstract
{
	public:
	Kernel3D(const Point& center, const double radius) : KernelAbstract(center, radius) {}
	Kernel3D(const Point& center, const Vector& radii) : KernelAbstract(center, radii) {}

	[[nodiscard]] bool boxOverlap(const Point& center, const double radius) const override
	{
		if (center.getX() + radius < boxMin().getX() || center.getY() + radius < boxMin().getY() ||
		    center.getZ() + radius < boxMin().getZ())
		{
			return false;
		}

		if (center.getX() - radius > boxMax().getX() || center.getY() - radius > boxMax().getY() ||
		    center.getZ() - radius > boxMax().getZ())
		{
			return false;
		}

		return true;
	}

	[[nodiscard]] bool boxOverlap(const Point& center, const Vector& radii) const override
	{
		if (center.getX() + radii.getX() < boxMin().getX() || center.getY() + radii.getY() < boxMin().getY() ||
		    center.getZ() + radii.getZ() < boxMin().getZ())
		{
			return false;
		}

		if (center.getX() - radii.getX() > boxMax().getX() || center.getY() - radii.getY() > boxMax().getY() ||
		    center.getZ() - radii.getZ() > boxMax().getZ())
		{
			return false;
		}

		return true;
	}
};

class KernelSphere : public Kernel3D
{
	double radius_;
	double radiusSq_;
	public:
	KernelSphere(const Point& center, const double radius) : Kernel3D(center, radius), radius_(radius), radiusSq_(radius*radius) {}

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
		double d3 = p.getZ() - center().getZ();
		return d1*d1 + d2*d2 + d3*d3 < radiusSq_;
	}
	/// @brief Geometric check from unibnOctree
	[[nodiscard]] IntersectionJudgement boxIntersect(const Point& octantCenter, const double octantRadius) const override
	{
		const Point& kernelCenter = this->center();
	
		// Symmetry trick: operate in first octant by taking abs difference.
		double x = std::abs(kernelCenter.getX() - octantCenter.getX());
		double y = std::abs(kernelCenter.getY() - octantCenter.getY());
		double z = std::abs(kernelCenter.getZ() - octantCenter.getZ());
	
		double maxDist = radius_ + octantRadius;
	
		// === OUTSIDE ===
		if (x > maxDist || y > maxDist || z > maxDist)
			return IntersectionJudgement::OUTSIDE;
	
		// === CONTAINS ===
		// Translate box corner to farthest corner from center, like in Octree::contains
		double cx = x + octantRadius;
		double cy = y + octantRadius;
		double cz = z + octantRadius;
	
		if ((cx * cx + cy * cy + cz * cz) < radiusSq_)
			return IntersectionJudgement::INSIDE;
	
		// === OVERLAPS ===
		// Mirror of Octree::overlaps
		int32_t numLessExtent = (x < octantRadius) + (y < octantRadius) + (z < octantRadius);
		if (numLessExtent > 1)
			return IntersectionJudgement::OVERLAP;
	
		x = std::max(x - octantRadius, 0.0);
		y = std::max(y - octantRadius, 0.0);
		z = std::max(z - octantRadius, 0.0);
	
		if ((x * x + y * y + z * z) < radiusSq_)
			return IntersectionJudgement::OVERLAP;
	
		return IntersectionJudgement::OUTSIDE;
	}

	/// @brief Geometric check from unibnOctree
	[[nodiscard]] IntersectionJudgement boxIntersect(const Point& octantCenter, const Vector& octantRadii) const override
	{
		const Point& kernelCenter = this->center();

		// Symmetric test: reflect into positive octant
		double dx = std::abs(kernelCenter.getX() - octantCenter.getX());
		double dy = std::abs(kernelCenter.getY() - octantCenter.getY());
		double dz = std::abs(kernelCenter.getZ() - octantCenter.getZ());

		const double rx = octantRadii.getX();
		const double ry = octantRadii.getY();
		const double rz = octantRadii.getZ();

		const double maxDx = radius_ + rx;
		const double maxDy = radius_ + ry;
		const double maxDz = radius_ + rz;

		// === OUTSIDE === (no overlap at all)
		if (dx > maxDx || dy > maxDy || dz > maxDz)
			return IntersectionJudgement::OUTSIDE;

		// === CONTAINS ===
		// Farthest corner from center (Minkowski sum check)
		double cx = dx + rx;
		double cy = dy + ry;
		double cz = dz + rz;
		if ((cx * cx + cy * cy + cz * cz) < radiusSq_)
			return IntersectionJudgement::INSIDE;

		// === OVERLAPS ===
		int32_t numInside = (dx < rx) + (dy < ry) + (dz < rz);
		if (numInside > 1)
			return IntersectionJudgement::OVERLAP;

		dx = std::max(dx - rx, 0.0);
		dy = std::max(dy - ry, 0.0);
		dz = std::max(dz - rz, 0.0);

		if ((dx * dx + dy * dy + dz * dz) < radiusSq_)
			return IntersectionJudgement::OVERLAP;

		return IntersectionJudgement::OUTSIDE;
	}
};

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
	 * @brief For the boxIntersect functions, we check if the box passed is inside,
	 * overlaps or is outside the bounding box of the square cube
	 * 
	 * @returns IntersectionJudgement, a enum value signaling each of the three conditions
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
		
		// Check if everything is inside
		if(highX <= boxMaxX && highY <= boxMaxY && highZ <= boxMaxZ &&
		   lowX >= boxMinX 	&& lowY >= boxMinY 	&& lowZ >= boxMinZ) {
			return KernelAbstract::IntersectionJudgement::INSIDE;
		}

		return KernelAbstract::IntersectionJudgement::OVERLAP;
	}

	[[nodiscard]] IntersectionJudgement boxIntersect(const Point& center, const Vector& radii) const override
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
		
		// Check if everything is inside
		if(highX <= boxMaxX && highY <= boxMaxY && highZ <= boxMaxZ &&
		   lowX >= boxMinX 	&& lowY >= boxMinY 	&& lowZ >= boxMinZ) {
			return KernelAbstract::IntersectionJudgement::INSIDE;
		}

		return KernelAbstract::IntersectionJudgement::OVERLAP;
	}
};