//
// Created by miguelyermo on 28/1/22.
//

#pragma once

#include <array>
#include <cstdint>
#include <immintrin.h> 
#include "geometry/box.hpp"
#include "geometry/point.hpp"

class KernelAbstract
/**
 * @brief Kernel to be used in the neighborhood searches. This Kernel must be implemented in concrete classes.
 */
{
	private:
	Vector radii_{};  // Search Radii
	Point  center_{}; // Center of the kernel
	Point  boxMin_{}; // Min Point_t of the Bounding Cube around the given kernel
	Point  boxMax_{}; // Max Point_t of the Bounding Cube around the given kernel

	public:

	enum class IntersectionJudgement {
		OUTSIDE = 0,
		OVERLAP = 1,
		INSIDE = 2
	};

	KernelAbstract(const Point& center, const double radius) : center_(center), radii_({ radius, radius, radius })
	{
		makeBox();
	};

	KernelAbstract(const Point& center, const Vector& radii) : center_(center), radii_(radii) { makeBox(); };

	virtual ~KernelAbstract() = default;

	void makeBox()
	{
		boxMin_ = center_ - radii_;
		boxMax_ = center_ + radii_;
	};

	[[nodiscard]] inline const Vector& radii() const { return radii_; }

	[[nodiscard]] inline const Point& center() const { return center_; }

	[[nodiscard]] inline Point center() { return center_; }

	[[nodiscard]] inline const Point& boxMin() const { return boxMin_; }

	[[nodiscard]] inline const Point& boxMax() const { return boxMax_; }

	// These functions must be implemented in each concrete Kernel
	[[nodiscard]] virtual bool isInside(const Point& p) const                       = 0;
	[[nodiscard]] virtual bool boxOverlap(const Point& center, double radius) const = 0;
	[[nodiscard]] virtual bool boxOverlap(const Point& center, const Vector& radii) const = 0;
	
	// Checks if box is entirely within kernel
	[[nodiscard]] virtual IntersectionJudgement boxIntersect(const Point& center, double radius) const = 0;
	[[nodiscard]] virtual IntersectionJudgement boxIntersect(const Point& center, const Vector& radii) const = 0;

	template <typename Encoder>
	[[nodiscard]] std::array<typename Encoder::key_t, 7> encodeBounds(const Box& bbox) const;
};