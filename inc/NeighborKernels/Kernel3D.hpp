//
// Created by ruben.laso on 13/10/22.
//

#ifndef KERNEL3D_HPP
#define KERNEL3D_HPP

#include "KernelAbstract.hpp"
#include "PointEncoding/common.hpp"
#include <array>

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

	/**
	 * Returns a vector where the first element is the encoding of the kernel center and the next 6
	 * are the encodings of the center point of each side of the bounding volume occupied by the kernel.
	 */
	template <typename Encoder>
	[[nodiscard]] std::array<typename Encoder::key_t, 7> encodeBounds(const Box& bbox) const {
		double dx[7] = {0, radii().getX(), -radii().getX(), 0, 0, 0, 0};
		double dy[7] = {0, 0, 0, radii().getY(), -radii().getY(), 0, 0};
		double dz[7] = {0, 0, 0, 0, 0, radii().getZ(), -radii().getZ()};
		auto result =std::array<typename Encoder::key_t, 7>();
		for(int i = 0; i<7; i++) {
			auto point = Point(center().getX() + dx[i], center().getY() + dy[i], center().getZ() + dz[i]);
			result[i] = PointEncoding::encodeFromPoint<Encoder>(point, bbox);
		}
		return result;
	}
};

#endif /* end of include guard: KERNEL3D_HPP */
