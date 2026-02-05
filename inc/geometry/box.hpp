#pragma once

#include <vector>
#include "point.hpp"
#include "util.hpp"

class Box
{
	private:
	Point  center_{};
	Vector radii_{};
	Point  min_{};
	Point  max_{};

	public:
	Box() = default;
	explicit Box(const Point& p, const Vector& radii) : center_(p), radii_(radii), min_(p - radii), max_(p + radii) {}
	explicit Box(const std::pair<Point, Point>& min_max) :
	  center_(midpoint(min_max.first, min_max.second)), radii_((min_max.second - min_max.first) / 2), min_(min_max.first),
	  max_(min_max.second)
	{}

	[[nodiscard]] inline bool isInside(const Point& p) const
	{
		if (p.getX() > min_.getX() && p.getY() > min_.getY() && p.getZ() > min_.getZ())
		{
			if (p.getX() < max_.getX() && p.getY() < max_.getY() && p.getZ() < max_.getZ()) { return true; }
		}
		return false;
	}

	[[nodiscard]] inline const Point& min() const { return min_; }
	[[nodiscard]] inline const Point& max() const { return max_; }

	inline void min(const Point& v)
	{
		min_    = v;
		radii_  = mbbRadii(min_, max_);
		center_ = mbbCenter(min_, radii_);
	}

	inline void max(const Point& v)
	{
		max_    = v;
		radii_  = mbbRadii(min_, max_);
		center_ = mbbCenter(min_, radii_);
	}

	[[nodiscard]] inline double minX() const { return min_.getX(); }
	[[nodiscard]] inline double minY() const { return min_.getY(); }
	[[nodiscard]] inline double minZ() const { return min_.getZ(); }

	[[nodiscard]] inline double maxX() const { return max_.getX(); }
	[[nodiscard]] inline double maxY() const { return max_.getY(); }
	[[nodiscard]] inline double maxZ() const { return max_.getZ(); }

	[[nodiscard]] inline Vector center() const { return center_; }
	[[nodiscard]] inline Vector radii() const { return radii_; }

	/** Calculate the center of the bounding box */
	static inline Point mbbCenter(const Vector& min, const Vector& radius)
	{
		Vector center(min.getX() + radius.getX(), min.getY() + radius.getY(), min.getZ() + radius.getZ());

		return center;
	}

	/** Calculate the radius in each axis and save the max radius of the bounding box */
	static inline Vector mbbRadii(const Vector& min, const Vector& max, float& maxRadius)
	{
		const auto x = (max.getX() - min.getX()) / 2.0;
		const auto y = (max.getY() - min.getY()) / 2.0;
		const auto z = (max.getZ() - min.getZ()) / 2.0;

		Vector radii(x, y, z);

		maxRadius = std::max({ x, y, z });

		return radii;
	}

	/** Calculate the radius in each axis */
	static inline Vector mbbRadii(const Vector& min, const Vector& max)
	{
		const auto x = (max.getX() - min.getX()) / 2.0;
		const auto y = (max.getY() - min.getY()) / 2.0;
		const auto z = (max.getZ() - min.getZ()) / 2.0;

		return { x, y, z };
	}
};


template<PointContainer Container>
inline Point mbb(const Container& points, Vector& maxRadius)
{
	double minX = std::numeric_limits<double>::max();
	double minY = std::numeric_limits<double>::max();
	double minZ = std::numeric_limits<double>::max();
	double maxX = -std::numeric_limits<double>::max();
	double maxY = -std::numeric_limits<double>::max();
	double maxZ = -std::numeric_limits<double>::max();

	#pragma omp parallel
	{
		double localMinX = minX, localMinY = minY, localMinZ = minZ;
		double localMaxX = maxX, localMaxY = maxY, localMaxZ = maxZ;

		#pragma omp for schedule(static) nowait
		for (size_t i = 0; i < points.size(); ++i) {
			const Point& p = points[i];
			localMinX = std::min(localMinX, p.getX());
			localMaxX = std::max(localMaxX, p.getX());
			localMinY = std::min(localMinY, p.getY());
			localMaxY = std::max(localMaxY, p.getY());
			localMinZ = std::min(localMinZ, p.getZ());
			localMaxZ = std::max(localMaxZ, p.getZ());
		}

		#pragma omp critical
		{
			minX = std::min(minX, localMinX);
			maxX = std::max(maxX, localMaxX);
			minY = std::min(minY, localMinY);
			maxY = std::max(maxY, localMaxY);
			minZ = std::min(minZ, localMinZ);
			maxZ = std::max(maxZ, localMaxZ);
		}
	}

	Point min(minX, minY, minZ);
	Point max(maxX, maxY, maxZ);
	Box box(std::pair<Point, Point>{ min, max });

	maxRadius = box.radii();
	return midpoint(min, max);
}


inline void makeBox(const Point& p, double radius, Vector& min, Vector& max)
{
	min.setX(p.getX() - radius);
	min.setY(p.getY() - radius);
	min.setZ(p.getZ() - radius);
	max.setX(p.getX() + radius);
	max.setY(p.getY() + radius);
	max.setZ(p.getZ() + radius);
}

inline void makeBoxCylinder(const Point& p, double radius, Vector& min, Vector& max)
{
	min.setX(p.getX() - radius);
	min.setY(p.getY() - radius);
	min.setZ(0);
	max.setX(p.getX() + radius);
	max.setY(p.getY() + radius);
	max.setZ(std::numeric_limits<double>::max());
}

inline void makeBox(const Point& p, const Vector& radius, Vector& min, Vector& max)
/**
   * This functions defines the box whose inner points will be considered as neighs of Point p.
   * @param p
   * @param radius Vector of radius. One per spatial direction.
   * @param min
   * @param max
   */
{
	min.setX(p.getX() - radius.getX());
	min.setY(p.getY() - radius.getY());
	min.setZ(p.getZ() - radius.getZ());
	max.setX(p.getX() + radius.getX());
	max.setY(p.getY() + radius.getY());
	max.setZ(p.getZ() + radius.getZ());
}

inline bool insideBox2D(const Point& point, const Vector& min, const Vector& max)
{
	if (point.getX() > min.getX() && point.getY() > min.getY())
	{
		if (point.getX() < max.getX() && point.getY() < max.getY()) { return true; }
	}

	return false;
}

inline bool insideCircle(const Point& p, const Point& c, const double r)
{
	return (p.getX() - c.getX()) * (p.getX() - c.getX()) + (p.getY() - c.getY()) * (p.getY() - c.getY()) < r * r;
}

inline bool insideBox3D(const Point& point, const Vector& min, const Vector& max)
{
	if (point.getX() > min.getX() && point.getY() > min.getY() && point.getZ() > min.getZ())
	{
		if (point.getX() < max.getX() && point.getY() < max.getY() && point.getZ() < max.getZ()) { return true; }
	}
	return false;
}

inline void makeBoxWithinHeights(const Point& p, double radius, Vector& min, Vector& max, double zMin, double zMax)
{
	min.setX(p.getX() - radius);
	min.setY(p.getY() - radius);
	min.setZ(zMin);
	max.setX(p.getX() + radius);
	max.setY(p.getY() + radius);
	max.setZ(zMax);
}