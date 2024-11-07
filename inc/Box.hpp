//
// Created by ruben.laso on 27/10/22.
//

#ifndef RULE_BASED_CLASSIFIER_CPP_BOX_HPP
#define RULE_BASED_CLASSIFIER_CPP_BOX_HPP

#include "Lpoint.hpp"
#include "point.hpp"
#include "util.hpp"
#include <vector>

class Box
{
	private:
	Point  center_{};
	Vector radii_{};
	Point  min_{};
	Point  max_{};

	public:
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

// Functions

inline Point mbb(const std::vector<Lpoint>& points, Vector& maxRadius)
/**
 * Computes the minimum bounding box of a set of points
 * @param points Array of points
 * @param numPoints Number of points
 * @param[out] maxRadius Maximum radius of the bounding box
 * @return (Point) center of the bounding box
 */
{
	Point min;
	Point max;

	min.setX(std::numeric_limits<double>::max());
	min.setY(std::numeric_limits<double>::max());
	min.setZ(std::numeric_limits<double>::max());

	max.setX(-std::numeric_limits<double>::max());
	max.setY(-std::numeric_limits<double>::max());
	max.setZ(-std::numeric_limits<double>::max());

	for (const Point& p : points)
	{
		if (p.getX() < min.getX()) { min.setX(p.getX()); }
		if (p.getX() > max.getX()) { max.setX(p.getX()); }
		if (p.getY() < min.getY()) { min.setY(p.getY()); }
		if (p.getY() > max.getY()) { max.setY(p.getY()); }
		if (p.getZ() < min.getZ()) { min.setZ(p.getZ()); }
		if (p.getZ() > max.getZ()) { max.setZ(p.getZ()); }
	}

	Box box(std::pair<Point, Point>{ min, max });

	maxRadius = box.radii();

	return midpoint(min, max);
}


inline Point mbb(const std::vector<Lpoint>& points, float& maxRadius)
/**
 * Computes the minimum bounding box of a set of points
 * @param points Array of points
 * @param numPoints Number of points
 * @param[out] maxRadius Maximum radius of the bounding box
 * @return (Point) center of the bounding box
 */
{
	Point min;
	Point max;

	min.setX(std::numeric_limits<double>::max());
	min.setY(std::numeric_limits<double>::max());
	min.setZ(std::numeric_limits<double>::max());

	max.setX(-std::numeric_limits<double>::max());
	max.setY(-std::numeric_limits<double>::max());
	max.setZ(-std::numeric_limits<double>::max());

	for (const Point& p : points)
	{
		if (p.getX() < min.getX()) { min.setX(p.getX()); }
		if (p.getX() > max.getX()) { max.setX(p.getX()); }
		if (p.getY() < min.getY()) { min.setY(p.getY()); }
		if (p.getY() > max.getY()) { max.setY(p.getY()); }
		if (p.getZ() < min.getZ()) { min.setZ(p.getZ()); }
		if (p.getZ() > max.getZ()) { max.setZ(p.getZ()); }
	}

	Box box(std::pair<Point, Point>{ min, max });

	maxRadius = std::max({ box.radii().getX(), box.radii().getY(), box.radii().getZ() });

	return midpoint(min, max);
}

inline Point mbb(const std::vector<Lpoint*>& points, float& maxRadius)
/**
 * Computes the minimum bounding box of a set of points
 * @param points Array of points' pointers
 * @param numPoints Number of points
 * @param[out] maxRadius Maximum radius of the bounding box
 * @return (Point) center of the bounding box
 */
{
	Point min;
	Point max;

	min.setX(std::numeric_limits<double>::max());
	min.setY(std::numeric_limits<double>::max());
	min.setZ(std::numeric_limits<double>::max());

	max.setX(-std::numeric_limits<double>::max());
	max.setY(-std::numeric_limits<double>::max());
	max.setZ(-std::numeric_limits<double>::max());

	for (Point* p : points)
	{
		if (p->getX() < min.getX()) { min.setX(p->getX()); }
		if (p->getX() > max.getX()) { max.setX(p->getX()); }
		if (p->getY() < min.getY()) { min.setY(p->getY()); }
		if (p->getY() > max.getY()) { max.setY(p->getY()); }
		if (p->getZ() < min.getZ()) { min.setZ(p->getZ()); }
		if (p->getZ() > max.getZ()) { max.setZ(p->getZ()); }
	}

	return Box::mbbRadii(min, max, maxRadius);
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
   * This functions defines the box whose inner points will be considered as neighs of Lpoint p.
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

inline bool insideBox2D(const Lpoint& point, const Vector& min, const Vector& max)
{
	if (point.getX() > min.getX() && point.getY() > min.getY())
	{
		if (point.getX() < max.getX() && point.getY() < max.getY()) { return true; }
	}

	return false;
}

inline bool insideCircle(const Lpoint& p, const Point& c, const double r)
{
	return (p.getX() - c.getX()) * (p.getX() - c.getX()) + (p.getY() - c.getY()) * (p.getY() - c.getY()) < r * r;
}

inline bool insideBox3D(const Lpoint& point, const Vector& min, const Vector& max)
{
	if (point.getX() > min.getX() && point.getY() > min.getY() && point.getZ() > min.getZ())
	{
		if (point.getX() < max.getX() && point.getY() < max.getY() && point.getZ() < max.getZ()) { return true; }
	}
	return false;
}

inline void makeBoxWithinHeights(const Lpoint& p, double radius, Vector& min, Vector& max, double zMin, double zMax)
{
	min.setX(p.getX() - radius);
	min.setY(p.getY() - radius);
	min.setZ(zMin);
	max.setX(p.getX() + radius);
	max.setY(p.getY() + radius);
	max.setZ(zMax);
}

#endif /* end of include guard: RULE_BASED_CLASSIFIER_CPP_BOX_HPP */
