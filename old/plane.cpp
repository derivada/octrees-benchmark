//
// Created by silvia.rodriguez.alcaraz on 26/2/21.
//

#include "Geometry/plane.hpp"
#include "Geometry/geometry.hpp"
#include "util.hpp"

Plane::Plane(const Vector& normal, const Point& centroid)
{
	A_ = normal.getX();
	B_ = normal.getY();
	C_ = normal.getZ();
	D_ = -(A_ * centroid.getX() + B_ * centroid.getY() + C_ * centroid.getZ());
}

double Plane::distance(const Point& p) // Not adding "toPoint" in the name to force polymorphism
{
	return fabs(signedDistance(p));
}

double Plane::signedDistance(const Point& p)
{
	double dist = A_ * p.getX() + B_ * p.getY() + C_ * p.getZ() + D_;
	dist /= sqrt(A_ * A_ + B_ * B_ + C_ * C_);

	return dist;
}

/** Estimate the best-fit plane of a set of points */
// TODO: Deberia ser un constructor. Como checkear si se ha construido bien?
int fitPlane(const std::vector<Lpoint*>& points, Plane& plane)
{
	if (points.size() < 3) return 0;
	Lpoint centroid;
	Vector normal = geometry::computePlane(points, centroid);
	plane         = Plane(normal, centroid);

	return isNumber(normal.getX());
}
