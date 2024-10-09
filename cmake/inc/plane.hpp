//
// Created by silvia.rodriguez.alcaraz on 26/2/21.
//

#ifndef PLANE_H
#define PLANE_H

#include "geometry.hpp"
#include "point.hpp"
#include "Lpoint.hpp"
class Plane
/**
 * Canonical implemention of a 3D geometric plane (Ax + By + Cz + D = 0)
 */
{
	double A_{}, B_{}, C_{}, D_{};

	public:
	Plane() = default;
	Plane(const Vector& normal, const Point& centroid);
	double distance(const Point& p);
	double signedDistance(const Point& p);

	// Overload << operator for Point.
	friend std::ostream& operator<<(std::ostream& out, const Plane& p)
	{
		out << p.A_ << " " << p.B_ << " " << p.C_ << " " << p.D_;
		return out;
	}
};

int fitPlane(const std::vector<Lpoint*>& points, Plane& plane);
#endif //PLANE_H
