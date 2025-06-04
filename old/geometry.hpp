//
// Created by miguelyermo on 17/3/20.
//

/*
* FILENAME :  geometry.h  
* PROJECT  :  rule-based-classifier-cpp
* DESCRIPTION :
*  
*
*
*
*
* AUTHOR :    Miguel Yermo        START DATE : 09:55 17/3/20
*
*/

#ifndef CPP_GEOMETRY_H
#define CPP_GEOMETRY_H

#include "point.hpp"

#include <Eigen/Eigen>
#include <iostream>
#include <vector>

class Lpoint;

namespace geometry
{
class Limits
/**
     * Class to store the limits of the axis aligned minimum bounding box
     */
{
	private:
	double minX_{};
	double minY_{};
	double maxX_{};
	double maxY_{};

	public:
	Limits() = default;
	Limits(double minX, double minY, double maxX, double maxY) : minX_(minX), minY_(minY), maxX_(maxX), maxY_(maxY) {}

	// Getters
	double minX() const { return minX_; }
	double minY() const { return minY_; }
	double maxX() const { return maxX_; }
	double maxY() const { return maxY_; }
	double rangeX() const { return maxX_ - minX_; }
	double rangeY() const { return maxY_ - minY_; }


	// Overloads

	// Overload << operator
	friend std::ostream& operator<<(std::ostream& out, const Limits& l)
	{
		out << "minX: " << l.minX_ << " "
		    << "minY: " << l.minY_ << " "
		    << "maxX: " << l.maxX_ << " "
		    << "maxY: " << l.maxY_ << "\n";

		return out;
	}
};


// Function declarations
Point                                           computeCentroid(const std::vector<Lpoint*>& points);
std::tuple<Vector, double, std::vector<double>> computeGeometryArma(const std::vector<Lpoint*>& points);
Vector                                          computeNormal(const std::vector<Point*>& points);
Vector                                          computeNormal(const std::vector<Lpoint*>& points);
std::tuple<Vector, double, std::vector<double>> computeGeometryArma(const std::vector<Lpoint*>& points);
Vector                                          computePlane(const std::vector<Lpoint*>& points, Lpoint& centroid);


std::pair<Vector, std::vector<double>> computeGeometry(const std::vector<Lpoint*>& points, Lpoint& centroid_);

Limits getmbb(const std::vector<Lpoint>& points);
Limits getmbb(const std::vector<Lpoint*>& points);


} // namespace geometry
#endif //CPP_GEOMETRY_H
