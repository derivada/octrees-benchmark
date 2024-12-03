//
// Created by miguel.yermo on 6/03/20.
//

/*
* FILENAME :  util.h  
* PROJECT  :  rule-based-classifier-cpp
* DESCRIPTION :
*  
*
*
*
*
* AUTHOR :    Miguel Yermo        START DATE : 14:17 6/03/20
*
*/

#pragma once

#include "point.hpp"
#include <algorithm>
#include <numbers>
#include <numeric>
#include <iomanip>

// Memory Handling
namespace mem
{
template<class C>
void free(C& c)
{
	c.clear();
	c.shrink_to_fit();
}
} // namespace mem

// Utility to check if a type is a subclass of Point
template <typename T>
concept PointType = std::is_base_of_v<Point, T>;

// Concept for Octree classes
template <typename T>
concept OctreeType = requires {
    typename T::PointType;  // T should have a member named PointType
    requires PointType<typename T::PointType>;  // PointType should satisfy the PointType concept
};


template<typename... T>
inline bool are_the_same(const std::vector<T...>& v1_, const std::vector<T...>& v2_)
{
	if (v1_.size() != v2_.size())
	{
		std::cerr << "Vectors are not of the same length" << '\n';
		return false;
	}

	auto v1 = v1_;
	auto v2 = v2_;
	std::sort(std::begin(v1), std::end(v1));
	std::sort(std::begin(v2), std::end(v2));

	size_t mismatches = 0;

	for (size_t i = 0; i < v1.size(); i++)
	{
		if (v1[i] != v2[i]) { ++mismatches; }
	}

	if (mismatches > 0) { std::cerr << "Vectors have " << mismatches << " mismatches" << '\n'; }

	return mismatches == 0;
}

template<typename Tp>
[[maybe_unused, nodiscard]] constexpr inline Tp deg2rad(const Tp& deg)
{
	return deg * std::numbers::pi / 180;
}

template<typename Tp>
[[maybe_unused, nodiscard]] constexpr inline Tp rad2deg(const Tp& rad)
{
	return rad * 180 / std::numbers::pi;
}

template<typename Tp, typename Container_t>
constexpr inline Tp average(const Container_t& container)
{
	if (container.empty()) { return {}; }

	return std::reduce(std::begin(container), std::end(container), Tp{}) / container.size();
}

template<typename Tp, typename Container_t, typename F>
constexpr inline Tp average(const Container_t& container, F unaryOp)
{
	if (container.empty()) { return {}; }

	return std::transform_reduce(std::begin(container), std::end(container), Tp{}, std::plus{}, unaryOp) /
	       container.size();
}

template<typename T>
constexpr inline T square(const T& n)
{
	return n * n;
}

template<typename T>
constexpr inline bool onInterval(const T& n, const T& min, const T& max)
{
	return n > min && n < max;
}

template<typename T>
constexpr inline T midpoint(const T& min, const T& max)
{
	return min + (max - min) / 2;
}

template<typename T>
constexpr inline bool isNumber(const T x)
{
	return (!std::isnan(x) && !std::isinf(x));
}

template <PointType Point_t>
inline double ccw(const Point_t* p1, const Point_t* p2, const Point_t* p3)
/**
 * Counter-clockwise situation of 3 points (ccw > 0, cw < 0, colinear = 0)
 */
{
	return (p2->getX() - p1->getX()) * (p3->getY() - p1->getY()) - (p2->getY() - p1->getY()) * (p3->getX() - p1->getX());
}

// TODO: put onRange and onIntensity in the same function? (the only use of onRange is here...)
template<typename T>
inline bool onRange(const T value, const T offset, const T center)
{
	return (value >= (center - offset) && value <= (center + offset));
}

template<typename T>
inline bool onIntensity(const T int1, const T int2, const T interval)
{
	const auto offset = std::max(int1, int2) * interval;

	return onRange(int1, offset, int2);
}

template<typename T>
inline bool onDegree(const Vector& normal, const T interval, const Vector& neighNormal)
{
	Vector degrees    = normal.vectorAngles();
	Vector epiDegrees = neighNormal.vectorAngles();
	Vector diffs(fabs(epiDegrees.getX() - degrees.getX()), fabs(epiDegrees.getY() - degrees.getY()),
	             fabs(epiDegrees.getZ() - degrees.getZ()));

	return (diffs.getX() < interval && diffs.getY() < interval && diffs.getZ() < interval);
}


inline std::string getCurrentDate() {
	auto t = std::time(nullptr);
	auto tm = *std::localtime(&t);
	std::ostringstream oss;
	oss << std::put_time(&tm, "%Y-%m-%d-%H:%M:%S");
	return oss.str();
}

// Width of the progress bar
constexpr int BAR_WIDTH = 70; 

inline void progressBar(size_t progress, size_t total) {
    int pos = static_cast<int>((float) progress  * BAR_WIDTH / 100.0);
    std::cout << "Reading point cloud [";
    for (int i = 0; i < BAR_WIDTH; ++i) {
        if (i < pos) {
            std::cout << "=";
        } else if (i == pos) {
            std::cout << ">";
        } else {
            std::cout << " ";
        }
    }
    std::cout << "] " << std::setw(3) << static_cast<int>(progress) << "%\r";
    std::cout.flush();
}

// in a .txt file we dont have a total number of points before reading, so we just print how many points we have read periodically
inline void progressNumber(size_t progress) {
    std::cout << "Points read: " << progress << "\r";
    std::cout.flush();
}