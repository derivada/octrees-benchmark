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

#include "Geometry/point.hpp"
#include <algorithm>
#include <numbers>
#include <numeric>
#include <iomanip>
#include <cstdint>
#include "papi.h"



constexpr size_t LOG_FIELD_WIDTH = 32;

template<typename T>
bool checkMemoryAlligned(std::vector<T> vec) {
  void* data = vec.data();
	constexpr std::size_t CACHE_LINE_SIZE = 64; // Typical cache line size
  return (reinterpret_cast<std::uintptr_t>(data) % CACHE_LINE_SIZE == 0);
}

size_t vectorMemorySize(const auto& vec) {
	return sizeof(vec) + vec.size() * sizeof(typename std::decay_t<decltype(vec)>::value_type);
}

template <typename Point_t>
std::vector<Point_t> generateGridCloud(size_t n) {
    std::vector<Point_t> points;
    points.reserve(n * n * n);
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < n; j++)
            for (size_t k = 0; k < n; k++) {
                points.push_back(Point_t(i * n * n + j * n + k, i, j, k));
            }

    return points;
}

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

template<typename... Time_t>
inline bool are_the_same(const std::vector<Time_t...>& v1_, const std::vector<Time_t...>& v2_)
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

template<typename Time_t>
constexpr inline Time_t square(const Time_t& n)
{
	return n * n;
}

template<typename Time_t>
constexpr inline bool onInterval(const Time_t& n, const Time_t& min, const Time_t& max)
{
	return n > min && n < max;
}

template<typename Time_t>
constexpr inline Time_t midpoint(const Time_t& min, const Time_t& max)
{
	return min + (max - min) / 2;
}

template<typename Time_t>
constexpr inline bool isNumber(const Time_t x)
{
	return (!std::isnan(x) && !std::isinf(x));
}

template <typename Point_t>
inline double ccw(const Point_t* p1, const Point_t* p2, const Point_t* p3)
/**
 * Counter-clockwise situation of 3 points (ccw > 0, cw < 0, colinear = 0)
 */
{
	return (p2->getX() - p1->getX()) * (p3->getY() - p1->getY()) - (p2->getY() - p1->getY()) * (p3->getX() - p1->getX());
}

// TODO: put onRange and onIntensity in the same function? (the only use of onRange is here...)
template<typename Time_t>
inline bool onRange(const Time_t value, const Time_t offset, const Time_t center)
{
	return (value >= (center - offset) && value <= (center + offset));
}

template<typename Time_t>
inline bool onIntensity(const Time_t int1, const Time_t int2, const Time_t interval)
{
	const auto offset = std::max(int1, int2) * interval;

	return onRange(int1, offset, int2);
}

template<typename Time_t>
inline bool onDegree(const Vector& normal, const Time_t interval, const Vector& neighNormal)
{
	Vector degrees    = normal.vectorAngles();
	Vector epiDegrees = neighNormal.vectorAngles();
	Vector diffs(fabs(epiDegrees.getX() - degrees.getX()), fabs(epiDegrees.getY() - degrees.getY()),
	             fabs(epiDegrees.getZ() - degrees.getZ()));

	return (diffs.getX() < interval && diffs.getY() < interval && diffs.getZ() < interval);
}


inline std::string getCurrentDate(const std::string fmt = "%Y-%m-%d-%H:%M:%S") {
	auto t = std::time(nullptr);
	auto tm = *std::localtime(&t);
	std::ostringstream oss;
	oss << std::put_time(&tm, fmt.c_str());
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
