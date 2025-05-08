#pragma once

#include "Geometry/Lpoint.hpp"
#include "Geometry/point.hpp"

/**
 * This file simply includes some class names for the templated classes.
 * Originally this also included Octree and LinearOctree, but that gave template deduction issues
 * when I tried using type_names.hpp inside one of the octree.hpp classes, since then a cyclic
 * dependence would be created.
 * 
 * For getting the string with the runtime type of a generic Octree_t, just do an constexpr if-else like in
 * octree_benchmark.hpp
 */
template <typename T>
std::string getPointName();

template <>
inline std::string getPointName<Lpoint>() { return "Lpoint"; }

template <>
inline std::string getPointName<Point>() { return "Point"; }