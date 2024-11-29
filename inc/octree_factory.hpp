#pragma once

#include "Lpoint.hpp"
#include "Lpoint64.hpp"
#include "point.hpp"
#include "octree.hpp"
#include "octree_linear.hpp"
#include "octree_linear_old.hpp"

#include <type_traits>

// Types of octree
enum class Octree_t {
    pointer,
    linear
};

template <PointType T>
std::string getPointName();

template <>
std::string getPointName<Lpoint64>() {
    return "Lpoint64";
}

template <>
std::string getPointName<Lpoint>() {
    return "Lpoint";
}

template <>
std::string getPointName<Point>() {
    return "Point";
}


// Get octree name along with its point type
template <OctreeType Octree_t, PointType Point_t>
std::string getOctreeName() {
    std::string pointTypeName = getPointName<Point_t>();
    if constexpr (std::is_same_v<Octree_t, LinearOctree<Point_t>>) {
        return "linear <" + pointTypeName + ">";
    } else if constexpr (std::is_same_v<Octree_t, Octree<Point_t>>) {
        return "pointer <" + pointTypeName + ">";
    } else {
        static_assert(false, "Unsupported octree type in getOctreeName");
    }
}

// Build octree of a given type and point type
template <Octree_t octree, PointType Point_t>
inline auto octreeFactory(std::vector<Point_t>& points) {
    if constexpr (octree == Octree_t::pointer) {
        return Octree<Point_t>(points);
    } else if constexpr (octree == Octree_t::linear) {
        return LinearOctree<Point_t>(points);
    } else {
        static_assert(false, "Unsupported octree type in octreeFactory");
    }
}