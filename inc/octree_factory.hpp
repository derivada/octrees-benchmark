#pragma once

#include "Geometry/Lpoint.hpp"
#include "Geometry/Lpoint64.hpp"
#include "Geometry/point.hpp"
#include "octree.hpp"
#include "octree_linear.hpp"
#include "PointEncoding/common.hpp"

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

template <OctreeType Octree_t, PointType Point_t, typename Encoder_t>
std::string getOctreeName() {
    auto pointTypeName = getPointName<Point_t>();
    auto encoderTypeName = PointEncoding::getEncoderName<Encoder_t>();
    if constexpr (std::is_same_v<Octree_t, LinearOctree<Point_t, Encoder_t>>) {
        return "linear <" + pointTypeName + ", " + encoderTypeName + ">";
    } else if constexpr (std::is_same_v<Octree_t, Octree<Point_t>>) {
        return "pointer <" + pointTypeName + ">";
    }
}

// Build octree of a given type and point type
template <Octree_t octree, PointType Point_t, typename Encoder_t>
inline auto octreeFactory(std::vector<Point_t>& points) {
    if constexpr (octree == Octree_t::pointer) {
        return Octree<Point_t>(points);
    } else if constexpr (octree == Octree_t::linear) {
        return LinearOctree<Point_t, Encoder_t>(points);
    }
}