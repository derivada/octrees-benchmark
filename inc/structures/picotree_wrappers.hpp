#pragma once

#include <iostream>
#include <pico_tree/kd_tree.hpp>
#include <streambuf>

#include "geometry/point.hpp"
#include "geometry/point_containers.hpp"
#include "picotree_profiler.hpp"

namespace pico_tree {

// point_traits: how do we access data on Point?
template <>
struct point_traits<Point> {
    using point_type = Point;
    using scalar_type = double;
    using size_type = size_t;
    static constexpr size_type dim = 3;

    inline static scalar_type const* data(point_type const& point) {
        // We take the address of x_. 
        // Because x, y, z are defined consecutively in the same access block,
        // (x_, y_, z_), standard compilers pack them contiguously for doubles.
        // id bytes should be ignored by pico_tree since dim is 3
        return &point.x_;
    }

    inline static constexpr size_type size(point_type const&) { return dim; }
};

// space_traits: how do we navigate PointsAoS?
template <>
struct space_traits<PointsAoS> {
    using space_type = PointsAoS;
    using point_type = Point;
    using scalar_type = double;
    using size_type = size_t;
    
    static size_type constexpr dim = 3;

    static_assert(dim != dynamic_extent, "STATIC_DIMENSION_REQUIRED");

    // Access point by index using PointsAoS::operator[]
    template <typename Index_>
    inline static point_type const& point_at(space_type const& space, Index_ idx) {
        return space[static_cast<size_type>(idx)];
    }

    inline static size_type size(space_type const& space) { return space.size(); }
    inline static size_type constexpr sdim(space_type const&) { return dim; }
};

}