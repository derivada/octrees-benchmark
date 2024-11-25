#pragma once
#include "Lpoint.hpp"
#include "octree.hpp"
#include "octree_linear.hpp"
#include "octree_linear_old.hpp"

enum class Octree_t 
{
	pointer,
    linear,
    linear_old
};


template <typename Octree_t>
std::string getOctreeName() {
    if constexpr (std::is_same_v<Octree_t, LinearOctree>) {
        return "linear";
    } else if constexpr (std::is_same_v<Octree_t, Octree>) {
        return "pointer";
    } else if constexpr (std::is_same_v<Octree_t, LinearOctreeOld>) {
        return "linearOld";
    } else {
        return "Unknown";
    }
}

template<Octree_t octree>
inline auto octreeFactory(std::vector<Lpoint>& points)
{
	if constexpr (octree == Octree_t::pointer) { return Octree(points); }
	else if constexpr (octree == Octree_t::linear) { return LinearOctree(points); }
    else if constexpr (octree == Octree_t::linear_old) { return LinearOctreeOld(points); }
	else { static_assert(false, "Unsupported Octree_t type in octreeFactory");  }
}