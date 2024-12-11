
#pragma once

#include "libmorton/morton.h"
#include "Geometry/point.hpp"
#include "Geometry/Box.hpp"

namespace PointEncoding {

    template <typename Encoder>
    std::string getEncoderName() {
        return Encoder::NAME;
    }

    template <typename Encoder>
    inline void getAnchorCoords(const Point& p, const Box &bbox, 
        typename Encoder::coords_t &x, typename Encoder::coords_t &y, typename Encoder::coords_t &z) {
        // Put physical coords into the unit cube
        float x_transf = ((p.getX() - bbox.center().getX())  + bbox.radii().getX()) / (2 * bbox.radii().getX());
        float y_transf = ((p.getY() - bbox.center().getY())  + bbox.radii().getY()) / (2 * bbox.radii().getY());
        float z_transf = ((p.getZ() - bbox.center().getZ())  + bbox.radii().getZ()) / (2 * bbox.radii().getZ());

        // Edge case of coordinates falling exactly into 1, which would be problematic
        if(x_transf + Encoder::EPS >= 1.0f) x_transf = 1.0f - Encoder::EPS;
        if(y_transf + Encoder::EPS >= 1.0f) y_transf = 1.0f - Encoder::EPS;
        if(z_transf + Encoder::EPS >= 1.0f) z_transf = 1.0f - Encoder::EPS;
        
        // Scale to [0,2^L]^3 for morton encoding
        x = (typename Encoder::coords_t) (x_transf * (1 << (Encoder::MAX_DEPTH)));
        y = (typename Encoder::coords_t) (y_transf * (1 << (Encoder::MAX_DEPTH)));
        z = (typename Encoder::coords_t) (z_transf * (1 << (Encoder::MAX_DEPTH)));
    }

    template <typename Encoder>
    inline std::pair<Point, Vector> getCenterAndRadii(typename Encoder::key_t code, uint32_t level, const Box &bbox, const float* halfLengths, const Vector* precomputedRadii) {
        // Decode the points
        typename Encoder::coords_t min_x, min_y, min_z;
        Encoder::decode(code, min_x, min_y, min_z);

        // Find the physical center by multiplying the encoding with the halflength
        // to get to the low corner of the cell, and then adding the radii of the cell
        Point center = Point(
            bbox.minX() + min_x * halfLengths[0] * 2, 
            bbox.minY() + min_y * halfLengths[1] * 2, 
            bbox.minZ() + min_z * halfLengths[2] * 2
        ) + precomputedRadii[level];
        
        return {center, precomputedRadii[level]};
    }

    template <typename Encoder>
    constexpr uint32_t countLeadingZeros(typename Encoder::key_t x)
    {
        #if defined(__GNUC__) || defined(__clang__)
            if (x == 0) return 8 * sizeof(typename Encoder::key_t);
            // 64-bit keys
            if constexpr (sizeof(typename Encoder::key_t) == 8) {
                return __builtin_clzll(x);
            }
            // 32-bit keys
            else {
                return __builtin_clz(x);
            }
        #else
            uint32_t depth = 0;
            for (; x != 1; x >>= 3, depth++);
            return depth;
        #endif
    }

    template <typename Encoder>
    constexpr bool isPowerOf8(typename Encoder::key_t n) {
        typename Encoder::key_t lz = countLeadingZeros<Encoder>(n - 1) - Encoder::UNUSED_BITS;
        return lz % 3 == 0 && !(n & (n - 1));
    }

    // Get the level in the octree of the given morton code
    template <typename Encoder>
    inline uint32_t getLevel(typename Encoder::key_t range) {
        assert(isPowerOf8<Encoder>(range));
        if(range == Encoder::UPPER_BOUND)
            return typename Encoder::key_t(0);
        return (countLeadingZeros<Encoder>(range - typename Encoder::key_t(1)) - Encoder::UNUSED_BITS) / typename Encoder::key_t(3);
    }

    // Get the sibling ID of the code at a given level
    template <typename Encoder>
    constexpr uint32_t getSiblingId(typename Encoder::key_t code, uint32_t level) {
        // Shift 3*(21-level) to get the 3 bits corresponding to the level
        return (code >> (typename Encoder::key_t(3) * (Encoder::MAX_DEPTH - level))) & typename Encoder::key_t(7);
    }   

    // Get the maximum range allowed in a level of the tree, equal to 1 << 3*(maxdepth-treeLevel)
    // e.g. at level 0 the range is the entire 63 bit span, at level 10 the range is 11*3 bit span
    // at level 20 (last to minimum), the range will just be 8 between each node, i.e. the 8 siblings that
    // can be on max level 21 between two nodes at level 20
    template <typename Encoder>
    constexpr typename Encoder::key_t nodeRange(uint32_t treeLevel)
    {
        assert(treeLevel < Encoder::MAX_DEPTH);
        uint32_t shifts = Encoder::MAX_DEPTH - treeLevel;

        return 1ul << (typename Encoder::key_t(3) * shifts);
    }

    template <typename Encoder>
    constexpr typename Encoder::key_t encodePlaceholderBit(typename Encoder::key_t code, int prefixLength) {
        int nShifts = 3 * Encoder::MAX_DEPTH - prefixLength;
        typename Encoder::key_t ret = code >> nShifts;
        typename Encoder::key_t placeHolderMask = typename Encoder::key_t(1) << prefixLength;

        return placeHolderMask | ret;
    }

    template <typename Encoder>
    constexpr uint32_t decodePrefixLength(typename Encoder::key_t code) {
        return typename Encoder::key_t(8) * sizeof(typename Encoder::key_t) - typename Encoder::key_t(1) - countLeadingZeros<Encoder>(code);
    }

    template <typename Encoder>
    constexpr typename Encoder::key_t decodePlaceholderBit(typename Encoder::key_t code) {
        int prefixLength        = decodePrefixLength<Encoder>(code);
        typename Encoder::key_t placeHolderMask = typename Encoder::key_t(1) << prefixLength;
        typename Encoder::key_t ret             = code ^ placeHolderMask;

        return ret << (typename Encoder::key_t(3) * Encoder::MAX_DEPTH - prefixLength);
    }

    template <typename Encoder>
    constexpr int32_t commonPrefix(typename Encoder::key_t key1, typename Encoder::key_t key2) {
        return int32_t(countLeadingZeros<Encoder>(key1 ^ key2)) - Encoder::UNUSED_BITS;
    }

    template <typename Encoder>
    constexpr unsigned octalDigit(typename Encoder::key_t code, uint32_t position) {
        return (code >> (typename Encoder::key_t(3) * (Encoder::MAX_DEPTH - position))) & typename Encoder::key_t(7);
    }

    template <typename Encoder>
    constexpr uint32_t log8ceil(typename Encoder::key_t n) {
        if (n == typename Encoder::key_t(0)) { return 0; }

        uint32_t lz = countLeadingZeros<Encoder>(n - typename Encoder::key_t(1));
        return Encoder::MAX_DEPTH - (lz - Encoder::UNUSED_BITS) / 3;
    }
}