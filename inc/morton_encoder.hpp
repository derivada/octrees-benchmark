#pragma once

#include "libmorton/morton.h"
#include "point.hpp"
#include "Box.hpp"

using morton_t = uint_fast64_t;
using coords_t = uint_fast32_t;

/**
* @namespace MortonEncoder
* 
* @brief A front-end for libmorton for getting adequate encodings needed on the linear octree
* 
* @cite Jeroen Baert. Libmorton: C++ Morton Encoding/Decoding Library. https://github.com/Forceflow/libmorton/tree/main
* 
* @authors Pablo Díaz Viñambres 
* 
* @date 16/11/2024
* 
*/
namespace MortonEncoder {
    /// @brief The maximum depth that this encoding allows (in Morton 64 bit integers, we need 3 bits for each level, so 21)
    constexpr unsigned MAX_DEPTH = 21;

    /// @brief The minimum unit of length of the encoded coordinates
    constexpr double EPS = 1.0f / (1 << MAX_DEPTH);

    /// @brief The minimum (strict) upper bound for every Morton code. Equal to the unused bit followed by 63 zeros.
    constexpr morton_t UPPER_BOUND = 0x8000000000000000;

    /// @brief The amount of bits that are not used, in Morton encodings this is the MSB of the key
    constexpr uint32_t UNUSED_BITS = 1;

    inline void getAnchorCoords(const Point& p, const Box &bbox, coords_t &x, coords_t &y, coords_t &z) {
        // Put physical coords into the unit cube
        float x_transf = ((p.getX() - bbox.center().getX())  + bbox.radii().getX()) / (2 * bbox.radii().getX());
        float y_transf = ((p.getY() - bbox.center().getY())  + bbox.radii().getY()) / (2 * bbox.radii().getY());
        float z_transf = ((p.getZ() - bbox.center().getZ())  + bbox.radii().getZ()) / (2 * bbox.radii().getZ());

        // Edge case of coordinates falling exactly into 1, which would be problematic
        if(x_transf + EPS >= 1.0f) x_transf = 1.0f - EPS;
        if(y_transf + EPS >= 1.0f) y_transf = 1.0f - EPS;
        if(z_transf + EPS >= 1.0f) z_transf = 1.0f - EPS;
        
        // Scale to [0,2^L]^3 for morton encoding
        x = (coords_t) (x_transf * (1 << (MAX_DEPTH)));
        y = (coords_t) (y_transf * (1 << (MAX_DEPTH)));
        z = (coords_t) (z_transf * (1 << (MAX_DEPTH)));
    }

    // This methods should not be called from the outside, as they use geometric information computed here
    // (radii and center) of the point cloud
    inline morton_t encodeMortonPoint(const Point& p, const Box &bbox) {
        // Utility method combining the two above
        coords_t x, y, z;
        getAnchorCoords(p, bbox, x, y, z);
        return libmorton::morton3D_64_encode(x, y, z);
    }

    inline void decodeMorton(morton_t code, coords_t &x, coords_t &y, coords_t &z) {
        libmorton::morton3D_64_decode(code, x, y, z);
    }

    inline std::pair<Point, Vector> getCenterAndRadii(morton_t code, uint32_t level, const Box &bbox, const float* halfLengths, const Vector* precomputedRadii) {
        // Decode the points
        coords_t min_x, min_y, min_z;
        decodeMorton(code, min_x, min_y, min_z);

        // Find the physical center by multiplying the encoding with the halflength
        // to get to the low corner of the cell, and then adding the radii of the cell
        Point center = Point(
            bbox.minX() + min_x * halfLengths[0] * 2, 
            bbox.minY() + min_y * halfLengths[1] * 2, 
            bbox.minZ() + min_z * halfLengths[2] * 2
        ) + precomputedRadii[level];
        
        return {center, precomputedRadii[level]};
    }

    constexpr uint32_t countLeadingZeros(uint64_t x)
    {
        #if defined(__GNUC__) || defined(__clang__)
            if (x == 0) return 8 * sizeof(uint64_t);
            return __builtin_clzll(x);
        #else
            uint32_t depth = 0;
            for (; x != 1; x >>= 3, depth++);
            return depth;
        #endif
    }

    constexpr bool isPowerOf8(morton_t n) {
        morton_t lz = countLeadingZeros(n - 1) - UNUSED_BITS;
        return lz % 3 == 0 && !(n & (n - 1));
    }

    // Get the level in the octree of the given morton code
    inline uint32_t getLevel(morton_t range) {
        assert(isPowerOf8(range));
        if(range == UPPER_BOUND)
            return 0U;
        return (countLeadingZeros(range - 1) - UNUSED_BITS) / 3;
    }

    // Get the sibling ID of the code at a given level
    constexpr uint32_t getSiblingId(morton_t code, uint32_t level) {
        // Shift 3*(21-level) to get the 3 bits corresponding to the level
        return (code >> (3u * (MAX_DEPTH - level))) & 7u;
    }   

    // Get the maximum range allowed in a level of the tree, equal to 1 << 3*(maxdepth-treeLevel)
    // e.g. at level 0 the range is the entire 63 bit span, at level 10 the range is 11*3 bit span
    // at level 20 (last to minimum), the range will just be 8 between each node, i.e. the 8 siblings that
    // can be on max level 21 between two nodes at level 20
    constexpr morton_t nodeRange(uint32_t treeLevel)
    {
        assert(treeLevel < MAX_DEPTH);
        uint32_t shifts = MAX_DEPTH - treeLevel;

        return 1ul << (3u * shifts);
    }

    
    constexpr morton_t encodePlaceholderBit(morton_t code, int prefixLength) {
        int nShifts = 3 * MAX_DEPTH - prefixLength;
        morton_t ret = code >> nShifts;
        morton_t placeHolderMask = 1UL << prefixLength;

        return placeHolderMask | ret;
    }

    constexpr uint32_t decodePrefixLength(morton_t code) {
        return 8 * sizeof(morton_t) - 1 - countLeadingZeros(code);
    }

    constexpr morton_t decodePlaceholderBit(morton_t code) {
        int prefixLength        = decodePrefixLength(code);
        morton_t placeHolderMask = 1UL << prefixLength;
        morton_t ret             = code ^ placeHolderMask;

        return ret << (3 * MAX_DEPTH - prefixLength);
    }


    constexpr int32_t commonPrefix(morton_t key1, morton_t key2) {
        return int32_t(countLeadingZeros(key1 ^ key2)) - UNUSED_BITS;
    }

    constexpr unsigned octalDigit(morton_t code, uint32_t position) {
        return (code >> (3u * (MAX_DEPTH - position))) & 7u;
    }

    constexpr uint32_t log8ceil(morton_t n) {
        if (n == 0) { return 0; }

        uint32_t lz = countLeadingZeros(n - 1);
        return MAX_DEPTH - (lz - UNUSED_BITS) / 3;
    }
};