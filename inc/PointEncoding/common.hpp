
#pragma once

#include "libmorton/morton.h"
#include <bitset>
#include "Geometry/point.hpp"
#include "Geometry/PointMetadata.hpp"
#include "Geometry/Box.hpp"
#include <vector>
#include <tuple>

/**
 * @namespace PointEncoding
 * @brief Namespace for point encoding utilities such as Morton and Hilbert SFC encoding and aux functions to extract information from keys.
 */
namespace PointEncoding {

    /**
     * @struct NoEncoder
     * @brief A dummy encoder that does nothing. Used for indicating that pointer octrees points have not encoded and sorted in the corresponding order.
     */
    struct NoEncoder { };
    
    /**
     * @brief Get anchor integer coordinates for a point within a bounding box. Necessary before point encoding.
     * 
     * @param p The point.
     * @param bbox The global bounding box of the point cloud.
     */
    template <typename Encoder>
    inline void getAnchorCoords(const Point& p, const Box &bbox, 
        typename Encoder::coords_t &x, typename Encoder::coords_t &y, typename Encoder::coords_t &z) {
        // Put physical coords into the unit cube
        double x_transf = ((p.getX() - bbox.center().getX())  + bbox.radii().getX()) / (2 * bbox.radii().getX());
        double y_transf = ((p.getY() - bbox.center().getY())  + bbox.radii().getY()) / (2 * bbox.radii().getY());
        double z_transf = ((p.getZ() - bbox.center().getZ())  + bbox.radii().getZ()) / (2 * bbox.radii().getZ());
        
        // Scale to [0,2^L)^3 for morton encoding, handle edge case where coordinate could be 2^L if _transf is exactly 1.0
        typename Encoder::coords_t maxCoord = (1u << Encoder::MAX_DEPTH) - 1u;
        x = std::min((typename Encoder::coords_t) (x_transf * (1 << (Encoder::MAX_DEPTH))), maxCoord);
        y = std::min((typename Encoder::coords_t) (y_transf * (1 << (Encoder::MAX_DEPTH))), maxCoord);
        z = std::min((typename Encoder::coords_t) (z_transf * (1 << (Encoder::MAX_DEPTH))), maxCoord);
    }

    template <typename Encoder>
    inline typename Encoder::key_t encodeFromPoint(const Point& p, const Box &bbox) {
        typename Encoder::coords_t x, y, z;
        getAnchorCoords<Encoder>(p, bbox, x, y, z);
		return Encoder::encode(x, y, z);
    }

    /**
     * @brief Get the center and radii of an octant at a given octree level.
     * 
     * @tparam Encoder The encoder type.
     * @param code The encoded key.
     * @param level The level in the octree.
     * @param bbox The bounding box.
     * @param halfLengths The half lengths of the bounding box.
     * @param precomputedRadii The precomputed radii corresponding to that level.
     * @return A pair containing the center point and the radii vector.
     */
    template <typename Encoder>
    inline std::pair<Point, Vector> getCenterAndRadii(typename Encoder::key_t code, uint32_t level, const Box &bbox, const double* halfLengths, const Vector* precomputedRadii) {
        // Decode the points back into their integer coordinates
        typename Encoder::coords_t min_x, min_y, min_z;
        Encoder::decode(code, min_x, min_y, min_z);

        // Now adjust the coordinates so they indicate the lowest code in the current level
        // In Morton curves this is not needed, but in Hilbert curves it is, since it can return any corner instead of lower one we need
        // Fun fact: finding this mistake when adding Hilbert curves took 6 hours of debugging
        typename Encoder::coords_t mask = ((1u << Encoder::MAX_DEPTH) - 1) ^ ((1u << (Encoder::MAX_DEPTH - level)) - 1);
        min_x &= mask, min_y &= mask, min_z &= mask;

        // Find the physical center by multiplying the encoding with the halfLength
        // to get to the low corner of the cell, and then adding the radii of the cell
        Point center = Point(
            bbox.minX() + min_x * halfLengths[0] * 2, 
            bbox.minY() + min_y * halfLengths[1] * 2, 
            bbox.minZ() + min_z * halfLengths[2] * 2
        ) + precomputedRadii[level];
        
        return {center, precomputedRadii[level]};
    }

    /// @brief Count the leading zeros in a key.
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

    /// @brief Check if a number is a power of 8.
    template <typename Encoder>
    constexpr bool isPowerOf8(typename Encoder::key_t n) {
        typename Encoder::key_t lz = countLeadingZeros<Encoder>(n - 1) - Encoder::UNUSED_BITS;
        return lz % 3 == 0 && !(n & (n - 1));
    }

    /// @brief Get the level in the octree of the given morton code
    template <typename Encoder>
    inline uint32_t getLevel(typename Encoder::key_t range) {
        assert(isPowerOf8<Encoder>(range));
        if(range == Encoder::UPPER_BOUND)
            return typename Encoder::key_t(0);
        return (countLeadingZeros<Encoder>(range - typename Encoder::key_t(1)) - Encoder::UNUSED_BITS) / typename Encoder::key_t(3);
    }

    /// @brief Get the sibling ID of the code at a given level
    template <typename Encoder>
    constexpr uint32_t getSiblingId(typename Encoder::key_t code, uint32_t level) {
        // Shift 3*(21-level) to get the 3 bits corresponding to the level
        return (code >> (typename Encoder::key_t(3) * (Encoder::MAX_DEPTH - level))) & typename Encoder::key_t(7);
    }   

    /**
     * @brief Get the maximum range allowed in a level of the tree.
     * 
     * @example At level 0 the range is the entire 63 bit span, at level 10 the range is 11*3 bit span
     * at level 20 (last to minimum), the range will just be 8 between each node, i.e. the 8 siblings that
     * can be on max level 21 between two nodes at level 20
     * 
     * @param treeLevel The level in the tree.
     * @return The maximum range allowed in the level.
     */
    template <typename Encoder>
    constexpr typename Encoder::key_t nodeRange(uint32_t treeLevel)
    {
        assert(treeLevel < Encoder::MAX_DEPTH);
        uint32_t shifts = Encoder::MAX_DEPTH - treeLevel;

        return 1ul << (typename Encoder::key_t(3) * shifts);
    }

    /// @brief Returns the amount of bits before the placeholder bit.
    template <typename Encoder>
    constexpr uint32_t decodePrefixLength(typename Encoder::key_t code) {
        return 8 * sizeof(typename Encoder::key_t) - 1 - countLeadingZeros<Encoder>(code);
    }

    /**
     * @brief Transforms the SFC (leaf) format into the placeholder bit format, by putting non empty octants into the 
     * beginning of the key and then adding a placeholder bit after them.
     * @example 0 100 000 000 000 ... 000 --> 0 000 000 ... 000 001 100
     * @tparam Encoder The encoder type.
     * @param code The encoded key.
     * @param level The level of the octree on which we are, i.e. the number of octants considered in the key
     * @return The encoded key with the placeholder bit.
     */
    template <typename Encoder>
    constexpr typename Encoder::key_t encodePlaceholderBit(typename Encoder::key_t code, int level) {
        typename Encoder::key_t ret = code >> 3 * (Encoder::MAX_DEPTH - level);
        typename Encoder::key_t placeHolderMask = typename Encoder::key_t(1) << (3*level);

        return placeHolderMask | ret;
    }

    /// @brief Inverse operation to encodePlaceholderBit
    template <typename Encoder>
    constexpr typename Encoder::key_t decodePlaceholderBit(typename Encoder::key_t code) {
        int prefixLength        = decodePrefixLength<Encoder>(code);
        typename Encoder::key_t placeHolderMask = typename Encoder::key_t(1) << prefixLength;
        typename Encoder::key_t ret             = code ^ placeHolderMask;

        return ret << (typename Encoder::key_t(3) * Encoder::MAX_DEPTH - prefixLength);
    }

    /// @brief Find the common prefix between two keys in placeholder format.
    template <typename Encoder>
    constexpr int32_t commonPrefix(typename Encoder::key_t key1, typename Encoder::key_t key2) {
        return int32_t(countLeadingZeros<Encoder>(key1 ^ key2)) - Encoder::UNUSED_BITS;
    }

    /// @brief Extract the octal digit at a given position in a code in the SFC format.
    template <typename Encoder>
    constexpr unsigned octalDigit(typename Encoder::key_t code, uint32_t position) {
        return (code >> (typename Encoder::key_t(3) * (Encoder::MAX_DEPTH - position))) & typename Encoder::key_t(7);
    }
    
    /// @brief Get the ceiling of the logarithm base 8 of a number.
    template <typename Encoder>
    constexpr uint32_t log8ceil(typename Encoder::key_t n) {
        if (n == typename Encoder::key_t(0)) { return 0; }

        uint32_t lz = countLeadingZeros<Encoder>(n - typename Encoder::key_t(1));
        return Encoder::MAX_DEPTH - (lz - Encoder::UNUSED_BITS) / 3;
    }

    /**
     * @brief This function computes the morton encodings of the points and sorts them in
     * the given order
     * 
     * @details The points array is changed after this step
     */
    template <typename Encoder, typename Point_t>
    void sortPoints(std::vector<Point_t> &points, std::vector<typename Encoder::key_t> &codes, const Box &bbox) {
        // Temporal vector of pairs
        std::vector<std::pair<typename Encoder::key_t, Point_t>> encoded_points(points.size());

        // Compute encodings in parallel
        #pragma omp parallel for
            for(size_t i = 0; i < points.size(); i++) {
                typename Encoder::coords_t x, y, z;
                PointEncoding::getAnchorCoords<Encoder>(points[i], bbox, x, y, z);
                encoded_points[i] = std::make_pair(Encoder::encode(x, y, z), points[i]);
            }

        // TODO: implement parallel radix sort
        std::sort(encoded_points.begin(), encoded_points.end(),
            [](const auto& a, const auto& b) {
                return a.first < b.first;  // Compare only the morton codes
        });
        
        // Copy back sorted codes and points in parallel
        codes.resize(points.size());
        #pragma omp parallel for
            for(size_t i = 0; i < points.size(); i++) {
                codes[i] = encoded_points[i].first;
                points[i] = encoded_points[i].second;
            }
    }

    template <typename Encoder, typename Point_t>
    void sortPointsMetadata(std::vector<Point_t> &points, std::vector<typename Encoder::key_t> &codes, std::vector<PointMetadata> &metadata, const Box &bbox) {
        // Temporal vector of pairs
        std::vector<std::tuple<typename Encoder::key_t, Point_t, PointMetadata>> encoded_points(points.size());

        // Compute encodings in parallel
        #pragma omp parallel for
            for(size_t i = 0; i < points.size(); i++) {
                typename Encoder::coords_t x, y, z;
                PointEncoding::getAnchorCoords<Encoder>(points[i], bbox, x, y, z);
                encoded_points[i] = std::make_tuple(Encoder::encode(x, y, z), points[i], metadata[i]);
            }
        
        // TODO: implement parallel radix sort
        std::sort(encoded_points.begin(), encoded_points.end(),
            [](const auto& a, const auto& b) {
                return std::get<0>(a) < std::get<0>(b);  // Compare only the morton codes
        });
        
        // Copy back sorted codes, points, and metadata in parallel
        codes.resize(points.size());
        #pragma omp parallel for
            for(size_t i = 0; i < points.size(); i++) {
                std::tie(codes[i], points[i], metadata[i]) = encoded_points[i];
            }
    }

}