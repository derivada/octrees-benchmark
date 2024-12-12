#pragma once

#include "libmorton/morton.h"
#include "Geometry/point.hpp"
#include "Geometry/Box.hpp"
#include "common.hpp"

namespace PointEncoding {
    /**
    * @struct HilbertEncoder64
    * 
    * @brief
    * 
    * @cit
    * 
    * @authors Pablo Díaz Viñambres 
    * 
    * @date 11/12/2024
    * 
    */
    struct HilbertEncoder64 {
        using key_t = uint_fast64_t;
        using coords_t = uint_fast32_t;

        static constexpr const char* NAME = "HilbertEncoder64";
        
        /// @brief The maximum depth that this encoding allows (in Morton 64 bit integers, we need 3 bits for each level, so 21)
        static constexpr unsigned MAX_DEPTH = 21;

        /// @brief The minimum unit of length of the encoded coordinates
        static constexpr double EPS = 1.0f / (1 << MAX_DEPTH);

        /// @brief The minimum (strict) upper bound for every Morton code. Equal to the unused bit followed by 63 zeros.
        static constexpr key_t UPPER_BOUND = 0x8000000000000000;

        /// @brief The amount of bits that are not used, in Morton encodings this is the MSB of the key
        static constexpr uint32_t UNUSED_BITS = 1;

        // This methods should not be called from the outside, as they use geometric information computed here
        // (radii and center) of the point cloud
        static inline key_t encode(const Point& p, const Box &bbox) {
            // TODO
            return key_t(0);
        }

        static inline void decode(key_t code, coords_t &x, coords_t &y, coords_t &z) {
            // TODO
            return;
        }
    };
};