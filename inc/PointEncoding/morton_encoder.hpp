#pragma once

#include "libmorton/morton.h"
#include "Geometry/point.hpp"
#include "Geometry/Box.hpp"
#include "common.hpp"

namespace PointEncoding {
    /**
    * @struct MortonEncoder3D
    * 
    * @brief A front-end for libmorton for getting adequate encodings needed on the linear octree. The keys are 64 bit unsigned integers.
    * 
    * @cite Jeroen Baert. Libmorton: C++ Morton Encoding/Decoding Library. https://github.com/Forceflow/libmorton/tree/main
    * 
    * @authors Pablo Díaz Viñambres 
    * 
    * @date 16/11/2024
    * 
    */
    struct MortonEncoder3D {
        using key_t = uint_fast64_t;
        using coords_t = uint_fast32_t;
        
        /// @brief The maximum depth that this encoding allows (in Morton 64 bit integers, we need 3 bits for each level, so 21)
        static constexpr unsigned MAX_DEPTH = 21;

        /// @brief The minimum unit of length of the encoded coordinates
        static constexpr double EPS = 1.0f / (1 << MAX_DEPTH);

        /// @brief The minimum (strict) upper bound for every Morton code. Equal to the unused bit followed by 63 zeros.
        static constexpr key_t UPPER_BOUND = 0x8000000000000000;

        /// @brief The amount of bits that are not used, in Morton encodings this is the MSB of the key
        static constexpr uint32_t UNUSED_BITS = 1;

        
        static inline key_t encode(coords_t x, coords_t y, coords_t z) {
            return libmorton::morton3D_64_encode(x, y, z);
        }

        static inline void decode(key_t code, coords_t &x, coords_t &y, coords_t &z) {
            libmorton::morton3D_64_decode(code, x, y, z);
        }
    };
};