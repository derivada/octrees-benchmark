#pragma once

#include "Geometry/point.hpp"
#include "Geometry/Box.hpp"
#include "common.hpp"
#include <bitset>

namespace PointEncoding {
    /**
    * @struct HilbertEncoder3D
    * 
    * @brief Implements the Hilbert space filling curve to encode integer point coordinates. The keys are 64 bit unsigned integers.
    * 
    * @date 11/12/2024
    * 
    * @cite https://doi.org/10.1016/j.newast.2016.10.007 Appendix A contains the algorithm used here
    * @cite https://www.semanticscholar.org/paper/A-class-of-fast-algorithms-for-the-Peano-Hilbert-Lam-Shapiro/2e2987a5070f79f8a94f110a3a2862cc98c94de3 
    * contains a great general explanation
    */
    struct HilbertEncoder3D {
        /// @brief The type for the output keys
        using key_t = uint_fast64_t;
        /// @brief the type for the input coordinates
        using coords_t = uint_fast32_t;
                
        /// @brief The maximum depth that this encoding allows (in Hilbert 64 bit integers, we need 3 bits for each level, so 21)
        static constexpr uint32_t MAX_DEPTH = 21;

        /// @brief The minimum unit of length of the encoded coordinates
        static constexpr double EPS = 1.0f / (1 << MAX_DEPTH);

        /// @brief The minimum (strict) upper bound for every Hilbert code. Equal to the unused bit followed by 63 zeros.
        static constexpr key_t UPPER_BOUND = 0x8000000000000000;

        /// @brief The amount of bits that are not used, in Hilbert encodings this is the MSB of the key
        static constexpr uint32_t UNUSED_BITS = 1;

        /// @brief A constant array to map adequately rotated x, y, z coordinates to their corresponding octant 
        static constexpr coords_t mortonToHilbert[8] = {0, 1, 3, 2, 7, 6, 4, 5};
        
        /**
         * @brief Encodes the given integer coordinates in the range [0,2^MAX_DEPTH]x[0,2^MAX_DEPTH]x[0,2^MAX_DEPTH] into their Hilbert key
         * The algorithm is described in the citations above but consists on something similar to the intertwinement of bits of Morton codes 
         * but with some extra rotations and reflections in each step.
         */
        static inline key_t encode(coords_t x, coords_t y, coords_t z) {
            key_t key = 0;
            for(int level = MAX_DEPTH - 1; level >= 0; level--) {
                // Find octant and append to the key (same as Morton codes)
                const coords_t xi = (x >> level) & 1u;
                const coords_t yi = (y >> level) & 1u;
                const coords_t zi = (z >> level) & 1u;
                const coords_t octant = (xi << 2) | (yi << 1) | zi;
                key <<= 3;
                key |= mortonToHilbert[octant];
                
                // Turn x, y, z (Karnaugh mapped operations, check citation and Lam and Shapiro paper detailing 2D case 
                // for understanding how this works)
                x ^= -(xi & ((!yi) | zi));
                y ^= -((xi & (yi | zi)) | (yi & (!zi)));
                z ^= -((xi & (!yi) & (!zi)) | (yi & (!zi)));

                if (zi) {
                    // Cylic anticlockwise rotation x, y, z -> y, z, x
                    coords_t temp = x;
                    x = y, y = z, z = temp;
                } else if (!yi) {
                    // Swap x and z
                    coords_t temp = x;
                    x = z, z = temp;
                }
            }
            return key;
        }

        /// @brief Decodes the given key and puts the coordinates into x, y, z
        static inline void decode(key_t code, coords_t &x, coords_t &y, coords_t &z) {
            // Initialize the coords values
            x = 0, y = 0, z = 0;
            for(int level = 0; level < MAX_DEPTH; level++) {
                // Extract the octant from the key and put the bits into xi, yi and zi
                const coords_t octant   = (code >> (3 * level)) & 7u;
                const coords_t xi = octant >> 2u;
                const coords_t yi = (octant >> 1u) & 1u;
                const coords_t zi = octant & 1u;

                if(yi ^ zi) {
                    // Cylic clockwise rotation x, y, z -> z, x, y
                    coords_t temp = x;
                    x = z, z = y, y = temp;
                } else if((!xi & !yi & !zi) || (xi & yi & zi)) {
                    // Swap x and z
                    coords_t temp = x;
                    x = z, z = temp;
                }

                // Turn x, y, z (Karnaugh mapped operations, check citation and Lam and Shapiro paper detailing 2D case 
                // for understanding how this works)
                const coords_t mask = (1u << level) - 1u;
                x ^= mask & (-(xi & (yi | zi)));
                y ^= mask & (-((xi & ((!yi) | (!zi))) | ((!xi) & yi & zi)));
                z ^= mask & (-((xi & (!yi) & (!zi)) | (yi & zi)));

                // Append the new bit to the position
                x |= (xi << level);
                y |= ((xi ^ yi) << level);
                z |= ((yi ^ zi) << level);
            }
            return;
        }
    };
};