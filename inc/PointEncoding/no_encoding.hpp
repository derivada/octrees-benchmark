#pragma once

#include "libmorton/morton.h"
#include "point_encoder.hpp"

namespace PointEncoding {
class NoEncoding : public PointEncoder {
    static constexpr unsigned MAX_DEPTH = 21;
    static constexpr double EPS = 1.0f / (1 << MAX_DEPTH);
    static constexpr key_t UPPER_BOUND = 0x8000000000000000;
    static constexpr uint32_t UNUSED_BITS = 1;

    inline key_t encode(coords_t x, coords_t y, coords_t z) const override {
        return 0;
    }

    /// @brief Decodes the given Morton key and puts the coordinates into x, y, z
    inline void decode(key_t code, coords_t &x, coords_t &y, coords_t &z) const override {
        x = 0, y = 0, z = 0;
    }

    key_t encodeFromPoint(const Point& p, const Box &bbox) const override {
        return 0;
    }

    // Getters
    inline uint32_t maxDepth() const override { return MAX_DEPTH; }
    inline double eps() const override { return EPS; }
    inline key_t upperBound() const override { return UPPER_BOUND; }
    inline uint32_t unusedBits() const override { return UNUSED_BITS; }
    inline std::string getEncoderName() const { return "NoEncoding"; };
    inline std::string getShortEncoderName() const { return "none"; };
};
};