#pragma once

#include "point_encoder.hpp"

namespace PointEncoding {

class HilbertEncoder2D : public PointEncoder {
public:
    /// @brief Max depth for 64-bit keys in 2D is 31 (2 bits per level)
    static constexpr uint32_t MAX_DEPTH = 31;
    static constexpr double EPS = 1.0 / (1ULL << MAX_DEPTH);
    static constexpr key_t UPPER_BOUND = 0x8000000000000000;
    static constexpr uint32_t UNUSED_BITS = 2;

    const uint8_t axis;

    HilbertEncoder2D(uint8_t axis) : axis(axis) {}

    /**
     * @brief Core encoding logic for a 2D Hilbert curve.
     * Uses the Lam and Shapiro iterative approach.
     */
    key_t encode2D(coords_t x, coords_t y) const {
        key_t d = 0;
        // Standard iterative Hilbert: rotate and flip quadrants
        for (int s = (1U << (MAX_DEPTH - 1)); s > 0; s >>= 1) {
            coords_t rx = (x & s) > 0;
            coords_t ry = (y & s) > 0;
            
            // Add the current quadrant's contribution to the key
            // Use (3 * rx) ^ ry for the standard Hilbert Gray code
            d += (key_t)s * s * ((3 * rx) ^ ry);
            
            // Rotate and flip if necessary
            if (ry == 0) {
                if (rx == 1) {
                    x = (1U << MAX_DEPTH) - 1 - x;
                    y = (1U << MAX_DEPTH) - 1 - y;
                }
                std::swap(x, y);
            }
        }
        return d;
    }

    key_t encode(coords_t x, coords_t y, coords_t z) const override {
        switch(axis) {
            case 0:  
                return encode2D(y, z);
            case 1:  
                return encode2D(x, z);
            case 2:  
                return encode2D(x, y);
            default: 
                return encode2D(x, y);
        }
    }

    /**
     * @brief Core decoding logic for a 2D Hilbert curve.
    */
    void decode2D(key_t d, coords_t &x, coords_t &y) const {
        x = y = 0;
        key_t t = d;
        for (int s = 1; s < (1U << MAX_DEPTH); s <<= 1) {
            coords_t rx = 1 & (t / 2);
            coords_t ry = 1 & (t ^ rx);
            
            if (ry == 0) {
                if (rx == 1) {
                    x = s - 1 - x;
                    y = s - 1 - y;
                }
                std::swap(x, y);
            }
            x += s * rx;
            y += s * ry;
            t >>= 2;
        }
    }

    void decode(key_t code, coords_t &x, coords_t &y, coords_t &z) const override {
        x = 0; y = 0; z = 0;
        switch(axis) {
            case 0: 
                decode2D(code, y, z); 
                break;
            case 1: 
                decode2D(code, x, z);
                break;
            case 2: 
                decode2D(code, x, y); 
                break;
            default: 
                decode2D(code, x, y); 
                break;
        }
    }

    key_t encodeFromPoint(const Point& p, const Box &bbox) const override {
        coords_t x, y, z;
        getAnchorCoords(p, bbox, x, y, z);
        return encode(x, y, z);
    }

    void encodeVectorized(const uint32_t *x, const uint32_t *y, const uint32_t *z, std::vector<key_t> &keys, size_t i) const override {
        return; 
    }

    // Getters
    inline uint32_t maxDepth() const override { return MAX_DEPTH; }
    inline double eps() const override { return EPS; }
    inline key_t upperBound() const override { return UPPER_BOUND; }
    inline uint32_t unusedBits() const override { return UNUSED_BITS; }
    inline bool is3D() const override { return false; };

    inline EncoderType getEncoder() const override {
        switch(axis) {
            case 0: 
                return EncoderType::HILBERT_ENCODER_2D_X;
            case 1: 
                return EncoderType::HILBERT_ENCODER_2D_Y;
            case 2: 
                return EncoderType::HILBERT_ENCODER_2D_Z;
            default: 
                return EncoderType::HILBERT_ENCODER_2D_Z;
        }
    }

    inline std::string getEncoderName() const override { 
        return "HilbertEncoder2D Axis " + std::string(1, static_cast<char>('X' + axis)); 
    }
    
    inline std::string getShortEncoderName() const override { 
        return "hilb_2d_axis_" + std::string(1, static_cast<char>('X' + axis)); 
    }
};

} // namespace PointEncoding