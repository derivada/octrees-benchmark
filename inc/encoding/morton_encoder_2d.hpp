#pragma once

#include "libmorton/morton.h"
#include "point_encoder.hpp"

namespace PointEncoding {
class MortonEncoder2D : public PointEncoder {
    /// @brief The maximum depth that this encoding allows (in Morton 64 bit integers, we need 2 bits for each level, so 31)
    static constexpr unsigned MAX_DEPTH = 31;

    /// @brief The minimum unit of length of the encoded coordinates
    static constexpr double EPS = 1.0f / (1 << MAX_DEPTH);

    /// @brief The minimum (strict) upper bound for every Morton code. Equal to the unused bit followed by 62 zeros.
    static constexpr key_t UPPER_BOUND = 0x4000000000000000;

    /// @brief The amount of bits that are not used, in Morton encodings this is the MSB of the key
    static constexpr uint32_t UNUSED_BITS = 2;
    
    const uint8_t axis = 0;



    /// @brief Encodes the given integer coordinates in the range [0,2^MAX_DEPTH]x[0,2^MAX_DEPTH]x[0,2^MAX_DEPTH] into their Morton key
    inline key_t encode(coords_t x, coords_t y, coords_t z) const override {
        switch(axis) {
            case 0:
                return libmorton::morton2D_64_encode(y, z);
            break;
            case 1:
                return libmorton::morton2D_64_encode(x, z);
            break;
            case 2:
                return libmorton::morton2D_64_encode(x, y);    
            break;
            default:
                return libmorton::morton2D_64_encode(x, y);
            break;
        }   
    }

    void encodeVectorized(const uint32_t *x, const uint32_t *y, const uint32_t *z, std::vector<key_t> &keys, size_t i) const override
    {
        return;
    }

    /// @brief Decodes the given Morton key and puts the coordinates into x, y, z
    inline void decode(key_t code, coords_t &x, coords_t &y, coords_t &z) const override {
        switch(axis) {
            case 0:
                return libmorton::morton2D_64_decode(code, y, z);
            break;
            case 1:
                return libmorton::morton2D_64_decode(code, x, z);
            break;
            case 2:
                return libmorton::morton2D_64_decode(code, x, y);    
            break;
            default:
                return libmorton::morton2D_64_decode(code, x, y);
            break;
        }
    }

    key_t encodeFromPoint(const Point& p, const Box &bbox) const override {
        coords_t x, y, z;
        getAnchorCoords(p, bbox, x, y, z);
		return encode(x, y, z);
    }

    // Getters
    inline uint32_t maxDepth() const override { return MAX_DEPTH; }
    inline double eps() const override { return EPS; }
    inline key_t upperBound() const override { return UPPER_BOUND; }
    inline uint32_t unusedBits() const override { return UNUSED_BITS; }
    inline EncoderType getEncoder() const override { 
        switch(axis) {
            case 0:
                return EncoderType::MORTON_ENCODER_2D_X;
            case 1:
                return EncoderType::MORTON_ENCODER_2D_Y;
            case 2:
                return EncoderType::MORTON_ENCODER_2D_Z;
            default:
                return EncoderType::MORTON_ENCODER_2D_Z;
        }
    };
    inline std::string getEncoderName() const override { return "MortonEncoder2D Axis " + std::string(1, static_cast<char>('X' + axis)); };
    inline std::string getShortEncoderName() const override { return "mort_2d_axis_" + std::string(1, static_cast<char>('X' + axis)); };
    inline bool is3D() const override { return false; };

    public:
    MortonEncoder2D(uint8_t axis): axis(axis){
    }
};
};