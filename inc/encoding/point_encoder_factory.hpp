#pragma once

#include "hilbert_encoder_2d.hpp"
#include "hilbert_encoder_3d.hpp"
#include "main_options.hpp"
#include "morton_encoder_2d.hpp"
#include "morton_encoder_3d.hpp"
#include "no_encoding.hpp"
#include "point_encoder.hpp"

namespace PointEncoding {
    inline PointEncoder& getEncoder(EncoderType type) {
        static MortonEncoder3D morton3D;
        static HilbertEncoder3D hilbert3D;
        static MortonEncoder2D morton2DX(0);
        static MortonEncoder2D morton2DY(1);
        static MortonEncoder2D morton2DZ(2);
        static HilbertEncoder2D hilbert2DX(0);
        static HilbertEncoder2D hilbert2DY(1);
        static HilbertEncoder2D hilbert2DZ(2);
        static NoEncoding noEncoding;
        
        switch (type) {
            case EncoderType::MORTON_ENCODER_3D:
                return morton3D;
            case EncoderType::HILBERT_ENCODER_3D:
                return hilbert3D;
            case EncoderType::MORTON_ENCODER_2D_X:
                return morton2DX;
            case EncoderType::MORTON_ENCODER_2D_Y:
                return morton2DY;
            case EncoderType::MORTON_ENCODER_2D_Z:
                return morton2DZ;
            case EncoderType::HILBERT_ENCODER_2D_X:
                return hilbert2DX;
            case EncoderType::HILBERT_ENCODER_2D_Y:
                return hilbert2DY;
            case EncoderType::HILBERT_ENCODER_2D_Z:
                return hilbert2DZ;
            case EncoderType::NO_ENCODING:
            default:
                return noEncoding;
        }
    }
}