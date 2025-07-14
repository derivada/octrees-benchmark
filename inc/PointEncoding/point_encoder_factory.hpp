#pragma once
#include "point_encoder.hpp"
#include "hilbert_encoder_3d.hpp"
#include "morton_encoder_3d.hpp"
#include "no_encoding.hpp"
#include "main_options.hpp"

namespace PointEncoding {
   
    inline PointEncoder& getEncoder(EncoderType type) {
        static HilbertEncoder3D hilbertEncoder;
        static MortonEncoder3D mortonEncoder;
        static NoEncoding noEncoding;
        
        switch (type) {
            case EncoderType::HILBERT_ENCODER_3D:
                return hilbertEncoder;
            case EncoderType::MORTON_ENCODER_3D:
                return mortonEncoder;
            case EncoderType::NO_ENCODING:
            default:
                return noEncoding;
        }
    }
} // namespace PointEncoding