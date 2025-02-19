#pragma once

#include "Geometry/Lpoint.hpp"
#include "Geometry/Lpoint64.hpp"
#include "Geometry/point.hpp"
#include "PointEncoding/common.hpp"
#include "PointEncoding/morton_encoder.hpp"
#include "PointEncoding/hilbert_encoder.hpp"

/**
 * This file simply includes some class names for the templated classes.
 * Originally this also included Octree and LinearOctree, but that gave template deduction issues
 * when I tried using type_names.hpp inside one of the octree.hpp classes, since then a cyclic
 * dependence would be created.
 * 
 * For getting the string with the runtime type of a generic Octree_t, just do an constexpr if-else like in
 * octree_benchmark.hpp
 */
template <typename T>
std::string getPointName();

template <>
inline std::string getPointName<Lpoint64>() { return "Lpoint64"; }

template <>
inline std::string getPointName<Lpoint>() { return "Lpoint"; }

template <>
inline std::string getPointName<Point>() { return "Point"; }

namespace PointEncoding {
    template <typename Encoder_t>
    std::string getEncoderName();

    template <>
    inline std::string getEncoderName<PointEncoding::NoEncoder>() { return "Unencoded"; }
    
    template <>
    inline std::string getEncoderName<PointEncoding::MortonEncoder3D>() { return "MortonEncoder3D"; }

    template <>
    inline std::string getEncoderName<PointEncoding::HilbertEncoder3D>() { return "HilbertEncoder3D"; }

    template <typename Encoder_t>
    std::string getShortEncoderName(); 
    template <>
    inline std::string getShortEncoderName<PointEncoding::NoEncoder>() { return "nocode"; }
    
    template <>
    inline std::string getShortEncoderName<PointEncoding::MortonEncoder3D>() { return "mort3d"; }

    template <>
    inline std::string getShortEncoderName<PointEncoding::HilbertEncoder3D>() { return "hilb3d"; }
}