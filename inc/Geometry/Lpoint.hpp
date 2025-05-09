#pragma once

#include "point.hpp"
#include <array>
#include <cstdint>

// Forward declaration
class Region;

// Follows Point Data Record Format 2 from the LAS standard
// https://www.asprs.org/wp-content/uploads/2010/12/LAS_1_4_r13.pdf
struct alignas(32) Lpoint : public Point {
protected:
    double     I_{};                // Intensity
    uint16_t  psId_{};             // Point Source ID
    uint16_t  r_{};                // Red
    uint16_t  g_{};                // Green
    uint16_t  b_{};                // Blue
    uint8_t   ud_{};               // User Data
    uint8_t   classification_{};  // Classification
    int8_t    sar_{};              // Scan Angle Rank

    union PackedFlags {
        uint8_t packed = 0;
        struct {
            uint8_t rn_   : 3;  // Return Number (bits 0–2)
            uint8_t nor_  : 3;  // Number of Returns (bits 3–5)
            uint8_t dir_  : 1;  // Scan Direction Flag (bit 6)
            uint8_t edge_ : 1;  // Edge of Flight Line (bit 7)
        } fields;

        PackedFlags() = default;
    } flags;

public:
    // Constructors
    Lpoint() = default;
    Lpoint(size_t id, double x, double y, double z) : Point(id, x, y, z) {}
    Lpoint(double x, double y) : Point(x, y) {}
    Lpoint(double x, double y, double z) : Point(x, y, z) {}
    explicit Lpoint(Point p) : Point(p.getX(), p.getY(), p.getZ()) {}

    // ISPRS-style input
    Lpoint(size_t id, double x, double y, double z, double I, uint8_t rn, uint8_t nor, uint8_t classification)
        : Point(id, x, y, z), I_(I), classification_(classification) {
        setPackedFields(rn, nor, false, false);
    }

    // Standard classified cloud
    Lpoint(size_t id, double x, double y, double z, double I, uint8_t rn, uint8_t nor, bool dir,
           bool edge, uint8_t classification)
        : Point(id, x, y, z), I_(I), classification_(classification) {
        setPackedFields(rn, nor, dir, edge);
    }

    // Classified cloud with RGB
    Lpoint(size_t id, double x, double y, double z, double I, uint8_t rn, uint8_t nor, bool dir,
           bool edge, uint8_t classification, uint16_t r, uint16_t g, uint16_t b)
        : Point(id, x, y, z), I_(I), classification_(classification), r_(r), g_(g), b_(b) {
        setPackedFields(rn, nor, dir, edge);
    }

    // Full record format (including extra LAS fields)
    Lpoint(size_t id, double x, double y, double z, double I, uint8_t rn, uint8_t nor, bool dir,
           bool edge, uint8_t classification, int8_t sar, uint8_t ud, uint16_t psId,
           uint16_t r, uint16_t g, uint16_t b)
        : Point(id, x, y, z), I_(I), classification_(classification),
          sar_(sar), ud_(ud), psId_(psId), r_(r), g_(g), b_(b) {
        setPackedFields(rn, nor, dir, edge);
    }

    // Setters and field helpers
    inline void setI(double I) { I_ = I; }

    inline void setPackedFields(uint8_t rn, uint8_t nor, bool dir, bool edge) {
        setRN(rn);
        setNOR(nor);
        setDir(dir);
        setEdge(edge);
    }

    inline void setRN(uint8_t rn) { flags.fields.rn_ = rn & 0b111; }
    inline void setNOR(uint8_t nor) { flags.fields.nor_ = nor & 0b111; }
    inline void setDir(bool dir) { flags.fields.dir_ = dir & 0b1; }
    inline void setEdge(bool edge) { flags.fields.edge_ = edge & 0b1; }

    // Accessors
    inline double          getI() const { return I_; }
    inline uint8_t        getClass() const { return classification_; }
    inline void           setClass(uint8_t classification) { classification_ = classification; }
    inline uint8_t        rn() const { return flags.fields.rn_; }
    inline uint8_t        nor() const { return flags.fields.nor_; }
    inline uint8_t        dir() const { return flags.fields.dir_; }
    inline uint8_t        edge() const { return flags.fields.edge_; }
    inline uint16_t       getR() const { return r_; }
    inline void           setR(uint16_t r) { r_ = r; }
    inline uint16_t       getG() const { return g_; }
    inline void           setG(uint16_t g) { g_ = g; }
    inline uint16_t       getB() const { return b_; }
    inline void           setB(uint16_t b) { b_ = b; }

    // Stub (can be implemented later)
    void setEigenvalues(const std::vector<double>& eigenvalues) {}
};
