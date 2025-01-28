#pragma once

#include "point.hpp"
#include <array>
#include <cstdint>

class Region; // Region forward declaration

class PointMetadata
{
	protected:
	double         I_{};             		// Intensity

	union PackedFlags {
		uint8_t packed = 0;
		struct {
			uint8_t rn_ : 3;   				// Return Number, bits 0-2
            uint8_t nor_ : 3;    			// Number of Returns (given pulse), bits 3-5
            uint8_t dir_ : 1;   			// Scan Direction Flag, bit 6
            uint8_t edge_ : 1;  			// Edge of Flight Line, bit 7
		} fields;
		PackedFlags() : packed(0) {}
	} flags;
	
	unsigned char classification_{}; // Classification of the point
	signed char           sar_{};            // Scan Angle Rank
	unsigned char  ud_{};             // User Data
	unsigned short psId_{};           // Point Source ID
	
	// this fields should be 4 bytes according to standard https://www.asprs.org/wp-content/uploads/2019/07/LAS_1_4_r15.pdf
	// leaving them at 2 bytes is benefitial since the struct is now 64 bytes though
	unsigned int r_{}; // Red channel (0-65535)
	unsigned int g_{}; // Green channel (0-65535)
	unsigned int b_{}; // Blue channel (0-65535)

	protected:

	public:
	// Default constructor
	PointMetadata() = default;

	// Reading points ISPRS format
	PointMetadata(double I, uint8_t rn, uint8_t nor, unsigned int classification) :
	    I_(I), classification_(classification) {
		setPackedFields(rn, nor, 0, 0);
	  };

	// Reading standard classified cloud
	PointMetadata(double I, uint8_t rn, uint8_t nor, bool dir,
	       bool edge, unsigned short classification) :
	  I_(I), classification_(classification) {
		setPackedFields(rn, nor, dir, edge);
	  };

	// Reading classified cloud with RGB
	PointMetadata(double I, uint8_t rn, uint8_t nor, bool dir,
	       bool edge, unsigned short classification, unsigned int r, unsigned int g, unsigned int b) :
	  I_(I), classification_(classification), r_(r), g_(g), b_(b) {
		setPackedFields(rn, nor, dir, edge);
	  };


	// Reading Point Data Record Format 2 (Babcock / Coremain request)
	PointMetadata( double I, uint8_t rn, uint8_t nor, bool dir,
	       bool edge, unsigned short classification, char sar, unsigned char ud, unsigned short psId,
	       unsigned int r, unsigned int g, unsigned int b) :
	  I_(I), classification_(classification), 
	  sar_(sar), ud_(ud), psId_(psId), r_(r), g_(g), b_(b) {
		setPackedFields(rn, nor, dir, edge);
	  };

	// Setters and getters
	inline void   setI(double I) { I_ = I; }
	inline void   setPackedFields(uint8_t rn, uint8_t nor, bool dir, bool edge) {
		setRN(rn);
		setNOR(nor);
		setDir(dir);
		setEdge(edge);
	}
	inline void   setRN(uint8_t rn) { flags.fields.rn_ = rn & 0b111; }
	inline void   setNOR(uint8_t nor) { flags.fields.nor_ = nor & 0b111; }
	inline void   setDir(bool dir) { flags.fields.dir_ = dir & 0b1; }
	inline void   setEdge(bool edge) { flags.fields.edge_ = edge & 0b1; }


	// TODO: delete gId references
	inline double 				 I() const { return I_; }
	inline unsigned short        getClass() const { return classification_; }
	inline void                  setClass(unsigned short classification) { classification_ = classification; }
	inline unsigned short        rn() const { return flags.fields.rn_; }
	inline unsigned short        nor() const { return flags.fields.nor_; }
	inline unsigned short        dir() const { return flags.fields.dir_; }
	inline unsigned short        edge() const { return flags.fields.edge_; }
	inline unsigned int          getR() const { return r_; }
	inline void                  setR(unsigned int r) { r_ = r; }
	inline unsigned int          getG() const { return g_; }
	inline void                  setG(unsigned int g) { g_ = g; }
	inline unsigned int          getB() const { return b_; }
	inline void                  setB(unsigned int b) { b_ = b; }
	// const Vector&                getNormal() const { return normal_; }
	// void                         setNormal(const Vector& normal) { normal_ = normal; }
	void                         setEigenvalues(const std::vector<double>& eigenvalues) {}
	double                       getI() const { return I_; }

};
