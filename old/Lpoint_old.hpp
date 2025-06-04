//
// Created by miguelyermo on 25/3/21.
//

#ifndef RULE_BASED_CLASSIFIER_CPP_LPOINT_HPP
#define RULE_BASED_CLASSIFIER_CPP_LPOINT_HPP

#include "point.hpp"
#include <array>

class Region; // Region forward declaration

class Lpoint : public Point
{
	protected:
	double         I_{};              // Intensity
	unsigned short rn_{};             // Return Number
	unsigned short nor_{};            // Number of Returns (given pulse)
	unsigned short dir_{};            // Scan Direction Flag
	unsigned short edge_{};           // Edge of Flight Line
	unsigned short classification_{}; // Classification of the point
	char           sar_{};            // Scan Angle Rank
	unsigned char  ud_{};             // User Data
	unsigned short psId_{};           // Point Source ID

	unsigned int r_{}; // Red channel (0-65535)
	unsigned int g_{}; // Green channel (0-65535)
	unsigned int b_{}; // Blue channel (0-65535)

	Vector normal_{};


	protected:

	public:
	// Default constructor
	Lpoint() : Point(){};
	Lpoint(size_t id, double x, double y, double z) : Point(id, x, y, z){};
	Lpoint(double x, double y) : Point(x, y){};
	Lpoint(double x, double y, double z) : Point(x, y, z){};
	explicit Lpoint(Point p) : Point(p.getX(), p.getY(), p.getZ()){};


	// Reading points ISPRS format
	Lpoint(size_t id, double x, double y, double z, double I, double rn, double nor, unsigned int classification) :
	  Point(id, x, y, z), I_(I), rn_(rn), nor_(nor), classification_(classification){};

	// Reading standard classified cloud
	Lpoint(size_t id, double x, double y, double z, double I, unsigned short rn, unsigned short nor, unsigned char dir,
	       unsigned char edge, unsigned short classification) :
	  Point(id, x, y, z),
	  I_(I), rn_(rn), nor_(nor), dir_(dir), edge_(edge), classification_(classification){};

	// Reading classified cloud with RGB
	Lpoint(size_t id, double x, double y, double z, double I, unsigned short rn, unsigned short nor, unsigned char dir,
	       unsigned char edge, unsigned short classification, unsigned int r, unsigned int g, unsigned int b) :
	  Point(id, x, y, z),
	  I_(I), rn_(rn), nor_(nor), dir_(dir), edge_(edge), classification_(classification), r_(r), g_(g), b_(b){};


	// Reading Point Data Record Format 2 (Babcock / Coremain request)
	Lpoint(size_t id, double x, double y, double z, double I, unsigned short rn, unsigned short nor, unsigned char dir,
	       unsigned char edge, unsigned short classification, char sar, unsigned char ud, unsigned short psId,
	       unsigned int r, unsigned int g, unsigned int b) :
	  Point(id, x, y, z),
	  I_(I), rn_(rn), nor_(nor), dir_(dir), edge_(edge), classification_(classification), sar_(sar), ud_(ud), psId_(psId),
	  r_(r), g_(g), b_(b){};

	// Setters and getters
	inline double I() const { return I_; }
	inline void   setI(double I) { I_ = I; }

	// TODO: delete gId references
	inline unsigned short        getClass() const { return classification_; }
	inline void                  setClass(unsigned short classification) { classification_ = classification; }
	inline unsigned short        rn() const { return rn_; }
	inline unsigned short        nor() const { return nor_; }
	inline unsigned short        dir() const { return dir_; }
	inline unsigned short        edge() const { return edge_; }
	inline unsigned int          getR() const { return r_; }
	inline void                  setR(unsigned int r) { r_ = r; }
	inline unsigned int          getG() const { return g_; }
	inline void                  setG(unsigned int g) { g_ = g; }
	inline unsigned int          getB() const { return b_; }
	inline void                  setB(unsigned int b) { b_ = b; }
	const Vector&                getNormal() const { return normal_; }
	void                         setNormal(const Vector& normal) { normal_ = normal; }
	void                         setEigenvalues(const std::vector<double>& eigenvalues) {}
	double                       getI() const { return I_; }

};

#endif //RULE_BASED_CLASSIFIER_CPP_LPOINT_HPP
