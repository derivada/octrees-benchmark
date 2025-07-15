//
// Created by miguelyermo on 1/3/20.
//

#pragma once

inline constexpr float SENSEPSILON = 0.001; // Sensible epsilon

#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

using Vector = class Point; // Point and Vector are exactly the same entity, but this is done
                            // to preserve mathematical correctness

struct alignas(32) Point
{
	protected:
	double       x_{};  // X Coordinate
	double       y_{};  // Y Coordinate
	double       z_{};  // Z Coordinate
	size_t id_{}; // Id of the point (in order of reading)

	public:
	Point() = default;
	explicit Point(size_t id) : id_(id) {}

	// 2D Geometric constructor ( Z = 0.0 )
	Point(double x, double y) : x_(x), y_(y) {}

	// 3D Geometric constructor
	Point(double x, double y, double z) : x_(x), y_(y), z_(z) {}

	// 3D Geometric constructor with ID
	Point(size_t id, double x, double y, double z) : id_(id), x_(x), y_(y), z_(z) {}


	// Reading points ISPRS format
	Point(size_t id, double x, double y, double z, double I, double rn, double nor, unsigned int classification) :
	  Point(id, x, y, z){};

	// Reading standard classified cloud
	Point(size_t id, double x, double y, double z, double I, unsigned short rn, unsigned short nor, unsigned char dir,
	       unsigned char edge, unsigned short classification) :
	  Point(id, x, y, z){};

	// Reading classified cloud with RGB
	Point(size_t id, double x, double y, double z, double I, unsigned short rn, unsigned short nor, unsigned char dir,
	       unsigned char edge, unsigned short classification, unsigned int r, unsigned int g, unsigned int b) :
	  Point(id, x, y, z){};


	// Reading Point Data Record Format 2 (Babcock / Coremain request)
	Point(size_t id, double x, double y, double z, double I, unsigned short rn, unsigned short nor, unsigned char dir,
	       unsigned char edge, unsigned short classification, char sar, unsigned char ud, unsigned short psId,
	       unsigned int r, unsigned int g, unsigned int b) :
	  Point(id, x, y, z){};


	// Point Methods
	/* 2D distance between two points */
	[[nodiscard]] inline double distance2D(const Point& p) const
	{
		double diffX = x_ - p.getX();
		double diffY = y_ - p.getY();

		return sqrt(diffX * diffX + diffY * diffY);
	}

	/* 2D distance between two points as coded in C version */
	[[nodiscard]] inline double distance2D100Rounded(const Point& p) const
	{
		double diffX = std::round(x_ * 100) - std::round(p.getX() * 100);
		double diffY = std::round(y_ * 100) - std::round(p.getY() * 100);

		return sqrt(diffX * diffX + diffY * diffY);
	}

	[[nodiscard]] inline double distance2Dsquared(const Point& p) const
	{
		const double diffX = x_ - p.getX();
		const double diffY = y_ - p.getY();

		return diffX * diffX + diffY * diffY;
	}

	/* 3D distance between two points */
	[[nodiscard]] inline double distance3D(const Point& p) const
	{
		double diffX = x_ - p.getX();
		double diffY = y_ - p.getY();
		double diffZ = z_ - p.getZ();

		return sqrt(diffX * diffX + diffY * diffY + diffZ * diffZ);
	}

	[[nodiscard]] inline double distanceToLine(const Point& l1, const Point& l2) const
	/**
	 * Computes the distance of this point to the line defined by two points
	 * @param l1 Origin of the line
	 * @param l2 End of the line
	 * @return
	 */
	{
		double y2_y1 = l2.getY() - l1.getY();
		double x2_x1 = l2.getX() - l1.getX();

		return fabs(y2_y1 * x_ - x2_x1 * y_ + l2.getX() * l1.getY() - l2.getY() * l1.getX()) /
		       sqrt(y2_y1 * y2_y1 + x2_x1 * x2_x1);
	}

	[[nodiscard]] inline double distanceToLine2(const Point& p, const Vector& d) const
	/**
	 * Computes the distance from this point to the line passing through p with direction d (unitary).
	 *
	 * @param p starting point of the line
	 * @param d unitary direction of the line
	 * @return distance to line
	 */
	{
		const Vector pt = *this - p;
		return (pt - d * (pt.dotProduct(d))).norm3D();
	}

	[[nodiscard]] inline Point getDest(Vector& v, double distance) const
	/**
	 * Get the destination point of moving a point form a vector a given distance
	 */
	{
		v.normalize2D();
		return { x_ + v.getX() * distance, y_ + v.getY() * distance, z_ };
	}

	[[nodiscard]] inline double getMaxCoordinate() const {
		return std::max(x_, std::max(y_, z_));
	}
	[[nodiscard]] inline double getMinCoordinate() const {
		return std::min(x_, std::min(y_, z_));
	}

	// Vector Methods
	/* Gets the perpendicular vector (XY plane) */
	[[nodiscard]] inline Vector perpenVector() const { return { -y_, x_, z_ }; }

	[[nodiscard]] inline bool isZero() const
	{
		return std::abs(x_) < SENSEPSILON && std::abs(y_) < SENSEPSILON && std::abs(z_) < SENSEPSILON;
	}

	/* Returns the 2D norm of the vector */
	[[nodiscard]] inline double norm2D() const { return std::sqrt(x_ * x_ + y_ * y_); }

	/* Returns the 3D norm of the vector */
	[[nodiscard]] inline double norm3D() const { return std::sqrt(x_ * x_ + y_ * y_ + z_ * z_); }

	/* Normalization of the vector in two dimensions */
	void normalize2D()
	{
		const double norm = norm2D();
		if (norm < SENSEPSILON) { std::cerr << "Warning: \"normalize2D\" called on a zero-ish vector\n"; }
		*this /= norm;
	}

	/* Normalization of the vector in three dimensions */
	void normalize3D()
	{
		const double norm = norm3D();
		if (norm2D() < SENSEPSILON) { std::cerr << "Warning: \"normalize3D\" called on a zero-ish vector\n"; }
		*this /= norm;
	}

	/* Project onto XY plane */
	void projectXY() { this->setZ(0.0); }

	/* Dot product in 2D */
	[[nodiscard]] inline double dotProduct2D(const Vector& vec) const { return x_ * vec.getX() + y_ * vec.getY(); }

	/* Dot product in 3D */
	[[nodiscard]] inline double dotProduct(const Vector& vec) const
	{
		return x_ * vec.getX() + y_ * vec.getY() + z_ * vec.getZ();
	}

	/* Cross product */
	[[nodiscard]] inline Vector crossProduct(const Vector& vec) const
	{
		return { y_ * vec.getZ() - z_ * vec.getY(), z_ * vec.getX() - x_ * vec.getZ(), x_ * vec.getY() - y_ * vec.getX() };
	}

	/* Angle between two points */
	[[nodiscard]] inline double angle(const Point& p) const
	{
		return acos(this->dotProduct(p) / (this->norm3D() * p.norm3D()));
	}


	/* Returns the angle formed by each coordinate with its corresponding edge */
	[[nodiscard]] inline Vector vectorAngles() const
	{
		double denom = norm3D();

		if (denom != 0)
		{
			return { (acos(x_ / denom) * 180.0 / M_PI), (acos(y_ / denom) * 180.0 / M_PI),
				       (acos(z_ / denom) * 180.0 / M_PI) };
		}

		return {};
	}


	// Overload << operator for Point.
	friend std::ostream& operator<<(std::ostream& out, const Point& p)
	{
		out << p.id_ << " " << p.x_ << " " << p.y_ << " " << p.z_;
		return out;
	}


	// Overload << operator for Point*
	// friend std::ostream& operator<<(std::ostream& out, const Point* p)
	// {
	// 	out << *p;
	// 	return out;
	// }

	friend Point abs(const Point& p) { return { std::abs(p.x_), std::abs(p.y_), std::abs(p.z_) }; }

	// Operator overloading (pointer version done with dereferencing in place)
	Point  operator+(const Point& v) const { return { x_ + v.getX(), y_ + v.getY(), z_ + v.getZ() }; }
	Point  operator-(const Point& v) const { return { x_ - v.getX(), y_ - v.getY(), z_ - v.getZ() }; }
	Point& operator+=(const Point& p)
	{
		x_ += p.getX();
		y_ += p.getY();
		z_ += p.getZ();

		return *this;
	}
	Point& operator-=(const Point& p)
	{
		x_ -= p.getX();
		y_ -= p.getY();
		z_ -= p.getZ();

		return *this;
	}

	bool operator==(const Point& p) const
	{
		return std::abs(x_ - p.getX()) < SENSEPSILON && std::abs(y_ - p.getY()) < SENSEPSILON &&
		       std::abs(z_ - p.getZ()) < SENSEPSILON;
	}

	bool operator!=(const Point& p) const { return !(*this == p); }

	Point& operator/=(const double val)
	{
		x_ /= val;
		y_ /= val;
		z_ /= val;

		return *this;
	}

	friend Point operator-(const Point& lhs, const double val) { return { lhs.x_ - val, lhs.y_ - val, lhs.z_ - val }; }

	friend Point operator+(const Point& lhs, const double val) { return { lhs.x_ + val, lhs.y_ + val, lhs.z_ + val }; }

	friend Point operator-(const double val, const Point& rhs) { return { rhs.x_ - val, rhs.y_ - val, rhs.z_ - val }; }

	friend Point operator+(const double val, const Point& rhs) { return { rhs.x_ + val, rhs.y_ + val, rhs.z_ + val }; }

	// Multiplication of vector and a scalar
	template<typename Time_t>
	Vector operator*(const Time_t scalar) const
	{
		return Vector(x_ * scalar, y_ * scalar, z_ * scalar);
	}

	// Division of vector and a scalar
	template<typename Time_t>
	Vector operator/(const Time_t scalar) const
	{
		return Vector(x_ / scalar, y_ / scalar, z_ / scalar);
	}

	template<typename Time_t>
	void operator/=(const Time_t scalar)
	{
		x_ /= scalar, y_ /= scalar, z_ /= scalar;
	}

    // Comparison operators based on lexicographic ordering in R^3 (used in approx search checker, might delete it later)
    bool operator<(const Point& other) const {
        if (std::abs(this->getX() - other.getX()) >= SENSEPSILON)
            return this->getX() < other.getX();
        if (std::abs(this->getY() - other.getY()) >= SENSEPSILON)
            return this->getY() < other.getY();
        return this->getZ() < other.getZ();
    }

    bool operator>(const Point& other) const {
        return other < *this;
    }

    bool operator<=(const Point& other) const {
        return !(other < *this);
    }

    bool operator>=(const Point& other) const {
        return !(*this < other);
    }

	// Getters and setters
	[[nodiscard]] inline size_t id() const { return id_; }
	inline void                       id(size_t id) { id_ = id; }
	[[nodiscard]] inline double       getX() const { return x_; }
	inline void                       setX(double x) { x_ = x; }
	[[nodiscard]] inline double       getY() const { return y_; }
	inline void                       setY(double y) { y_ = y; }
	[[nodiscard]] inline double       getZ() const { return z_; }
	inline void                       setZ(double z) { z_ = z; }
};