//
// Created by miguelyermo on 17/3/20.
//

#include "armadillo"
#include "geometry.hpp"
#include "Lpoint.hpp"

using namespace arma;

namespace geometry
{

// TODO: Migrate usages to Armadillo
Vector computePlane(const std::vector<Lpoint*>& points, Lpoint& centroid)
{
	// copy coordinates to  matrix in Eigen format

	if (points.size() < 3) std::cout << "Not enough points to build a plane\n";

	std::vector<Eigen::Vector3d> c;
	for (const Lpoint* p : points)
	{
		c.emplace_back(p->getX(), p->getY(), p->getZ());
	}

	size_t                                                                 num_atoms = c.size();
	Eigen::Matrix<Eigen::Vector3d::Scalar, Eigen::Dynamic, Eigen::Dynamic> coord(3, num_atoms);
	for (size_t i = 0; i < num_atoms; ++i)
		coord.col(i) = c[i];

	// calculate centroid
	Eigen::Vector3d mean(coord.row(0).mean(), coord.row(1).mean(), coord.row(2).mean());

	// subtract mean
	coord.row(0).array() -= mean(0);
	coord.row(1).array() -= mean(1);
	coord.row(2).array() -= mean(2);

	// we only need the left-singular matrix here
	//  http://math.stackexchange.com/questions/99299/best-fitting-plane-given-a-set-of-points
	auto            svd          = coord.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
	Eigen::Vector3d plane_normal = svd.matrixU().rightCols<1>();

	// Correct orientation (normals pointing up)
	Eigen::Vector3d viewPoint(0, 0, 1.0);
	double          dot = plane_normal.dot(viewPoint);
	if (dot < 0) { plane_normal *= -1; }

	// Build normal vector in the usual class
	Vector normal(plane_normal[0], plane_normal[1], plane_normal[2]);

	// Set centroid coordinates in the usual class
	centroid.setX(mean[0]);
	centroid.setY(mean[1]);
	centroid.setZ(mean[2]);

	return normal;
}

Point computeCentroid(const std::vector<Lpoint*>& points)
{
	double x{}, y{}, z{};
	size_t numPoints = points.size();
	for (const auto& p : points)
	{
		x += p->getX();
		y += p->getY();
		z += p->getZ();
	}
	return Vector(x, y, z);
}

void computeSVD(const std::vector<Point*>& points, mat& U, vec& s, mat& V)
/**
 * Given a list of points, computes the Singular Value Decomposition of those points
 * @param points
 * @param U
 * @param s
 * @param V
 */
{
	mat    P(3, points.size());
	size_t idx = 0;

	for (const Point* p : points)
	{
		P(0, idx) = p->getX();
		P(1, idx) = p->getY();
		P(2, idx) = p->getZ();
		idx++;
	}

	vec mean = arma::mean(P, 1);
	P.row(0) -= mean(0);
	P.row(1) -= mean(1);
	P.row(2) -= mean(2);

	// SVD
	svd_econ(U, s, V, P); // Faster version of svd. Same results at least in segmentation
}

void computeSVD(const std::vector<Lpoint*>& points, mat& U, vec& s, mat& V)
/**
 * Given a list of points, computes the Singular Value Decomposition of those points
 * @param points
 * @param U
 * @param s
 * @param V
 */
{
	mat    P(3, points.size());
	size_t idx = 0;

	for (const Lpoint* p : points)
	{
		P(0, idx) = p->getX();
		P(1, idx) = p->getY();
		P(2, idx) = p->getZ();
		idx++;
	}

	vec mean = arma::mean(P, 1);
	P.row(0) -= mean(0);
	P.row(1) -= mean(1);
	P.row(2) -= mean(2);

	// SVD
	svd_econ(U, s, V, P); // Faster version of svd. Same results at least in segmentation
}

std::tuple<Vector, double, std::vector<double>> computeGeometryArma(const std::vector<Lpoint*>& points)
/**
 * Given a list of points, computes its normal and its curvature
 * @param points
 */
{
	mat U, V;
	vec s, normal(3);
	computeSVD(points, U, s, V);


	// Normal Vector
	normal = U.col(2);
	// Correct orientation (normals pointing up)
	if (normal[2] < 0) normal *= -1;

	// Cast to Vectors ([] does not check bounds, () does!)
	Vector n(normal[0], normal[1], normal[2]);

	// Curvature
	std::vector<double> eigenvalues{ s[0] * s[0], s[1] * s[1], s[2] * s[2] };
	double              curvature = eigenvalues[2] / (eigenvalues[0] + eigenvalues[1] + eigenvalues[2]);

	return std::tuple(n, curvature, eigenvalues);
}

Vector computeNormal(const std::vector<Point*>& points)
{
	vec normal(3);

	// SVD
	mat U, V;
	vec s;
	computeSVD(points, U, s, V);

	normal = U.col(2);

	// Correct orientation (normals pointing up)
	if (normal[2] < 0) normal *= -1;

	// Cast to Vectors ([] does not check bounds, () does!)
	Vector n(normal[0], normal[1], normal[2]);

	return n;
}

Vector computeNormal(const std::vector<Lpoint*>& points)
{
	vec normal(3);

	// SVD
	mat U, V;
	vec s;
	computeSVD(points, U, s, V);

	normal = U.col(2);

	// Correct orientation (normals pointing up)
	if (normal[2] < 0) normal *= -1;

	// Cast to Vectors ([] does not check bounds, () does!)
	Vector n(normal[0], normal[1], normal[2]);

	return n;
}

std::pair<Vector, std::vector<double>> computeGeometry(const std::vector<Lpoint*>& points, Lpoint& centroid_)
{
	std::pair<Vector, std::vector<double>> geometry;
	// copy coordinates to  matrix in Eigen format

	if (points.size() < 3) std::cout << "Not enough points to build a plane\n";

	std::vector<Eigen::Vector3d> c;
	for (const Lpoint* p : points)
	{
		c.emplace_back(p->getX(), p->getY(), p->getZ());
	}

	size_t                                                                 num_atoms = c.size();
	Eigen::Matrix<Eigen::Vector3d::Scalar, Eigen::Dynamic, Eigen::Dynamic> coord(3, num_atoms);
	for (size_t i = 0; i < num_atoms; ++i)
		coord.col(i) = c[i];

	// calculate centroid
	Eigen::Vector3d centroid(coord.row(0).mean(), coord.row(1).mean(), coord.row(2).mean());

	// subtract centroid
	coord.row(0).array() -= centroid(0);
	coord.row(1).array() -= centroid(1);
	coord.row(2).array() -= centroid(2);

	// we only need the left-singular matrix here
	//  http://math.stackexchange.com/questions/99299/best-fitting-plane-given-a-set-of-points
	auto            svd          = coord.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
	Eigen::Vector3d plane_normal = svd.matrixU().rightCols<1>();
	Eigen::Vector3d sing_values  = svd.singularValues();

	// Correct orientation (normals pointing up)
	Eigen::Vector3d viewPoint(0, 0, 1.0);
	double          dot = plane_normal.dot(viewPoint);
	if (dot < 0) { plane_normal *= -1; }

	// Build normal vector and eigenvalues in the usual class
	Vector              normal(plane_normal[0], plane_normal[1], plane_normal[2]);
	std::vector<double> eigenvalues{ sing_values[0] * sing_values[0], sing_values[1] * sing_values[1],
		                               sing_values[2] * sing_values[2] };
	assert(eigenvalues[0] >= eigenvalues[1] >= eigenvalues[2] >= 0); // decreasing order

	// Set centroid coordinates in the usual class
	centroid_.setX(centroid[0]);
	centroid_.setY(centroid[1]);
	centroid_.setZ(centroid[2]);

	geometry.first  = normal;
	geometry.second = eigenvalues;

	return geometry;
}

Limits getmbb(const std::vector<Lpoint*>& points)
/**
 * Computes the axis oriented Minimum Bounding Box of a set of pointers to points
 */
{
	double minX = std::numeric_limits<double>::max();
	double minY = std::numeric_limits<double>::max();
	double maxX = std::numeric_limits<double>::min();
	double maxY = std::numeric_limits<double>::min();

	for (const Lpoint* p : points)
	{
		if (p->getX() < minX) minX = p->getX();
		if (p->getY() < minY) minY = p->getY();
		if (p->getX() > maxX) maxX = p->getX();
		if (p->getY() > maxY) maxY = p->getY();
	}

	Limits limits(minX, minY, maxX, maxY);

	return limits;
}

} // namespace geometry