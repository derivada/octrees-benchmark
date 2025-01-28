//
// Created by miguel.yermo on 5/03/20.
//

#include "octree.hpp"

#include "Geometry/Box.hpp"
#include "NeighborKernels/KernelFactory.hpp"
#include "PointEncoding/common.hpp"
#include "PointEncoding/morton_encoder.hpp"
#include "PointEncoding/hilbert_encoder.hpp"
#include <algorithm>
#include <unordered_map>
#include "Geometry/Lpoint.hpp"
#include "Geometry/Lpoint64.hpp"

// TODO: move this implementations to octree.hpp so we dont need to predefine all possible pointer octrees
template class Octree<Point, PointEncoding::NoEncoder>;
template class Octree<Lpoint, PointEncoding::NoEncoder>;
template class Octree<Lpoint64, PointEncoding::NoEncoder>;
template class Octree<Point, PointEncoding::HilbertEncoder3D>;
template class Octree<Lpoint, PointEncoding::HilbertEncoder3D>;
template class Octree<Lpoint64, PointEncoding::HilbertEncoder3D>;
template class Octree<Point, PointEncoding::MortonEncoder3D>;
template class Octree<Lpoint, PointEncoding::MortonEncoder3D>;
template class Octree<Lpoint64, PointEncoding::MortonEncoder3D>;

template <PointType Point_t, typename Encoder_t>
Octree<Point_t, Encoder_t>::Octree() = default;

template <PointType Point_t, typename Encoder_t>
Octree<Point_t, Encoder_t>::Octree(std::vector<Point_t>& points, std::optional<std::reference_wrapper<std::vector<PointMetadata>>> metadata)
{
	center_ = mbb(points, radius_);
	octants_.reserve(OCTANTS_PER_NODE);
	buildOctree(points);
}

template <PointType Point_t, typename Encoder_t>
Octree<Point_t, Encoder_t>::Octree(std::vector<Point_t*>& points, std::optional<std::reference_wrapper<std::vector<PointMetadata>>> metadata)
{
	center_ = mbb(points, radius_);
	octants_.reserve(OCTANTS_PER_NODE);
	buildOctree(points);
}

template <PointType Point_t, typename Encoder_t>
Octree<Point_t, Encoder_t>::Octree(const Vector& center, const float radius) : center_(center), radius_(radius)
{
	octants_.reserve(OCTANTS_PER_NODE);
};

template <PointType Point_t, typename Encoder_t>
Octree<Point_t, Encoder_t>::Octree(Vector center, float radius, std::vector<Point_t*>& points) : center_(center), radius_(radius)
{
	octants_.reserve(OCTANTS_PER_NODE);
	buildOctree(points);
}

template <PointType Point_t, typename Encoder_t>
Octree<Point_t, Encoder_t>::Octree(Vector center, float radius, std::vector<Point_t>& points) : center_(center), radius_(radius)
{
	octants_.reserve(OCTANTS_PER_NODE);
	buildOctree(points);
}

template <PointType Point_t, typename Encoder_t>
void Octree<Point_t, Encoder_t>::computeOctreeLimits()
/**
   * Compute the minimum and maximum coordinates of the octree bounding box.
   */
{
	min_.setX(center_.getX() - radius_);
	min_.setY(center_.getY() - radius_);
	max_.setX(center_.getX() + radius_);
	max_.setY(center_.getY() + radius_);
}

template <PointType Point_t, typename Encoder_t>
std::vector<std::pair<Point, size_t>> Octree<Point_t, Encoder_t>::computeNumPoints() const
/**
 * @brief Returns a vector containing the number of points of all populated octants
 * @param numPoints
 */
{
	std::vector<std::pair<Point, size_t>> numPoints;

	std::vector<std::reference_wrapper<const Octree>> toVisit;
	toVisit.emplace_back(*this);

	while (!toVisit.empty())
	{
		const auto& octant = toVisit.back().get();
		toVisit.pop_back();

		if (octant.isLeaf())
		{
			if (!octant.isEmpty()) { numPoints.emplace_back(octant.center_, octant.points_.size()); }
		}
		else { std::copy(std::begin(octant.octants_), std::end(octant.octants_), std::back_inserter(toVisit)); }
	}

	return numPoints;
}

template <PointType Point_t, typename Encoder_t>
std::vector<std::pair<Point, double>> Octree<Point_t, Encoder_t>::computeDensities() const
/*
 * Returns a vector containing the densities of all populated octrees
 */
{
	std::vector<std::pair<Point, double>> densities;

	std::vector<std::reference_wrapper<const Octree>> toVisit;
	toVisit.emplace_back(*this);

	while (!toVisit.empty())
	{
		const auto& octant = toVisit.back().get();
		toVisit.pop_back();

		if (octant.isLeaf())
		{
			if (!octant.isEmpty()) { densities.emplace_back(octant.center_, octant.getDensity()); }
		}
		else { std::copy(std::begin(octant.octants_), std::end(octant.octants_), std::back_inserter(toVisit)); }
	}

	return densities;
}

template <PointType Point_t, typename Encoder_t>
void Octree<Point_t, Encoder_t>::writeDensities(const std::filesystem::path& path) const
/**
 * @brief Compute and write to file the density of each non-empty octan of a given octree.
 * @param path
 */
{
	const auto densities = computeDensities();

	std::ofstream f(path);
	f << std::fixed << std::setprecision(2);
	for (const auto& v : densities)
	{
		f << v.first.getX() << " " << v.first.getY() << v.first.getZ() << " " << v.second << "\n";
	}
}

template <PointType Point_t, typename Encoder_t>
void Octree<Point_t, Encoder_t>::writeNumPoints(const std::filesystem::path& path) const
/**
 * @brief Compute and write to file the density of each non-empty octan of a given octree.
 * @param path
 */
{
	const auto numPoints = computeNumPoints();

	std::ofstream f(path);
	f << std::fixed << std::setprecision(2);
	for (const auto& v : numPoints)
	{
		f << v.first.getX() << " " << v.first.getY() << v.first.getZ() << " " << v.second << "\n";
	}
}

// FIXME: This function may overlap with some parts of extractPoint[s]
template <PointType Point_t, typename Encoder_t>
const Octree<Point_t, Encoder_t>* Octree<Point_t, Encoder_t>::findOctant(const Point_t* p) const
/**
 * @brief Find the octant containing a given point.
 * @param p
 * @return
 */
{
	if (isLeaf())
	{
		for (const auto& point : points_)
		{
			auto it = std::find(points_.begin(), points_.end(), point);
			if (it != points_.end()) // Found
			{
				return this; // If findOctant is const, fiesta loca!!
			}
		}
	}
	else
	{
		// If Point_t is const, fiesta loca!
		return octants_[octantIdx(p)].findOctant(p);
	}

	return nullptr;
}

template <PointType Point_t, typename Encoder_t>
bool Octree<Point_t, Encoder_t>::isInside2D(const Point& p) const
/**
   * Checks if a point is inside the octree limits.
   * @param p
   * @return
   */
{
	if (p.getX() >= min_.getX() && p.getY() >= min_.getY())
	{
		if (p.getX() <= max_.getX() && p.getY() <= max_.getY()) { return true; }
	}

	return false;
}

template <PointType Point_t, typename Encoder_t>
void Octree<Point_t, Encoder_t>::insertPoints(std::vector<Point_t>& points)
{
	for (Point_t& p : points)
	{
		insertPoint(&p);
	}
}

template <PointType Point_t, typename Encoder_t>
void Octree<Point_t, Encoder_t>::insertPoints(std::vector<Point_t*>& points)
{
	for (Point_t* p : points)
	{
		insertPoint(p);
	}
}

template <PointType Point_t, typename Encoder_t>
void Octree<Point_t, Encoder_t>::insertPoint(Point_t* p)
{
	unsigned int idx = 0;

	if (isLeaf())
	{
		if (isEmpty()) { points_.emplace_back(p); }
		else
		{
			// NOTE: This was wrong because it allowed nodes with MAX_POINTS + 1 children
			// if (points_.size() > MAX_POINTS && radius_ >= MIN_OCTANT_RADIUS) <-- wrong
			if (points_.size() >= MAX_POINTS && radius_ >= MIN_OCTANT_RADIUS)
			{
				createOctants(); // Creation of children octree
				fillOctants();   // Move points from current Octree to its corresponding children.
				idx = octantIdx(p);
				octants_[idx].insertPoint(p);
			}
			else { points_.emplace_back(p); }
		}
	}
	else
	{
		idx = octantIdx(p);
		octants_[idx].insertPoint(p);
	}
}

template <PointType Point_t, typename Encoder_t>
void Octree<Point_t, Encoder_t>::createOctants()
{
	for (size_t i = 0; i < OCTANTS_PER_NODE; i++)
	{
		auto newCenter = center_;
		newCenter.setX(newCenter.getX() + radius_ * ((i & 4U) != 0U ? 0.5F : -0.5F));
		newCenter.setY(newCenter.getY() + radius_ * ((i & 2U) != 0U ? 0.5F : -0.5F));
		newCenter.setZ(newCenter.getZ() + radius_ * ((i & 1U) != 0U ? 0.5F : -0.5F));
		octants_.emplace_back(newCenter, 0.5F * radius_);
	}
}

template <PointType Point_t, typename Encoder_t>
void Octree<Point_t, Encoder_t>::fillOctants()
{
	for (Point_t* p : points_)
	{
		// Idx of the octant where a point should be inserted.
		const auto idx = octantIdx(p);
		octants_[idx].insertPoint(p);
	}

	points_.clear();
}

template <PointType Point_t, typename Encoder_t>
size_t Octree<Point_t, Encoder_t>::octantIdx(const Point_t* p) const
{
	size_t child = 0;

	if (p->getX() >= center_.getX()) { child |= 4U; }
	if (p->getY() >= center_.getY()) { child |= 2U; }
	if (p->getZ() >= center_.getZ()) { child |= 1U; }

	return child;
}

template <PointType Point_t, typename Encoder_t>
void Octree<Point_t, Encoder_t>::buildOctree(std::vector<Point_t>& points)
/**
   * Build the Octree
   */
{
	computeOctreeLimits();
	insertPoints(points);
}

template <PointType Point_t, typename Encoder_t>
void Octree<Point_t, Encoder_t>::buildOctree(std::vector<Point_t*>& points)
/**
   * Build the Octree
   */
{
	computeOctreeLimits();
	insertPoints(points);
}

template <PointType Point_t, typename Encoder_t>
std::vector<Point_t*> Octree<Point_t, Encoder_t>::KNN(const Point& p, const size_t k, const size_t maxNeighs) const
/**
 * @brief KNN algorithm. Returns the min(k, maxNeighs) nearest neighbors of a given point p
 * @param p
 * @param k
 * @param maxNeighs
 * @return
 */
{
	std::vector<Point_t*>             knn{};
	std::unordered_map<size_t, bool> wasAdded{};

	double r = 1.0;

	size_t nmax = std::min(k, maxNeighs);

	while (knn.size() <= nmax)
	{
		auto neighs = searchNeighbors<Kernel_t::sphere>(p, r);

		if (knn.size() + neighs.size() > nmax)
		{ // Add all points if there is room for them
			std::sort(neighs.begin(), neighs.end(),
			          [&p](Point_t* a, Point_t* b) { return a->distance3D(p) < b->distance3D(p); });
		}

		for (const auto& n : neighs)
		{
			if (!wasAdded[n->id()])
			{
				wasAdded[n->id()] = true;
				knn.push_back(n); // Conditional inserting?
			}
		}
		// TODO: Max radius?
		r *= 2;
	}
	return knn;
}

template <PointType Point_t, typename Encoder_t>
void Octree<Point_t, Encoder_t>::writeOctree(std::ofstream& f, size_t index) const
{
	index++;
	f << "Depth: " << index << " "
	  << "numPoints: " << points_.size() << "\n";
	f << "Center: " << center_ << " Radius: " << radius_ << "\n";

	if (isLeaf())
	{
		for (const auto& p : points_)
		{
			// f << "\t " << *p << " " << p->getClass() << " at: " << p << "\n";
			f << "\t " << *p << " "  << " at: " << p << "\n";
		}
	}
	else
	{
		for (const auto& octant : octants_)
		{
			octant.writeOctree(f, index);
		}
	}
}

template <PointType Point_t, typename Encoder_t>
void Octree<Point_t, Encoder_t>::extractPoint(const Point_t* p)
/**
 * Searches for p and (if found) removes it from the octree.
 *
 * @param p
 */
{
	unsigned int idx = 0;

	if (isLeaf())
	{
		auto index = std::find(points_.begin(), points_.end(), p);
		if (index != points_.end()) { points_.erase(index); }
	}
	else
	{
		idx = octantIdx(p);
		octants_[idx].extractPoint(p);
		if (octants_[idx].isLeaf() && octants_[idx].isEmpty())
		// Leaf has been emptied. Check if all octants are empty leaves, and clear octants_ if so
		{
			bool emptyLeaves = true;
			for (size_t i = 0; emptyLeaves && i < OCTANTS_PER_NODE; i++)
			{
				emptyLeaves = octants_[i].isLeaf() && octants_[i].isEmpty();
			}
			if (emptyLeaves) { octants_.clear(); }
		}
	}
}

template <PointType Point_t, typename Encoder_t>
Point_t* Octree<Point_t, Encoder_t>::extractPoint()
/**
 * Searches for a point and, if it founds one, removes it from the octree.
 *
 * @return pointer to one of the octree's points, or nullptr if the octree is empty
 */
{
	if (isLeaf())
	{
		if (points_.empty()) { return nullptr; }

		auto* p = points_.back();
		points_.pop_back();
		return p;
	}

	int nonEmptyOctantId = -1;
	int i                = 0;
	for (const auto& octant : octants_)
	{
		if (!octants_[i].isLeaf() || !octants_[i].isEmpty())
		{
			nonEmptyOctantId = i;
			break;
		}
		i++;
	}

	if (nonEmptyOctantId == -1)
	{
		std::cerr << "Warning: Found octree with 8 empty octants\n";
		return nullptr;
	}

	auto* p = octants_[nonEmptyOctantId].extractPoint();

	if (octants_[nonEmptyOctantId].isLeaf() && octants_[nonEmptyOctantId].isEmpty())
	// Leaf has been emptied. Check if all octants are empty leaves, and clear octants_ if so
	{
		bool emptyLeaves = true;
		for (const auto& octant : octants_)
		{
			emptyLeaves = octants_[i].isLeaf() && octants_[i].isEmpty();
		}
		if (emptyLeaves) { octants_.clear(); }
	}

	return p;
}

template <PointType Point_t, typename Encoder_t>
void Octree<Point_t, Encoder_t>::extractPoints(std::vector<Point_t>& points)
{
	for (Point_t& p : points)
	{
		extractPoint(&p);
	}
}

template <PointType Point_t, typename Encoder_t>
void Octree<Point_t, Encoder_t>::extractPoints(std::vector<Point_t*>& points)
{
	for (Point_t* p : points)
	{
		extractPoint(p);
	}
}

template <PointType Point_t, typename Encoder_t>
std::vector<Point_t*> Octree<Point_t, Encoder_t>::searchEraseCircleNeighbors(const std::vector<Point_t*>& points, double radius)
/*
 * Searches points' circle neighbors and erases them from the octree.
 */
{
	std::vector<Point_t*> pointsNeighbors{};

	for (const auto* p : points)
	{
		auto pNeighbors = searchCircleNeighbors(p, radius);
		if (!pNeighbors.empty())
		{
			extractPoints(pNeighbors);
			pointsNeighbors.reserve(pointsNeighbors.size() + pNeighbors.size());
			std::move(std::begin(pNeighbors), std::end(pNeighbors), std::back_inserter(pointsNeighbors));
		}
	}

	return pointsNeighbors;
}

template <PointType Point_t, typename Encoder_t>
std::vector<Point_t*> Octree<Point_t, Encoder_t>::searchEraseSphereNeighbors(const std::vector<Point_t*>& points, float radius)
{
	std::vector<Point_t*> pointsNeighbors{};

	for (const auto* p : points)
	{
		auto pNeighbors = searchSphereNeighbors(*p, radius);
		if (!pNeighbors.empty())
		{
			extractPoints(pNeighbors);
			pointsNeighbors.reserve(pointsNeighbors.size() + pNeighbors.size());
			std::move(std::begin(pNeighbors), std::end(pNeighbors), std::back_inserter(pointsNeighbors));
		}
	}

	return pointsNeighbors;
}

/** Connected inside a spherical shell*/
template <PointType Point_t, typename Encoder_t>
std::vector<Point_t*> Octree<Point_t, Encoder_t>::searchConnectedShellNeighbors(const Point& point, const float nextDoorDistance,
                                                           const float minRadius, const float maxRadius) const
{
	std::vector<Point_t*> connectedShellNeighs;

	auto connectedSphereNeighs = searchSphereNeighbors(point, maxRadius);
	connectedSphereNeighs      = connectedNeighbors(&point, connectedSphereNeighs, nextDoorDistance);
	for (auto* n : connectedSphereNeighs)
	{
		if (n->distance3D(point) >= minRadius) { connectedShellNeighs.push_back(n); }
	}

	return connectedShellNeighs;
}

/** Connected circle neighbors*/
template <PointType Point_t, typename Encoder_t>
std::vector<Point_t*> Octree<Point_t, Encoder_t>::searchEraseConnectedCircleNeighbors(const float nextDoorDistance)
{
	std::vector<Point_t*> connectedCircleNeighbors;

	auto* p = extractPoint();
	if (p == nullptr) { return connectedCircleNeighbors; }
	connectedCircleNeighbors.push_back(p);

	auto closeNeighbors = searchEraseCircleNeighbors(std::vector<Point_t*>{ p }, nextDoorDistance);
	while (!closeNeighbors.empty())
	{
		connectedCircleNeighbors.insert(connectedCircleNeighbors.end(), closeNeighbors.begin(), closeNeighbors.end());
		closeNeighbors = searchEraseCircleNeighbors(closeNeighbors, nextDoorDistance);
	}

	return connectedCircleNeighbors;
}

template <PointType Point_t, typename Encoder_t>
std::vector<Point_t*> Octree<Point_t, Encoder_t>::connectedNeighbors(const Point* point, std::vector<Point_t*>& neighbors,
                                                const float nextDoorDistance)
/**
	 * Filters neighbors which are not connected to point through a chain of next-door neighbors. Erases neighbors in the
	 * process.
	 *
	 * @param point
	 * @param neighbors
	 * @param radius
	 * @return
	 */
{
	std::vector<Point_t*> connectedNeighbors;
	if (neighbors.empty()) { return connectedNeighbors; }

	auto waiting = extractCloseNeighbors(point, neighbors, nextDoorDistance);

	while (!waiting.empty())
	{
		auto* v = waiting.back();
		waiting.pop_back();
		auto vCloseNeighbors = extractCloseNeighbors(v, neighbors, nextDoorDistance);
		waiting.insert(waiting.end(), vCloseNeighbors.begin(), vCloseNeighbors.end());

		connectedNeighbors.push_back(v);
	}

	return connectedNeighbors;
}

template <PointType Point_t, typename Encoder_t>
std::vector<Point_t*> Octree<Point_t, Encoder_t>::extractCloseNeighbors(const Point* p, std::vector<Point_t*>& neighbors, const float radius)
/**
	 * Fetches neighbors within radius from p, erasing them from neighbors and returning them.
	 *
	 * @param p
	 * @param neighbors
	 * @param radius
	 * @return
	 */
{
	std::vector<Point_t*> closeNeighbors;

	for (size_t i = 0; i < neighbors.size();)
	{
		if (neighbors[i]->distance3D(*p) <= radius)
		{
			closeNeighbors.push_back(neighbors[i]);
			neighbors.erase(neighbors.begin() + i);
		}
		else { i++; }
	}

	return closeNeighbors;
}

template <PointType Point_t, typename Encoder_t>
std::vector<Point_t*> Octree<Point_t, Encoder_t>::kClosestCircleNeighbors(const Point_t* p, const size_t k) const
/**
	 * Fetches the (up to if not enough points in octree) k closest neighbors with respect to 2D-distance.
	 *
	 * @param p
	 * @param k
	 * @return
	 */
{
	double               rMin = SENSEPSILON * static_cast<double>(k);
	const double         rMax = 2.0 * M_SQRT2 * radius_;
	std::vector<Point_t*> closeNeighbors;
	for (closeNeighbors = searchCircleNeighbors(p, rMin); closeNeighbors.size() < k && rMin < 2 * rMax; rMin *= 2)
	{
		closeNeighbors = searchCircleNeighbors(p, rMin);
	}

	while (closeNeighbors.size() > k)
	{
		size_t furthestIndex;
		double furthestDistanceSquared = 0.0;
		for (size_t i = 0; i < closeNeighbors.size(); i++)
		{
			const auto iDistanceSquared = p->distance2Dsquared(*closeNeighbors[i]);
			if (iDistanceSquared > furthestDistanceSquared)
			{
				furthestDistanceSquared = iDistanceSquared;
				furthestIndex           = i;
			}
		}
		closeNeighbors.erase(closeNeighbors.begin() + furthestIndex);
	}
	return closeNeighbors;
}

template <PointType Point_t, typename Encoder_t>
std::vector<Point_t*> Octree<Point_t, Encoder_t>::nCircleNeighbors(const Point_t* p, const size_t n, float& radius, const float minRadius,
                                              const float maxRadius, const float maxIncrement,
                                              const float maxDecrement) const
/**
	 * Radius-adaptive search method for circle neighbors.
	 *
	 * @param p
	 * @param n
	 * @param radius
	 * @param minRadius
	 * @param maxRadius
	 * @param maxStep
	 * @return circle neighbors
	 */
{
	auto neighs = searchCircleNeighbors(p, radius);

	float radiusOffset = (float(n) - float(neighs.size())) * SENSEPSILON;
	if (radiusOffset > maxIncrement) { radiusOffset = maxIncrement; }
	else if (radiusOffset < -maxDecrement) { radiusOffset = -maxDecrement; }

	radius += radiusOffset;
	if (radius > maxRadius) { radius = maxRadius; }
	else if (radius < minRadius) { radius = minRadius; }

	return neighs;
}

template <PointType Point_t, typename Encoder_t>
std::vector<Point_t*> Octree<Point_t, Encoder_t>::nSphereNeighbors(const Point_t& p, const size_t n, float& radius, const float minRadius,
                                              const float maxRadius, const float maxStep) const
/**
	 * Radius-adaptive search method for sphere neighbors.
	 *
	 * @param p
	 * @param n
	 * @param radius
	 * @param minRadius
	 * @param maxRadius
	 * @param maxStep
	 * @return sphere neighbors
	 */
{
	auto neighs = searchSphereNeighbors(p, radius);

	float radiusOffset = (float(n) - float(neighs.size())) * SENSEPSILON;
	if (radiusOffset > maxStep) { radiusOffset = maxStep; }
	else if (radiusOffset < -maxStep) { radiusOffset = -maxStep; }

	radius += radiusOffset;
	if (radius > maxRadius) { radius = maxRadius; }
	else if (radius < minRadius) { radius = minRadius; }

	return neighs;
}
