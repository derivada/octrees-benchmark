//
// Created by miguel.yermo on 5/03/20.
//

#include "octree_v2.hpp"

#include "Box.hpp"
#include "NeighborKernels/KernelFactory.hpp"

#include <algorithm>
#include <unordered_map>

OctreeV2::OctreeV2() = default;

// This constructor is used to initialize child nodes without adding any extra new points
OctreeV2::OctreeV2(const Vector& center, const float radius) : center_(center), radius_(radius) {};


void OctreeV2::computeOctreeLimits()
/**
   * Compute the minimum and maximum coordinates of the octree bounding box.
   */
{
	min_.setX(center_.getX() - radius_);
	min_.setY(center_.getY() - radius_);
	max_.setX(center_.getX() + radius_);
	max_.setY(center_.getY() + radius_);
}

std::vector<std::pair<Point, size_t>> OctreeV2::computeNumPoints() const
/**
 * @brief Returns a vector containing the number of points of all populated octants
 * @param numPoints
 */
{
	std::vector<std::pair<Point, size_t>> numPoints;

	std::vector<std::reference_wrapper<const OctreeV2>> toVisit;
	toVisit.emplace_back(*this);

	while (!toVisit.empty())
	{
		const auto& octant = toVisit.back().get();
		toVisit.pop_back();

		if (octant.isLeaf())
		{
			if (!octant.isEmpty()) { numPoints.emplace_back(octant.center_, octant.points_.size()); }
		}
		else 
		{ 
			auto octants = octant.getOctants();
			std::copy(std::begin(octants), std::end(octants), std::back_inserter(toVisit)); 
		}
	}

	return numPoints;
}

std::vector<std::pair<Point, double>> OctreeV2::computeDensities() const
/*
 * Returns a vector containing the densities of all populated octrees
 */
{
	std::vector<std::pair<Point, double>> densities;

	std::vector<std::reference_wrapper<const OctreeV2>> toVisit;
	toVisit.emplace_back(*this);

	while (!toVisit.empty())
	{
		const auto& octant = toVisit.back().get();
		toVisit.pop_back();

		if (octant.isLeaf())
		{
			if (!octant.isEmpty()) { densities.emplace_back(octant.center_, octant.getDensity()); }
		}
		else 
		{ 
			auto octants = octant.getOctants();
			std::copy(std::begin(octants), std::end(octants), std::back_inserter(toVisit)); 
		}
	}

	return densities;
}

void OctreeV2::writeDensities(const std::filesystem::path& path) const
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

void OctreeV2::writeNumPoints(const std::filesystem::path& path) const
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
const OctreeV2* OctreeV2::findOctant(const Lpoint* p) const
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
		// If Lpoint is const, fiesta loca!
		auto octant = getOctant(octantIdx(p));
		return octant->findOctant(p);
	}

	return nullptr;
}

bool OctreeV2::isInside2D(const Point& p) const
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

void OctreeV2::insertPoints(std::vector<Lpoint>& points)
{
	// for (Lpoint& p : points)
	// {
	// 	insertPoint(&p);
	// }

	// We start by checking if we need to create the child octants
	if (points.size() > MAX_POINTS && radius_ >= MIN_OCTANT_RADIUS && !maxDepthReached()) {
		std::vector<std::vector<Lpoint*>> child_points(OCTANTS_PER_NODE, std::vector<Lpoint*>());
		// Compute the morton codes of each point and redistribute them into an array for each octant
		for(auto& p : points)
		{
			child_points[octantIdx(&p)].emplace_back(&p);
		}
		// Create the new octants
		createOctants();
		// Insert the points into the new octants recursively
		for (int i = 0; i < OCTANTS_PER_NODE; i++)
		{
			auto octant = getOctant(i);
			octant->insertPoints(child_points[i]);
		}
	} else {
		// Fit all points in the current octree
		points_storage_.reserve(points.size());
		points_.reserve(points.size());
		
		// Move actual points to storage and update pointers
		for(int i = 0; i < points.size(); i++)
		{
			points_storage_[i] = points[i];  // Copy the point
			points_[i] = &points_storage_[i];  // Store pointer to the copied point
		}
	}
}

void OctreeV2::insertPoints(std::vector<Lpoint*>& points)
{
	// for (Lpoint* p : points)
	// {
	// 	insertPoint(p);
	// }
	
	if (points.size() > MAX_POINTS && radius_ >= MIN_OCTANT_RADIUS && !maxDepthReached()) {
		std::vector<std::vector<Lpoint*>> child_points(OCTANTS_PER_NODE, std::vector<Lpoint*>());
		// Compute the morton codes of each point and redistribute them into an array for each octant
		for(const auto& p : points)
		{
			child_points[octantIdx(p)].emplace_back(p);
		}
		// Create the new octants
		createOctants();
		// Insert the points into the new octants recursively
		for (int i = 0; i < OCTANTS_PER_NODE; i++)
		{
			auto octant = getOctant(i);
			octant->insertPoints(child_points[i]);
		}
	} else {
		// Fit all points in the current octree
		points_storage_.resize(points.size());
		points_.resize(points.size());
		
		// Copy points to storage and update pointers
		for(size_t i = 0; i < points.size(); i++)
		{
			points_storage_[i] = *points[i];  // Copy the point that the pointer points to
			points_[i] = &points_storage_[i];  // Store pointer to our copied point
		}
	}
}

void OctreeV2::insertPoint(Lpoint* p)
{
	unsigned int idx = 0;

	if (isLeaf())
	{
		if (isEmpty()) { points_.emplace_back(p); }
		else
		{
			if (points_.size() > MAX_POINTS && radius_ >= MIN_OCTANT_RADIUS && !maxDepthReached())
			{
				createOctants(); // Creation of children octree
				fillOctants();   // Move points from current Octree to its corresponding children.
				idx = octantIdx(p);
				auto octant = getOctant(idx);
				octant->insertPoint(p);
			}
			else { 
				points_.emplace_back(p); 
			}
		}
	}
	else
	{
		idx = octantIdx(p);
		auto octant = getOctant(idx);
		octant->insertPoint(p);
	}
}

void OctreeV2::fillOctants()
{
	// Vector for re-allocating the points
	for (Lpoint* p : points_)
	{
		// Idx of the octant where a point should be inserted.
		const auto idx = octantIdx(p);
		auto octant = getOctant(idx);
		octant->insertPoint(p);
	}

	points_.clear();
}

inline size_t OctreeV2::octantIdx(const Point* p) const
{
	size_t child = 0;

	if (p->getX() >= center_.getX()) { child |= 4U; }
	if (p->getY() >= center_.getY()) { child |= 2U; }
	if (p->getZ() >= center_.getZ()) { child |= 1U; }

	return child;
}

void OctreeV2::buildOctree(std::vector<Lpoint>& points)
/**
   * Build the Octree
   */
{
	computeOctreeLimits();
	insertPoints(points);
}

void OctreeV2::buildOctree(std::vector<Lpoint*>& points)
/**
   * Build the Octree
   */
{
	computeOctreeLimits();
	insertPoints(points);
}

std::vector<Lpoint*> OctreeV2::KNN(const Point& p, const size_t k, const size_t maxNeighs) const
/**
 * @brief KNN algorithm. Returns the min(k, maxNeighs) nearest neighbors of a given point p
 * @param p
 * @param k
 * @param maxNeighs
 * @return
 */
{
	std::vector<Lpoint*>             knn{};
	std::unordered_map<size_t, bool> wasAdded{};

	double r = 1.0;

	size_t nmax = std::min(k, maxNeighs);

	while (knn.size() <= nmax)
	{
		auto neighs = searchNeighbors<Kernel_t::sphere>(p, r);

		if (knn.size() + neighs.size() > nmax)
		{ // Add all points if there is room for them
			std::sort(neighs.begin(), neighs.end(),
			          [&p](Lpoint* a, Lpoint* b) { return a->distance3D(p) < b->distance3D(p); });
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

void OctreeV2::writeOctree(std::ofstream& f, size_t index) const
{
	index++;
	f << "Depth: " << index << " "
	  << "numPoints: " << points_.size() << "\n";
	f << "Center: " << center_ << " Radius: " << radius_ << "\n";

	if (isLeaf())
	{
		for (const auto& p : points_)
		{
			f << "\t " << *p << " " << p->getClass() << " at: " << p << "\n";
		}
	}
	else
	{
		for (const auto& octant : getOctants())
		{
			octant.get().writeOctree(f, index);
		}
	}
}

void OctreeV2::extractPoint(const Lpoint* p)
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
		auto octant = getOctant(idx);
		octant->extractPoint(p);
		if (octant->isLeaf() && octant->isEmpty())
		// Leaf has been emptied. Check if all octants are empty leaves, and clear octants_ if so
		{
			bool emptyLeaves = true;
			for (size_t i = 0; emptyLeaves && i < OCTANTS_PER_NODE; i++)
			{
				auto octant_i = getOctant(i);
				emptyLeaves = octant_i->isLeaf() && octant_i->isEmpty();
			}
			if (emptyLeaves) { clearOctants(); }
		}
	}
}

Lpoint* OctreeV2::extractPoint()
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
	for (const auto& octant : getOctants())
	{
		auto octant_i = getOctant(i);
		if (!octant_i->isLeaf() || !octant_i->isEmpty())
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
	auto octant = getOctant(nonEmptyOctantId);
	auto* p = octant->extractPoint();

	if (octant->isLeaf() && octant->isEmpty())
	// Leaf has been emptied. Check if all octants are empty leaves, and clear octants_ if so
	{
		bool emptyLeaves = true;
		for (const auto& octant : getOctants())
		{
			auto octant_i = getOctant(i);
			emptyLeaves = octant_i->isLeaf() && octant_i->isEmpty();
		}
		if (emptyLeaves) { clearOctants(); }
	}

	return p;
}

void OctreeV2::extractPoints(std::vector<Lpoint>& points)
{
	for (Lpoint& p : points)
	{
		extractPoint(&p);
	}
}

void OctreeV2::extractPoints(std::vector<Lpoint*>& points)
{
	for (Lpoint* p : points)
	{
		extractPoint(p);
	}
}

std::vector<Lpoint*> OctreeV2::searchEraseCircleNeighbors(const std::vector<Lpoint*>& points, double radius)
/*
 * Searches points' circle neighbors and erases them from the octree.
 */
{
	std::vector<Lpoint*> pointsNeighbors{};

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

std::vector<Lpoint*> OctreeV2::searchEraseSphereNeighbors(const std::vector<Lpoint*>& points, float radius)
{
	std::vector<Lpoint*> pointsNeighbors{};

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
std::vector<Lpoint*> OctreeV2::searchConnectedShellNeighbors(const Point& point, const float nextDoorDistance,
                                                           const float minRadius, const float maxRadius) const
{
	std::vector<Lpoint*> connectedShellNeighs;

	auto connectedSphereNeighs = searchSphereNeighbors(point, maxRadius);
	connectedSphereNeighs      = connectedNeighbors(&point, connectedSphereNeighs, nextDoorDistance);
	for (auto* n : connectedSphereNeighs)
	{
		if (n->distance3D(point) >= minRadius) { connectedShellNeighs.push_back(n); }
	}

	return connectedShellNeighs;
}

/** Connected circle neighbors*/
std::vector<Lpoint*> OctreeV2::searchEraseConnectedCircleNeighbors(const float nextDoorDistance)
{
	std::vector<Lpoint*> connectedCircleNeighbors;

	auto* p = extractPoint();
	if (p == nullptr) { return connectedCircleNeighbors; }
	connectedCircleNeighbors.push_back(p);

	auto closeNeighbors = searchEraseCircleNeighbors(std::vector<Lpoint*>{ p }, nextDoorDistance);
	while (!closeNeighbors.empty())
	{
		connectedCircleNeighbors.insert(connectedCircleNeighbors.end(), closeNeighbors.begin(), closeNeighbors.end());
		closeNeighbors = searchEraseCircleNeighbors(closeNeighbors, nextDoorDistance);
	}

	return connectedCircleNeighbors;
}

std::vector<Lpoint*> OctreeV2::connectedNeighbors(const Point* point, std::vector<Lpoint*>& neighbors,
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
	std::vector<Lpoint*> connectedNeighbors;
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

std::vector<Lpoint*> OctreeV2::extractCloseNeighbors(const Point* p, std::vector<Lpoint*>& neighbors, const float radius)
/**
	 * Fetches neighbors within radius from p, erasing them from neighbors and returning them.
	 *
	 * @param p
	 * @param neighbors
	 * @param radius
	 * @return
	 */
{
	std::vector<Lpoint*> closeNeighbors;

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

std::vector<Lpoint*> OctreeV2::kClosestCircleNeighbors(const Lpoint* p, const size_t k) const
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
	std::vector<Lpoint*> closeNeighbors;
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

std::vector<Lpoint*> OctreeV2::nCircleNeighbors(const Lpoint* p, const size_t n, float& radius, const float minRadius,
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

std::vector<Lpoint*> OctreeV2::nSphereNeighbors(const Lpoint& p, const size_t n, float& radius, const float minRadius,
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
