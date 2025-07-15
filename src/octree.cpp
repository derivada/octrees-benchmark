//
// Created by miguel.yermo on 5/03/20.
//

#include "octree.hpp"

#include "Geometry/Box.hpp"
#include "NeighborKernels/KernelFactory.hpp"
#include <algorithm>
#include <unordered_map>
#include "Geometry/Lpoint.hpp"
#include "main_options.hpp"

template class Octree<PointsAoS>;
template class Octree<PointsSoA>;

template <PointContainer Container>
Octree<Container>::Octree(Container& container, Box box)
    : container_(container), center_(box.center()), radii_(box.radii())
{
    octants_.reserve(OCTANTS_PER_NODE);
    std::vector<size_t> allIndices(container.size());
    std::iota(allIndices.begin(), allIndices.end(), 0);
    buildOctree(allIndices);
}

template <PointContainer Container>
Octree<Container>::Octree(Container& container)
    : container_(container), center_(mbb(container, radii_))
{
    octants_.reserve(OCTANTS_PER_NODE);
    std::vector<size_t> allIndices(container.size());
    std::iota(allIndices.begin(), allIndices.end(), 0);
    buildOctree(allIndices);
}

template <PointContainer Container>
Octree<Container>::Octree(Container& container, const Point& center, const Vector& radii)
    : container_(container), center_(center), radii_(radii)
{
    octants_.reserve(OCTANTS_PER_NODE);
}

template <PointContainer Container>
Octree<Container>::Octree(Container& container, Point center, Vector radii, std::vector<size_t>& points)
    : container_(container), center_(center), radii_(radii)
{
    octants_.reserve(OCTANTS_PER_NODE);
    buildOctree(points);
}

/// @brief Fill the missing data in the octree log for the Pointer-based Octree 
template <PointContainer Container>
void Octree<Container>::logOctreeData(std::shared_ptr<BuildLog> log) const 
{
    std::vector<std::pair<std::reference_wrapper<const Octree>, size_t>> toVisit;
    toVisit.emplace_back(std::cref(*this), 0);

    while (!toVisit.empty())
    {
        auto [octant_ref, depth] = toVisit.back();
        toVisit.pop_back();
        const Octree& octant = octant_ref.get();

        // Add size of Octree node itself
        log->memoryUsed += sizeof(Octree);

        // For leaf nodes, count memory used by indices
        if (octant.isLeaf()) {
            log->leafNodes++;
            const auto& pointIndices = octant.getPoints();
            log->memoryUsed += sizeof(size_t) * pointIndices.size();  // each index is a size_t

            if (depth > log->maxDepthSeen) {
                log->maxDepthSeen = depth;
                auto minRadii = octant.getRadii();
                log->minRadiusAtMaxDepth = std::min({minRadii.getX(), minRadii.getY(), minRadii.getZ()});
            }
        } else {
            log->internalNodes++;
            for (const auto& child : octant.getOctants()) {
                toVisit.emplace_back(std::cref(child), depth + 1);
            }
        }
    }

    log->totalNodes = log->leafNodes + log->internalNodes;
}

template <PointContainer Container>
std::vector<std::pair<Point, size_t>> Octree<Container>::computeNumPoints() const
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

template <PointContainer Container>
std::vector<std::pair<Point, double>> Octree<Container>::computeDensities() const
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

template <PointContainer Container>
void Octree<Container>::writeDensities(const std::filesystem::path& path) const
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

template <PointContainer Container>
void Octree<Container>::writeNumPoints(const std::filesystem::path& path) const
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

// TODO
// FIXME: This function may overlap with some parts of extractPoint[s]
// template <PointContainer Container>
// const Octree<Container>* Octree<Container>::findOctant(const Point* p) const
// /**
//  * @brief Find the octant containing a given point.
//  * @param p
//  * @return
//  */
// {
// 	if (isLeaf())
// 	{
// 		for (const auto& point : points_)
// 		{
// 			auto it = std::find(points_.begin(), points_.end(), point);
// 			if (it != points_.end()) // Found
// 			{
// 				return this; // If findOctant is const, fiesta loca!!
// 			}
// 		}
// 	}
// 	else
// 	{
// 		// If Point_t is const, fiesta loca!
// 		return octants_[octantIdx(p)].findOctant(p);
// 	}

// 	return nullptr;
// }

template <PointContainer Container>
void Octree<Container>::insertPoints(std::vector<size_t>& points)
{
	for (size_t p : points)
	{
		insertPoint(p);
	}
}

template <PointContainer Container>
void Octree<Container>::insertPoint(size_t pointIndex)
{
	unsigned int idx = 0;

	if (isLeaf())
	{
		if (isEmpty()) {
			points_.emplace_back(pointIndex);
		} else {
			const auto maxRadius = std::max({radii_.getX(), radii_.getY(), radii_.getZ()});
			if (points_.size() >= mainOptions.maxPointsLeaf && maxRadius >= MIN_OCTANT_RADIUS)
			{
				createOctants();  // Create children
				fillOctants();    // Move existing points into children
				idx = octantIdx(pointIndex); // Compute correct child
				octants_[idx].insertPoint(pointIndex);       // Recurse into child
			}
			else {
				points_.emplace_back(pointIndex);
			}
		}
	}
	else
	{
		idx = octantIdx(pointIndex);
		octants_[idx].insertPoint(pointIndex);
	}
}

template <PointContainer Container>
void Octree<Container>::createOctants()
{
	for (size_t i = 0; i < OCTANTS_PER_NODE; i++)
	{
		auto newCenter = center_;
		newCenter.setX(newCenter.getX() + radii_.getX() * ((i & 4U) != 0U ? 0.5F : -0.5F));
		newCenter.setY(newCenter.getY() + radii_.getY() * ((i & 2U) != 0U ? 0.5F : -0.5F));
		newCenter.setZ(newCenter.getZ() + radii_.getZ() * ((i & 1U) != 0U ? 0.5F : -0.5F));
		Vector newRadii_(radii_.getX()*0.5F, radii_.getY()*0.5F, radii_.getZ()*0.5F);
		octants_.emplace_back(container_, newCenter, newRadii_);
	}
}

template <PointContainer Container>
void Octree<Container>::fillOctants()
{
	for (size_t p : points_)
	{
		// Idx of the octant where a point should be inserted.
		const auto idx = octantIdx(p);
		octants_[idx].insertPoint(p);
	}

	points_.clear();
}

template <PointContainer Container>
size_t Octree<Container>::octantIdx(size_t pointIndex) const
{
    const auto& p = container_[pointIndex];  // Get the point from container by index
    size_t child = 0;

    if (p.getX() >= center_.getX()) { child |= 4U; }
    if (p.getY() >= center_.getY()) { child |= 2U; }
    if (p.getZ() >= center_.getZ()) { child |= 1U; }

    return child;
}

template <PointContainer Container>
void Octree<Container>::buildOctree(std::vector<size_t>& points)
/**
   * Build the Octree
   */
{
	insertPoints(points);
}

template <PointContainer Container>
std::vector<size_t> Octree<Container>::KNN(const Point& p, const size_t k, const size_t maxNeighs) const
/**
 * @brief KNN algorithm. Returns the min(k, maxNeighs) nearest neighbors of a given point p
 * @param p
 * @param k
 * @param maxNeighs
 * @return
 */
{
	std::vector<size_t>             knn{};
	std::unordered_map<size_t, bool> wasAdded{};

	double r = 1.0;

	size_t nmax = std::min(k, maxNeighs);

	while (knn.size() <= nmax)
	{
		auto neighs = searchNeighbors<Kernel_t::sphere>(p, r);

		if (knn.size() + neighs.size() > nmax)
		{ // Add all points if there is room for them
			std::sort(neighs.begin(), neighs.end(),
			          [&](size_t a, size_t b) { return container_[a].distance3D(p) < container_[b].distance3D(p); });
		}

		for (const size_t p : neighs)
		{
			if (!wasAdded[p])
			{
				wasAdded[p] = true;
				knn.push_back(p); // Conditional inserting?
			}
		}
		// TODO: Max radius?
		r *= 2;
	}
	return knn;
}

template <PointContainer Container>
void Octree<Container>::writeOctree(std::ofstream& f, size_t index) const
{
    // Increment depth index for this node
    index++;

    // Write current node depth and number of points in this node (leaf or not)
    f << "Depth: " << index << " numPoints: " << points_.size() << "\n";

    // Write center and radii info
    f << "Center: " << center_ << " Radii: " << radii_.getX() << ", " << radii_.getY() << ", " << radii_.getZ() << "\n";

    if (isLeaf())
    {
        // points_ stores indexes into container_, so output points via container_
        for (size_t idx : points_)
        {
            const auto& p = container_[idx]; // get actual point reference from container_

            // Assuming operator<< is defined for your Point type
            f << "\t " << p << " at index: " << idx << "\n";
        }
    }
    else
    {
        // Recursively write each child octant
        for (const auto& octant : octants_)
        {
            octant.writeOctree(f, index);
        }
    }
}

// TODO
// template <PointContainer Container>
// void Octree<Container>::extractPoint(const Point_t* p)
// /**
//  * Searches for p and (if found) removes it from the octree.
//  *
//  * @param p
//  */
// {
// 	unsigned int idx = 0;

// 	if (isLeaf())
// 	{
// 		auto index = std::find(points_.begin(), points_.end(), p);
// 		if (index != points_.end()) { points_.erase(index); }
// 	}
// 	else
// 	{
// 		idx = octantIdx(p);
// 		octants_[idx].extractPoint(p);
// 		if (octants_[idx].isLeaf() && octants_[idx].isEmpty())
// 		// Leaf has been emptied. Check if all octants are empty leaves, and clear octants_ if so
// 		{
// 			bool emptyLeaves = true;
// 			for (size_t i = 0; emptyLeaves && i < OCTANTS_PER_NODE; i++)
// 			{
// 				emptyLeaves = octants_[i].isLeaf() && octants_[i].isEmpty();
// 			}
// 			if (emptyLeaves) { octants_.clear(); }
// 		}
// 	}
// }

// template <PointContainer Container>
// Point_t* Octree<Container>::extractPoint()
// /**
//  * Searches for a point and, if it founds one, removes it from the octree.
//  *
//  * @return pointer to one of the octree's points, or nullptr if the octree is empty
//  */
// {
// 	if (isLeaf())
// 	{
// 		if (points_.empty()) { return nullptr; }

// 		auto* p = points_.back();
// 		points_.pop_back();
// 		return p;
// 	}

// 	int nonEmptyOctantId = -1;
// 	int i                = 0;
// 	for (const auto& octant : octants_)
// 	{
// 		if (!octants_[i].isLeaf() || !octants_[i].isEmpty())
// 		{
// 			nonEmptyOctantId = i;
// 			break;
// 		}
// 		i++;
// 	}

// 	if (nonEmptyOctantId == -1)
// 	{
// 		std::cerr << "Warning: Found octree with 8 empty octants\n";
// 		return nullptr;
// 	}

// 	auto* p = octants_[nonEmptyOctantId].extractPoint();

// 	if (octants_[nonEmptyOctantId].isLeaf() && octants_[nonEmptyOctantId].isEmpty())
// 	// Leaf has been emptied. Check if all octants are empty leaves, and clear octants_ if so
// 	{
// 		bool emptyLeaves = true;
// 		for (const auto& octant : octants_)
// 		{
// 			emptyLeaves = octants_[i].isLeaf() && octants_[i].isEmpty();
// 		}
// 		if (emptyLeaves) { octants_.clear(); }
// 	}

// 	return p;
// }

// template <PointContainer Container>
// void Octree<Container>::extractPoints(std::vector<Point_t>& points)
// {
// 	for (Point_t& p : points)
// 	{
// 		extractPoint(&p);
// 	}
// }

// template <PointContainer Container>
// void Octree<Container>::extractPoints(std::vector<Point_t*>& points)
// {
// 	for (Point_t* p : points)
// 	{
// 		extractPoint(p);
// 	}
// }

// template <PointContainer Container>
// std::vector<Point_t*> Octree<Container>::searchEraseCircleNeighbors(const std::vector<Point_t*>& points, double radius)
// /*
//  * Searches points' circle neighbors and erases them from the octree.
//  */
// {
// 	std::vector<Point_t*> pointsNeighbors{};

// 	for (const auto* p : points)
// 	{
// 		auto pNeighbors = searchCircleNeighbors(p, radius);
// 		if (!pNeighbors.empty())
// 		{
// 			extractPoints(pNeighbors);
// 			pointsNeighbors.reserve(pointsNeighbors.size() + pNeighbors.size());
// 			std::move(std::begin(pNeighbors), std::end(pNeighbors), std::back_inserter(pointsNeighbors));
// 		}
// 	}

// 	return pointsNeighbors;
// }

// template <PointContainer Container>
// std::vector<Point_t*> Octree<Container>::searchEraseSphereNeighbors(const std::vector<Point_t*>& points, float radius)
// {
// 	std::vector<Point_t*> pointsNeighbors{};

// 	for (const auto* p : points)
// 	{
// 		auto pNeighbors = searchSphereNeighbors(*p, radius);
// 		if (!pNeighbors.empty())
// 		{
// 			extractPoints(pNeighbors);
// 			pointsNeighbors.reserve(pointsNeighbors.size() + pNeighbors.size());
// 			std::move(std::begin(pNeighbors), std::end(pNeighbors), std::back_inserter(pointsNeighbors));
// 		}
// 	}

// 	return pointsNeighbors;
// }

// /** Connected inside a spherical shell*/
// template <PointContainer Container>
// std::vector<Point_t*> Octree<Container>::searchConnectedShellNeighbors(const Point& point, const float nextDoorDistance,
//                                                            const float minRadius, const float maxRadius) const
// {
// 	std::vector<Point_t*> connectedShellNeighs;

// 	auto connectedSphereNeighs = searchSphereNeighbors(point, maxRadius);
// 	connectedSphereNeighs      = connectedNeighbors(&point, connectedSphereNeighs, nextDoorDistance);
// 	for (auto* n : connectedSphereNeighs)
// 	{
// 		if (n->distance3D(point) >= minRadius) { connectedShellNeighs.push_back(n); }
// 	}

// 	return connectedShellNeighs;
// }

// /** Connected circle neighbors*/
// template <PointContainer Container>
// std::vector<Point_t*> Octree<Container>::searchEraseConnectedCircleNeighbors(const float nextDoorDistance)
// {
// 	std::vector<Point_t*> connectedCircleNeighbors;

// 	auto* p = extractPoint();
// 	if (p == nullptr) { return connectedCircleNeighbors; }
// 	connectedCircleNeighbors.push_back(p);

// 	auto closeNeighbors = searchEraseCircleNeighbors(std::vector<Point_t*>{ p }, nextDoorDistance);
// 	while (!closeNeighbors.empty())
// 	{
// 		connectedCircleNeighbors.insert(connectedCircleNeighbors.end(), closeNeighbors.begin(), closeNeighbors.end());
// 		closeNeighbors = searchEraseCircleNeighbors(closeNeighbors, nextDoorDistance);
// 	}

// 	return connectedCircleNeighbors;
// }

// template <PointContainer Container>
// std::vector<Point_t*> Octree<Container>::connectedNeighbors(const Point* point, std::vector<Point_t*>& neighbors,
//                                                 const float nextDoorDistance)
// /**
// 	 * Filters neighbors which are not connected to point through a chain of next-door neighbors. Erases neighbors in the
// 	 * process.
// 	 *
// 	 * @param point
// 	 * @param neighbors
// 	 * @param radius
// 	 * @return
// 	 */
// {
// 	std::vector<Point_t*> connectedNeighbors;
// 	if (neighbors.empty()) { return connectedNeighbors; }

// 	auto waiting = extractCloseNeighbors(point, neighbors, nextDoorDistance);

// 	while (!waiting.empty())
// 	{
// 		auto* v = waiting.back();
// 		waiting.pop_back();
// 		auto vCloseNeighbors = extractCloseNeighbors(v, neighbors, nextDoorDistance);
// 		waiting.insert(waiting.end(), vCloseNeighbors.begin(), vCloseNeighbors.end());

// 		connectedNeighbors.push_back(v);
// 	}

// 	return connectedNeighbors;
// }

// template <PointContainer Container>
// std::vector<Point_t*> Octree<Container>::extractCloseNeighbors(const Point* p, std::vector<Point_t*>& neighbors, const float radius)
// /**
// 	 * Fetches neighbors within radius from p, erasing them from neighbors and returning them.
// 	 *
// 	 * @param p
// 	 * @param neighbors
// 	 * @param radius
// 	 * @return
// 	 */
// {
// 	std::vector<Point_t*> closeNeighbors;

// 	for (size_t i = 0; i < neighbors.size();)
// 	{
// 		if (neighbors[i]->distance3D(*p) <= radius)
// 		{
// 			closeNeighbors.push_back(neighbors[i]);
// 			neighbors.erase(neighbors.begin() + i);
// 		}
// 		else { i++; }
// 	}

// 	return closeNeighbors;
// }

// template <PointContainer Container>
// std::vector<Point_t*> Octree<Container>::kClosestCircleNeighbors(const Point_t* p, const size_t k) const
// /**
// 	 * Fetches the (up to if not enough points in octree) k closest neighbors with respect to 2D-distance.
// 	 *
// 	 * @param p
// 	 * @param k
// 	 * @return
// 	 */
// {
// 	double               rMin = SENSEPSILON * static_cast<double>(k);
// 	const double         rMax = 2.0 * M_SQRT2 * std::max(radii_.getX(), std::max(radii_.getY(), radii_.getZ()));
// 	std::vector<Point_t*> closeNeighbors;
// 	for (closeNeighbors = searchCircleNeighbors(p, rMin); closeNeighbors.size() < k && rMin < 2 * rMax; rMin *= 2)
// 	{
// 		closeNeighbors = searchCircleNeighbors(p, rMin);
// 	}

// 	while (closeNeighbors.size() > k)
// 	{
// 		size_t furthestIndex;
// 		double furthestDistanceSquared = 0.0;
// 		for (size_t i = 0; i < closeNeighbors.size(); i++)
// 		{
// 			const auto iDistanceSquared = p->distance2Dsquared(*closeNeighbors[i]);
// 			if (iDistanceSquared > furthestDistanceSquared)
// 			{
// 				furthestDistanceSquared = iDistanceSquared;
// 				furthestIndex           = i;
// 			}
// 		}
// 		closeNeighbors.erase(closeNeighbors.begin() + furthestIndex);
// 	}
// 	return closeNeighbors;
// }

// template <PointContainer Container>
// std::vector<Point_t*> Octree<Container>::nCircleNeighbors(const Point_t* p, const size_t n, float& radius, const float minRadius,
//                                               const float maxRadius, const float maxIncrement,
//                                               const float maxDecrement) const
// /**
// 	 * Radius-adaptive search method for circle neighbors.
// 	 *
// 	 * @param p
// 	 * @param n
// 	 * @param radius
// 	 * @param minRadius
// 	 * @param maxRadius
// 	 * @param maxStep
// 	 * @return circle neighbors
// 	 */
// {
// 	auto neighs = searchCircleNeighbors(p, radius);

// 	float radiusOffset = (float(n) - float(neighs.size())) * SENSEPSILON;
// 	if (radiusOffset > maxIncrement) { radiusOffset = maxIncrement; }
// 	else if (radiusOffset < -maxDecrement) { radiusOffset = -maxDecrement; }

// 	radius += radiusOffset;
// 	if (radius > maxRadius) { radius = maxRadius; }
// 	else if (radius < minRadius) { radius = minRadius; }

// 	return neighs;
// }

// template <PointContainer Container>
// std::vector<Point_t*> Octree<Container>::nSphereNeighbors(const Point_t& p, const size_t n, float& radius, const float minRadius,
//                                               const float maxRadius, const float maxStep) const
// /**
// 	 * Radius-adaptive search method for sphere neighbors.
// 	 *
// 	 * @param p
// 	 * @param n
// 	 * @param radius
// 	 * @param minRadius
// 	 * @param maxRadius
// 	 * @param maxStep
// 	 * @return sphere neighbors
// 	 */
// {
// 	auto neighs = searchSphereNeighbors(p, radius);

// 	float radiusOffset = (float(n) - float(neighs.size())) * SENSEPSILON;
// 	if (radiusOffset > maxStep) { radiusOffset = maxStep; }
// 	else if (radiusOffset < -maxStep) { radiusOffset = -maxStep; }

// 	radius += radiusOffset;
// 	if (radius > maxRadius) { radius = maxRadius; }
// 	else if (radius < minRadius) { radius = minRadius; }

// 	return neighs;
// }
