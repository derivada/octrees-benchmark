//
// Created by Pablo Díaz Viñambres on 07/11/2024.
//

#include "octree_linear.hpp"
#include "octree_linear_node.hpp"

#include "Box.hpp"
#include "NeighborKernels/KernelFactory.hpp"

#include <algorithm>
#include <unordered_map>
#include <filesystem>
#include <bits/types.h>
#include <fstream>

std::vector<std::pair<Point, size_t>> LinearOctree::computeNumPoints() const
/**
 * @brief Returns a vector containing the number of points of all populated octants
 * @param numPoints
 */
{
	std::vector<std::pair<Point, size_t>> result;
	int total = 0;
	for (auto& [code, node] : nodes) {
		result.push_back({getNodeCenter(code), node->points.size()});
	}

	return result;
}

std::vector<std::pair<Point, double>> LinearOctree::computeDensities() const
/*
 * Returns a vector containing the densities of all populated octrees
 */
{
	std::vector<std::pair<Point, double>> result;
	int total = 0;
	for (auto& [code, node] : nodes) {
		result.push_back({getNodeCenter(code), getDensity(code)});
	}

	return result;
}

void LinearOctree::writeDensities(const std::filesystem::path& path) const
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

void LinearOctree::writeNumPoints(const std::filesystem::path& path) const
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

const LinearOctreeNode* LinearOctree::findNode(const Lpoint* p) const
/**
 * @brief Find the octant containing a given point.
 * @param p
 * @return
 */
{
	morton_t code = 0;
	bool not_found_flag;
	while(!isLeaf(code) && getDepth(code) <= MAX_DEPTH) {
		not_found_flag = true;
		// TODO: use octantIdx
		for(uint8_t index = 0; index < 8; index++) {
			morton_t childCode = getChildrenCode(code, index);
			// If the node point is inside, go into the leaf
			if(isInside(*p, childCode)) {
				code = childCode;
				not_found_flag = false;
			}
		}
		if(not_found_flag)
			break;
	}
	
	if(not_found_flag || !isInside(*p, code)) {    
		return nullptr;
	} else {
		return nodes.at(code);
	}
}

inline uint8_t LinearOctree::octantIdx(const Lpoint* p, morton_t code) const
{
	for(int index = 0; index<OCTANTS_PER_NODE; index++) {
		if(isInside(*p, getChildrenCode(code, index))) {
			return index;
		}
	}
	return 0;
}

std::vector<Lpoint*> LinearOctree::KNN(const Point& p, const size_t k, const size_t maxNeighs) const
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
	const double rMax = radii.getMaxCoordinate(); // Use maximum radius as upper bound

	while (knn.size() <= nmax && r <= rMax)
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
		r *= 2;
	}
	return knn;
}

void LinearOctree::writeOctree(std::ofstream& f, morton_t code = 0) const
{
	if(!isNode(code)) return;
	LinearOctreeNode* node = nodes.at(code);

	f << "Depth: " << static_cast<int>(getDepth(code)) << " "
	  << "numPoints: " << node->points.size() << "\n";
	f << "Center: " << getNodeCenter(code) << " Radius: " << getNodeRadii(code) << "\n";

	if (isLeaf(code))
	{
		for (const auto& p : node->points)
		{
			f << "\t " << *p << " " << p->getClass() << " at: " << p << "\n";
		}
	}
	else
	{
		for(int index = 0; index < OCTANTS_PER_NODE; index++) {
			morton_t childCode = getChildrenCode(code, index);
			if(isNode(childCode))
				writeOctree(f, childCode);
		}
	}
}

void LinearOctree::extractPoint(const Lpoint* p, morton_t code)
/**
 * Searches for p and (if found) removes it from the octree.
 *
 * @param p
 */
{
	// TODO: make this algorithm non-recursive
	if(!isNode(code)) return;
	LinearOctreeNode* node = nodes.at(code);
	if (isLeaf(code))
	{
		// Find point with the same ID inside points node and remove it
		// TODO: could sort points inside a leaf by ID and use binary search, locality shouldn't be affected too much since
		// we would be at the finest level of the space filling curve
		std::cout << "leaf size before: " << node->points.size() << std::endl;
		auto new_end = std::remove_if(node->points.begin(), node->points.end(), [&](const Lpoint* point_ptr) {
			return point_ptr->id() == p->id();
		});
		node->points.erase(new_end, node->points.end());

		std::cout << "leaf size after: " << node->points.size() << std::endl;
		// Delete the node if all of its points were erased
		if(node->points.empty()) {
			delete node;
			nodes.erase(code);
		}
	}
	else
	{
		uint8_t index = octantIdx(p, code);
		std::cout << index << std::endl;
		morton_t childCode = getChildrenCode(code, index);
		extractPoint(p, childCode);
	}
}

Lpoint* LinearOctree::extractPoint(morton_t code)
/**
 * Searches for a point and, if it founds one, removes it from the octree.
 *
 * @return pointer to one of the octree's points, or nullptr if the octree is empty
 */
{
	// TODO: make this algorithm non-recursive
	if(!isNode(code)) return nullptr;
	LinearOctreeNode* node = nodes.at(code);
	
	if (isLeaf(code))
	{
		auto* p = node->points.back();
		node->points.pop_back();
		// Delete the node if all of its points were erased
		if(node->points.empty()) {
			delete node;
			nodes.erase(code);
		}
		return p;
	}

	for (uint8_t index = 0; index < OCTANTS_PER_NODE; index++)
	{
		morton_t childCode = getChildrenCode(code, index);
		if(isNode(childCode)) {
			auto* p = extractPoint(childCode);
			return p;
		}
	}

	std::cerr << "Warning: Found octree with 8 empty octants\n";
	return nullptr;
}

void LinearOctree::extractPoints(std::vector<Lpoint>& points)
{
	for (Lpoint& p : points)
	{
		extractPoint(&p, 0);
	}
}

void LinearOctree::extractPoints(std::vector<Lpoint*>& points)
{
	for (Lpoint* p : points)
	{
		extractPoint(p, 0);
	}
}

std::vector<Lpoint*> LinearOctree::searchEraseCircleNeighbors(const std::vector<Lpoint*>& points, double radius)
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

std::vector<Lpoint*> LinearOctree::searchEraseSphereNeighbors(const std::vector<Lpoint*>& points, float radius)
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
std::vector<Lpoint*> LinearOctree::searchConnectedShellNeighbors(const Point& point, const float nextDoorDistance,
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
std::vector<Lpoint*> LinearOctree::searchEraseConnectedCircleNeighbors(const float nextDoorDistance)
{
	std::vector<Lpoint*> connectedCircleNeighbors;

	auto* p = extractPoint(0);
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

std::vector<Lpoint*> LinearOctree::connectedNeighbors(const Point* point, std::vector<Lpoint*>& neighbors,
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

std::vector<Lpoint*> LinearOctree::extractCloseNeighbors(const Point* p, std::vector<Lpoint*>& neighbors, const float radius)
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

std::vector<Lpoint*> LinearOctree::kClosestCircleNeighbors(const Lpoint* p, const size_t k) const
/**
	 * Fetches the (up to if not enough points in octree) k closest neighbors with respect to 2D-distance.
	 *
	 * @param p
	 * @param k
	 * @return
	 */
{
	double               rMin = SENSEPSILON * static_cast<double>(k);
	const double         rMax = 2.0 * M_SQRT2 * std::max(radii.getX(), std::max(radii.getY(), radii.getZ()));
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

std::vector<Lpoint*> LinearOctree::nCircleNeighbors(const Lpoint* p, const size_t n, float& radius, const float minRadius,
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

std::vector<Lpoint*> LinearOctree::nSphereNeighbors(const Lpoint& p, const size_t n, float& radius, const float minRadius,
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
