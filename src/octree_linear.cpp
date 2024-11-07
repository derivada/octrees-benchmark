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

std::vector<std::pair<Point, size_t>> LinearOctree::computeNumPoints() const
/**
 * @brief Returns a vector containing the number of points of all populated octants
 * @param numPoints
 */
{
	return std::vector<std::pair<Point, size_t>>{};
}

std::vector<std::pair<Point, double>> LinearOctree::computeDensities() const
/*
 * Returns a vector containing the densities of all populated octrees
 */
{
	return std::vector<std::pair<Point, double>>{};
}

void LinearOctree::writeDensities(const std::filesystem::path& path) const
/**
 * @brief Compute and write to file the density of each non-empty octan of a given octree.
 * @param path
 */
{

}

void LinearOctree::writeNumPoints(const std::filesystem::path& path) const
/**
 * @brief Compute and write to file the density of each non-empty octan of a given octree.
 * @param path
 */
{

}

/* // FIXME: This function may overlap with some parts of extractPoint[s]
const LinearOctree* LinearOctree::findOctant(const Lpoint* p) const
/**
 * @brief Find the octant containing a given point.
 * @param p
 * @return
 */


std::vector<Lpoint*> LinearOctree::KNN(const Point& p, const size_t k, const size_t maxNeighs) const
/**
 * @brief KNN algorithm. Returns the min(k, maxNeighs) nearest neighbors of a given point p
 * @param p
 * @param k
 * @param maxNeighs
 * @return
 */
{
	return std::vector<Lpoint*>{};
}

void LinearOctree::writeOctree(std::ofstream& f, size_t index) const
{

}

void LinearOctree::extractPoint(const Lpoint* p)
/**
 * Searches for p and (if found) removes it from the octree.
 *
 * @param p
 */
{
	
}

Lpoint* LinearOctree::extractPoint()
/**
 * Searches for a point and, if it founds one, removes it from the octree.
 *
 * @return pointer to one of the octree's points, or nullptr if the octree is empty
 */
{
	return nullptr;
}

void LinearOctree::extractPoints(std::vector<Lpoint>& points)
{
	for (Lpoint& p : points)
	{
		extractPoint(&p);
	}
}

void LinearOctree::extractPoints(std::vector<Lpoint*>& points)
{
	for (Lpoint* p : points)
	{
		extractPoint(p);
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
	/* TODO because we dont have radius_
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
	return closeNeighbors; */
	return std::vector<Lpoint*>{};
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
