#pragma once
#include <vector>
#include "Geometry/point.hpp"
#include <random>

struct SearchSet {
    const size_t numSearches;
    std::vector<Point> searchPoints;
    std::vector<uint32_t> searchKNNLimits;
    constexpr static uint32_t MIN_KNN = 5;
    constexpr static uint32_t MAX_KNN = 100;
    const bool isSequential;
    std::mt19937 rng;

    // Random subset of size numSearches (may have repeated points)
    template <PointType Point_t>
    SearchSet(size_t numSearches, const std::vector<Point_t>& points, bool sequential = false)
        : numSearches(numSearches), isSequential(sequential) {
        rng.seed(42);
        searchPoints.resize(numSearches);
        searchKNNLimits.resize(numSearches);
        std::uniform_int_distribution<size_t> knnDist(MIN_KNN, MAX_KNN);
        if(sequential) {
            std::uniform_int_distribution<size_t> startIndexDist(0, points.size() - numSearches);
            size_t startIndex = startIndexDist(rng);
            for (size_t i = 0; i < numSearches; ++i) {
                searchPoints[i] = points[startIndexDist(rng) + i];
                searchKNNLimits[i] = knnDist(rng);
            }
        } else {
            std::uniform_int_distribution<size_t> indexDist(0, points.size() - 1);
            for (size_t i = 0; i < numSearches; ++i) {
                searchPoints[i] = points[indexDist(rng)];
                searchKNNLimits[i] = knnDist(rng);
            }
        }
    }
};