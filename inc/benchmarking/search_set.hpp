#pragma once

#include <random>
#include <vector>
#include "geometry/point.hpp"
#include "main_options.hpp"

struct SearchSet {
    std::vector<std::vector<size_t>> searchPoints;
    int currentRepeat;
    const size_t numSearches, cloudSize;
    bool sequential;

    /**
     * @brief Construct a vector of n indexes that can be either sequential or randomly sampled.
     * 
     * @param n The size of the searchSet to be contructed
     * @param points A reference to the array of points (not stored for now)
     * @param seq Whether to choose a sequential slice of points or sample random points
     * @param all Whether to have all the indexes on the searchSet. If set to true, then n=cloudSize and seq=True
     * Only use all in small clouds and with a reasonable radius, since otherwise it will take forever to run the neighborSearches!
     */
    SearchSet(size_t n, size_t cloudSize, bool seq = mainOptions.sequentialSearches, 
            bool all = mainOptions.searchAll)
        : numSearches(all ? cloudSize : n), cloudSize(cloudSize), sequential(seq || all), currentRepeat(0) {
        std::mt19937 rng;
        rng.seed(42);
        searchPoints.resize(mainOptions.repeats);
        for(int repeat = 0; repeat < mainOptions.repeats; repeat++) {
            searchPoints[repeat].reserve(numSearches);
            if(sequential) {
                std::uniform_int_distribution<size_t> startIndexDist(0, cloudSize - numSearches);
                size_t startIndex = startIndexDist(rng);
                for (size_t i = 0; i < numSearches; ++i) {
                    searchPoints[repeat].push_back(startIndex + i);
                }
            } else {
                std::uniform_int_distribution<size_t> indexDist(0, cloudSize - 1);
                for (size_t i = 0; i < numSearches; ++i) {
                    searchPoints[repeat].push_back(indexDist(rng));
                }
            }
        }
    }

    void nextRepeat() {
        currentRepeat += 1;
        currentRepeat %= searchPoints.size();
    }

    void reset() {
        currentRepeat = 0;
    }
};