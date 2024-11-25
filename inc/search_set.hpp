#pragma once

#include <vector>
#include <random>
#include "Lpoint.hpp"

class SearchSet {
    public: 
        const size_t numSearches;
        std::vector<size_t> searchPointIndexes;
        std::vector<uint32_t> searchKNNLimits;
        constexpr static uint32_t MIN_KNN = 5;
        constexpr static uint32_t MAX_KNN = 100;
        std::mt19937 rng;

        SearchSet(size_t numSearches, std::vector<Lpoint> &points): numSearches(numSearches) {
            rng.seed(42);
            searchPointIndexes.resize(numSearches);
            searchKNNLimits.resize(numSearches);
            std::uniform_int_distribution<size_t> indexDist(0, points.size()-1);
            std::uniform_int_distribution<size_t> knnDist(MIN_KNN, MAX_KNN);
            
            for(int i = 0; i<numSearches; i++) {
                searchPointIndexes[i] = indexDist(rng);
            }
        }
};