#pragma once

#include "benchmarking.hpp"
#include <random>
#include <omp.h>
#include <set>
#include "NeighborKernels/KernelFactory.hpp"
#include "type_names.hpp"
#include "TimeWatcher.hpp"
#include "neigh_search_result.hpp"
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

template <PointType Point_t>
struct ResultSet {
    const std::shared_ptr<const SearchSet> searchSet;
    std::vector<std::vector<Point_t*>> resultsNeigh;
    std::vector<NeighSearchResult<Point_t>> resultsNeighStruct;
    std::vector<std::vector<Point_t>> resultsNeighCopy;
    std::vector<std::vector<Point_t*>> resultsNeighOld;
    std::vector<size_t> resultsNumNeigh;
    std::vector<size_t> resultsNumNeighOld;
    std::vector<std::vector<Point_t*>> resultsKNN;
    std::vector<std::vector<Point_t*>> resultsRingNeigh;
    std::vector<NeighSearchResult<Point_t>> resultsSearchApproxUpper;
    std::vector<NeighSearchResult<Point_t>> resultsSearchApproxLower;
    double tolerancePercentageUsed;
    
    ResultSet(const std::shared_ptr<const SearchSet> searchSet): searchSet(searchSet) {  }

    // Generic check for neighbor results
    std::vector<size_t> checkNeighResults(std::vector<std::vector<Point_t*>> &results1, std::vector<std::vector<Point_t*>> &results2)
    {
        std::vector<size_t> wrongSearches;
        for (size_t i = 0; i < results1.size(); i++) {
            auto v1 = results1[i];
            auto v2 = results2[i];
            if (v1.size() != v2.size()) {
                wrongSearches.push_back(i);
            } else {
                std::sort(v1.begin(), v1.end(), [](Point_t *p, Point_t* q) -> bool {
                    return p->id() < q->id();
                });
                std::sort(v2.begin(), v2.end(), [](Point_t *p, Point_t* q) -> bool {
                    return p->id() < q->id();
                });
                for (size_t j = 0; j < v1.size(); j++) {
                    if (v1[j]->id() != v2[j]->id()) {
                        wrongSearches.push_back(i);
                        break;
                    }
                }
            }
        }
        return wrongSearches;
    }

    std::vector<size_t> checkNeighVsStructResults(
        const std::vector<std::vector<Point_t*>>& results1, 
        const std::vector<NeighSearchResult<Point_t>>& results2)
    {
        std::vector<size_t> wrongSearches;
        
        for (size_t i = 0; i < results1.size() && i < results2.size(); i++) {
            // Get the original vector of pointers
            auto v1 = results1[i];
            
            // Extract IDs from the first vector
            std::vector<int> ids1;
            ids1.reserve(v1.size());
            for (const auto* ptr : v1) {
                ids1.push_back(ptr->id());
            }
            
            // Extract IDs from the NeighSearchResult using its iterator
            std::vector<int> ids2;
            for (const auto& point : results2[i]) {
                ids2.push_back(point.id());
            }
            
            // Check sizes first
            if (ids1.size() != ids2.size()) {
                wrongSearches.push_back(i);
                continue;
            }
            
            // Sort both ID vectors for comparison
            std::sort(ids1.begin(), ids1.end());
            std::sort(ids2.begin(), ids2.end());
            
            // Compare the sorted ID lists
            if (!std::equal(ids1.begin(), ids1.end(), ids2.begin())) {
                wrongSearches.push_back(i);
            }
        }
        
        // If vector sizes differ, mark all extra indices as wrong
        if (results1.size() != results2.size()) {
            for (size_t i = std::min(results1.size(), results2.size()); 
                 i < std::max(results1.size(), results2.size()); i++) {
                wrongSearches.push_back(i);
            }
        }
        
        return wrongSearches;
    }

    // Generic check for the number of neighbors
    std::vector<size_t> checkNumNeighResults(std::vector<size_t> &results1, std::vector<size_t> &results2)
    {
        std::vector<size_t> wrongSearches;
        for (size_t i = 0; i < results1.size(); i++) {
            auto n1 = results1[i];
            auto n2 = results2[i];
            if (n1 != n2) {
                wrongSearches.push_back(i);
            }
        }
        return wrongSearches;
    }

    // Generic check for neighbor results
    std::vector<size_t> checkNeighResultsPtrVsCopy(std::vector<std::vector<Point_t*>> &resultsPtr, std::vector<std::vector<Point_t>> &resultsCopy)
    {
        std::vector<size_t> wrongSearches;
        for (size_t i = 0; i < resultsPtr.size(); i++) {
            auto vPtr = resultsPtr[i];
            auto vCopy = resultsCopy[i];
            if (vPtr.size() != vCopy.size()) {
                wrongSearches.push_back(i);
            } else {
                std::sort(vPtr.begin(), vPtr.end(), [](Point_t *p, Point_t* q) -> bool {
                    return p->id() < q->id();
                });
                std::sort(vCopy.begin(), vCopy.end(), [](Point_t p, Point_t q) -> bool {
                    return p.id() < q.id();
                });
                for (size_t j = 0; j < vPtr.size(); j++) {
                    if (vPtr[j]->id() != vCopy[j].id()) {
                        wrongSearches.push_back(i);
                        break;
                    }
                }
            }
        }
        return wrongSearches;
    }

    // Operation for checking neighbor results
    void checkOperationNeigh(
        std::vector<std::vector<Point_t*>> &results1,
        std::vector<std::vector<Point_t*>> &results2,
        size_t printingLimit = 10)
    {
        std::vector<size_t> wrongSearches = checkNeighResults(results1, results2);
        if (wrongSearches.size() > 0) {
            std::cout << "Wrong results at " << wrongSearches.size() << " search sets" << std::endl;
            for (size_t i = 0; i < std::min(wrongSearches.size(), printingLimit); i++) {
                size_t idx = wrongSearches[i];
                size_t nPoints1 = results1[idx].size(), nPoints2 = results2[idx].size();
                std::cout << "\tAt set " << idx << " with "
                        << nPoints1 << " VS " << nPoints2 << " points found" << std::endl;
            }

            if (wrongSearches.size() > printingLimit) {
                std::cout << "\tAnd at " << (wrongSearches.size() - printingLimit) << " other search instances..." << std::endl;
            }
        } else {
            std::cout << "All results are right!" << std::endl;
        }
    }

    void checkOperationNeighVsStruct(
        std::vector<std::vector<Point_t*>>& results, 
        std::vector<NeighSearchResult<Point_t>>& resultsStruct,
        size_t printingLimit = 10) 
    {
        std::vector<size_t> wrongSearches = checkNeighVsStructResults(results, resultsStruct);
        if (wrongSearches.size() > 0) {
            std::cout << "Wrong results at " << wrongSearches.size() << " search sets" << std::endl;
            for (size_t i = 0; i < std::min(wrongSearches.size(), printingLimit); i++) {
                size_t idx = wrongSearches[i];
                size_t nPoints1 = results[idx].size(), nPoints2 = resultsStruct[idx].size();
                std::cout << "\tAt set " << idx << " with "
                        << nPoints1 << " VS " << nPoints2 << " points found" << std::endl;
            }

            if (wrongSearches.size() > printingLimit) {
                std::cout << "\tAnd at " << (wrongSearches.size() - printingLimit) << " other search instances..." << std::endl;
            }
        } else {
            std::cout << "All results are right!" << std::endl;
        }
    }

    // Operation for checking number of neighbor results
    void checkOperationNumNeigh(
        std::vector<size_t> &results1,
        std::vector<size_t> &results2,
        size_t printingLimit = 10)
    {
        std::vector<size_t> wrongSearches = checkNumNeighResults(results1, results2);
        if (wrongSearches.size() > 0) {
            std::cout << "Wrong results at " << wrongSearches.size() << " search sets" << std::endl;
            for (size_t i = 0; i < std::min(wrongSearches.size(), printingLimit); i++) {
                size_t idx = wrongSearches[i];
                size_t nPoints1 = results1[idx], nPoints2 = results2[idx];
                std::cout << "\tAt set " << idx << " with "
                        << nPoints1 << " VS " << nPoints2 << " points found" << std::endl;
            }

            if (wrongSearches.size() > printingLimit) {
                std::cout << "\tAnd at " << (wrongSearches.size() - printingLimit) << " other search instances..." << std::endl;
            }
        } else {
            std::cout << "All results are right!" << std::endl;
        }
    }

    // Operation for checking neighbor results
    void checkOperationNeighPtrVsCopy(
        std::vector<std::vector<Point_t*>> &results1,
        std::vector<std::vector<Point_t>> &results2,
        size_t printingLimit = 10)
    {
        std::vector<size_t> wrongSearches = checkNeighResultsPtrVsCopy(results1, results2);
        if (wrongSearches.size() > 0) {
            std::cout << "Wrong results at " << wrongSearches.size() << " search sets" << std::endl;
            for (size_t i = 0; i < std::min(wrongSearches.size(), printingLimit); i++) {
                size_t idx = wrongSearches[i];
                size_t nPoints1 = results1[idx].size(), nPoints2 = results2[idx].size();
                std::cout << "\tAt set " << idx << " with "
                        << nPoints1 << " VS " << nPoints2 << " points found" << std::endl;
            }

            if (wrongSearches.size() > printingLimit) {
                std::cout << "\tAnd at " << (wrongSearches.size() - printingLimit) << " other search instances..." << std::endl;
            }
        } else {
            std::cout << "All results are right!" << std::endl;
        }
    }

    void checkResults(std::shared_ptr<ResultSet<Point_t>> other, size_t printingLimit = 10) {
        // Ensure the search sets are the same
        assert(searchSet == other->searchSet && "The search sets of the benchmarks are not the same");
        
        // Check neighbor search results if available
        if (!resultsNeigh.empty() && !other->resultsNeigh.empty()) {
            std::cout << "Checking search results for neighbor searches..." << std::endl;
            checkOperationNeigh(resultsNeigh, other->resultsNeigh, printingLimit);
        }

        // Check number of neighbor search results if available
        if (!resultsNumNeigh.empty() && !other->resultsNumNeigh.empty()) {
            std::cout << "Checking search results for number of neighbor searches..." << std::endl;
            checkOperationNumNeigh(resultsNumNeigh, other->resultsNumNeigh, printingLimit);
        }
    }

    void checkResultsAlgo(size_t printingLimit = 10) {
        if(!resultsNeigh.empty() && !resultsNeighOld.empty()) {
            std::cout << "Checking search results on both implementations of neighbor searches..." << std::endl;
            checkOperationNeigh(resultsNeigh, resultsNeighOld, printingLimit);
        }
        if(!resultsNumNeigh.empty() && !resultsNumNeighOld.empty()) {
            std::cout << "Checking search results on both implementations of number of neighbors searches..." << std::endl;
            checkOperationNumNeigh(resultsNumNeigh, resultsNumNeighOld, printingLimit);
        }
    }

    void checkResultsPtrVsCopy(size_t printingLimit = 10) { 
        if(!resultsNeigh.empty() && !resultsNeighCopy.empty()) {
            std::cout << "Checking search results on pointer result vs copy result variants..." << std::endl;
            checkOperationNeighPtrVsCopy(resultsNeigh, resultsNeighCopy, printingLimit);
            TimeWatcher twPtr, twCpy;
            twPtr.start();
            double acc = 0.0;
            auto startPtr = std::chrono::high_resolution_clock::now();
            for (const auto& neighVec : resultsNeigh) {
                for (const auto* point : neighVec) {
                    acc += point->getX();
                }
            }
            auto endPtr = std::chrono::high_resolution_clock::now();
            std::cout << acc << std::endl;
            
            acc = 0.0;
            auto startCpy = std::chrono::high_resolution_clock::now();
            for (const auto& neighVec : resultsNeighCopy) {
                for (const auto& point : neighVec) {
                    acc += point.getX();
                }
            }
            auto endCpy = std::chrono::high_resolution_clock::now();
            std::cout << acc << std::endl;
            
            auto durationPtr = std::chrono::duration_cast<std::chrono::nanoseconds>(endPtr - startPtr).count();
            auto durationCpy = std::chrono::duration_cast<std::chrono::nanoseconds>(endCpy - startCpy).count();
            
            std::cout << "time to traverse ptr: " << durationPtr << " nanoseconds\n";
            std::cout << "time to traverse cpy: " << durationCpy << " nanoseconds\n";
        }
    }
    
    void checkResultStructSearches() {
        if (resultsNeigh.empty() || resultsNeighStruct.empty()) {
            std::cout << "Ptr vs struct searches results were not computed! Not checking.\n";
            return;
        }
        std::cout << "Checking search results for neighbor searches..." << std::endl;
        checkOperationNeighVsStruct(resultsNeigh, resultsNeighStruct);
    }


    void checkResultsApproxSearches(size_t printingLimit = 10) {
        if (resultsSearchApproxLower.empty() || resultsSearchApproxUpper.empty() || resultsNeigh.empty()) {
            std::cout << "Approximate searches results were not computed! Not checking approximation results.\n";
            return;
        }
    
        size_t printingOn = std::min(searchSet->numSearches, printingLimit);
        std::cout << "Approximate searches results (printing " << printingOn 
                  << " searches of a total of " << searchSet->numSearches << " searches performed):\n";
        std::cout << "Tolerance percentage used: " << tolerancePercentageUsed << "%\n";
    
        // Column headers
        std::cout << std::left 
                  << std::setw(10) << "Search #" 
                  << std::setw(15) << "Lower bound" 
                  << std::setw(15) << "Exact search" 
                  << std::setw(15) << "Upper bound"
                  // << std::setw(20) << "Lower ⊆ Exact ⊆ Upper?"
                  << "\n";
    
        double totalDiffLower = 0.0, totalDiffUpper = 0.0;
        size_t nnzSearches = searchSet->numSearches;
    
        for (size_t i = 0; i < searchSet->numSearches; i++) {
            if (resultsNeigh[i].empty()) {
                nnzSearches--; // Avoid division by zero
                continue;
            }
    
            // Compute percentage differences
            totalDiffLower += (static_cast<double>(resultsNeigh[i].size() - resultsSearchApproxLower[i].size()) / resultsNeigh[i].size()) * 100.0;
            totalDiffUpper += (static_cast<double>(resultsSearchApproxUpper[i].size() - resultsNeigh[i].size()) / resultsNeigh[i].size()) * 100.0;
    
            // Convert exact and upper sets to unordered sets for fast lookup
            // std::set<Point_t> exactSet;
            // for (const auto* p : resultsNeigh[i]) {
            //     exactSet.insert(*p);
            // }
            // std::set<Point_t> upperSet(resultsSearchApproxUpper[i].begin(), resultsSearchApproxUpper[i].end());

            // // Check if lower ⊆ exact
            // bool lowerSubset = true;
            // for (const Point_t& p : resultsSearchApproxLower[i]) {
            //     if (exactSet.find(p) == exactSet.end()) {
            //         lowerSubset = false;
            //         std::cout << "point not in exact but in lower: " << p << std::endl;
            //         break;
            //     }
            // }
    
            // // Check if exact ⊆ upper
            // bool upperSubset = true;
            // for (const Point_t* p : resultsNeigh[i]) {
            //     if (upperSet.find(*p) == upperSet.end()) {
            //         upperSubset = false;
            //         std::cout << "point not in upper but in exact: " << *p << std::endl;
            //         break;
            //     }
            // }
    
            // std::string subsetCheck = (lowerSubset && upperSubset) ? "Yes" : "NO";
    
            if (i < printingOn) {
                std::cout << std::left 
                          << std::setw(10) << (i + 1) 
                          << std::setw(15) << resultsSearchApproxLower[i].size() 
                          << std::setw(15) << resultsNeigh[i].size() 
                          << std::setw(15) << resultsSearchApproxUpper[i].size()
                          // << std::setw(20) << subsetCheck
                          << "\n";
            }
        }
    
        std::cout << "On average over all searches done, lower bound searches found " 
                  << (totalDiffLower / nnzSearches) << "% fewer points.\n";
        std::cout << "On average over all searches done, upper bound searches found " 
                  << (totalDiffUpper / nnzSearches) << "% more points.\n";
    }    
};


template <template <typename, typename> class Octree_t, PointType Point_t, typename Encoder_t>
class OctreeBenchmark {
    private:
        const std::unique_ptr<Octree_t<Point_t, Encoder_t>> oct;
        const std::string comment;

        std::vector<Point_t>& points;
        std::ofstream &outputFile;
        
        const bool checkResults, useWarmup, useParallel;
        const std::shared_ptr<const SearchSet> searchSet;
        std::shared_ptr<ResultSet<Point_t>> resultSet;

        #pragma GCC push_options
        #pragma GCC optimize("O0")
        void preventOptimization(size_t value) {
            volatile size_t* dummy = &value;
            (void) *dummy;
        }
        #pragma GCC pop_options
        
        template<Kernel_t kernel>
        size_t searchNeigh(float radii) {
            if(checkResults && resultSet->resultsNeigh.empty())
                resultSet->resultsNeigh.resize(searchSet->numSearches);
            size_t averageResultSize = 0;
            #pragma omp parallel for if (useParallel) schedule(static) reduction(+:averageResultSize)
                for(size_t i = 0; i<searchSet->numSearches; i++) {
                    auto result = oct->template searchNeighbors<kernel>(searchSet->searchPoints[i], radii);
                    averageResultSize += result.size();
                    if(checkResults)
                        resultSet->resultsNeigh[i] = result;
                }
            
            averageResultSize = averageResultSize / searchSet->numSearches;
            return averageResultSize;
        }

        template<Kernel_t kernel>
        size_t searchNeighCopy(float radii) {
            if(checkResults && resultSet->resultsNeighCopy.empty())
                resultSet->resultsNeighCopy.resize(searchSet->numSearches);
            size_t averageResultSize = 0;
            #pragma omp parallel for if (useParallel) schedule(static) reduction(+:averageResultSize)
                for(size_t i = 0; i<searchSet->numSearches; i++) {
                    auto result = oct->template searchNeighborsCopy<kernel>(searchSet->searchPoints[i], radii);
                    averageResultSize += result.size();
                    if(checkResults)
                        resultSet->resultsNeighCopy[i] = result;
                }
            
            averageResultSize = averageResultSize / searchSet->numSearches;
            return averageResultSize;
        }

        template<Kernel_t kernel>
        size_t searchNeighStruct(float radii) {
            if(checkResults && resultSet->resultsNeighStruct.empty())
                resultSet->resultsNeighStruct.resize(searchSet->numSearches);
            size_t averageResultSize = 0;
            #pragma omp parallel for if (useParallel) schedule(static) reduction(+:averageResultSize)
                for(size_t i = 0; i<searchSet->numSearches; i++) {
                    auto result = oct->template searchNeighborsStruct<kernel>(searchSet->searchPoints[i], radii);
                    averageResultSize += result.size();
                    if(checkResults)
                        resultSet->resultsNeighStruct[i] = result;
                }
            
            averageResultSize = averageResultSize / searchSet->numSearches;
            return averageResultSize;
        }

        template<Kernel_t kernel>
        size_t searchNeighApprox(float radii, double tolerancePercentage, bool upperBound) {
            if(checkResults && upperBound && resultSet->resultsSearchApproxUpper.empty())
                resultSet->resultsSearchApproxUpper.resize(searchSet->numSearches);
            if(checkResults && !upperBound && resultSet->resultsSearchApproxLower.empty())
                resultSet->resultsSearchApproxLower.resize(searchSet->numSearches);
            resultSet->tolerancePercentageUsed = tolerancePercentage;

            size_t averageResultSize = 0;
            #pragma omp parallel for if (useParallel) schedule(static) reduction(+:averageResultSize)
                for(size_t i = 0; i<searchSet->numSearches; i++) {
                    auto result = oct->template searchNeighborsApprox<kernel>(searchSet->searchPoints[i], radii, tolerancePercentage, upperBound);
                    averageResultSize += result.size();
                    if(checkResults) {
                        if(upperBound)
                            resultSet->resultsSearchApproxUpper[i] = result;
                        else
                            resultSet->resultsSearchApproxLower[i] = result;
                    }
                }
            
            averageResultSize = averageResultSize / searchSet->numSearches;
            return averageResultSize;
        }

        template<Kernel_t kernel>
        size_t searchNeighOld(float radii) {
            if(checkResults && resultSet->resultsNeighOld.empty())
                resultSet->resultsNeighOld.resize(searchSet->numSearches);
            size_t averageResultSize = 0;
            #pragma omp parallel for if (useParallel) schedule(static) reduction(+:averageResultSize)
                for(size_t i = 0; i<searchSet->numSearches; i++) {
                    auto result = oct->template searchNeighborsOld<kernel>(searchSet->searchPoints[i], radii);
                    averageResultSize += result.size();
                    if(checkResults)
                        resultSet->resultsNeighOld[i] = result;
                }
            averageResultSize = averageResultSize / searchSet->numSearches;
            return averageResultSize;
        }

        template<Kernel_t kernel>
        size_t searchNumNeigh(float radii) {
            if(checkResults && resultSet->resultsNumNeigh.empty())
                resultSet->resultsNumNeigh.resize(searchSet->numSearches);
            size_t averageResultSize = 0;
            #pragma omp parallel for if (useParallel) schedule(static) reduction(+:averageResultSize)
                for(size_t i = 0; i<searchSet->numSearches; i++) {
                    auto result = oct->template numNeighbors<kernel>(searchSet->searchPoints[i], radii);
                    averageResultSize += result;
                    if(checkResults)
                        resultSet->resultsNumNeigh[i] = result;
                }
            averageResultSize = averageResultSize / searchSet->numSearches;
            return averageResultSize;
        }

        template<Kernel_t kernel>
        size_t searchNumNeighOld(float radii) {
            if(checkResults && resultSet->resultsNumNeighOld.empty())
                resultSet->resultsNumNeighOld.resize(searchSet->numSearches);
            size_t averageResultSize = 0;
            #pragma omp parallel for if (useParallel) schedule(static) reduction(+:averageResultSize)
                for(size_t i = 0; i<searchSet->numSearches; i++) {
                    auto result = oct->template numNeighborsOld<kernel>(searchSet->searchPoints[i], radii);
                    averageResultSize += result;
                    if(checkResults)
                        resultSet->resultsNumNeighOld[i] = result;
                }
            averageResultSize = averageResultSize / searchSet->numSearches;
            return averageResultSize;
        }

        void KNN() {
            if(checkResults && resultSet->resultsKNN.empty())
                resultSet->resultsKNN.resize(searchSet->numSearches);
            #pragma omp parallel for if (useParallel) schedule(static)
                for(size_t i = 0; i<searchSet->numSearches; i++) {
                    if(checkResults) {
                        resultSet->resultsKNN[i] = oct->template KNN(searchSet->searchPoints[i], searchSet->searchKNNLimits[i], searchSet->searchKNNLimits[i]);
                    } else {
                        preventOptimization(oct->template KNN(searchSet->searchPoints[i], searchSet->searchKNNLimits[i], searchSet->searchKNNLimits[i]));
                    }
                }
        }

        size_t searchNeighRing(Vector &innerRadii, Vector &outerRadii) {
            if(checkResults && resultSet->resultsRingNeigh.empty())
                resultSet->resultsRingNeigh.resize(searchSet->numSearches);
            size_t averageResultSize = 0;
            #pragma omp parallel for if (useParallel) schedule(static) reduction(+:averageResultSize)
                for(size_t i = 0; i<searchSet->numSearches; i++) {
                    auto result = oct->template searchNeighborsRing(searchSet->searchPoints[i], innerRadii, outerRadii);
                    averageResultSize += result.size();
                    if(checkResults)
                        resultSet->resultsRingNeigh[i] = result;
                }
            averageResultSize = averageResultSize / searchSet->numSearches;
            return averageResultSize;
        }

        inline void appendToCsv(const std::string& operation, 
                            const std::string& kernel, const float radius, const benchmarking::Stats<>& stats, size_t averageResultSize = 0, double tolerancePercentage = 0.0) {
            // Check if the file is empty and append header if it is
            if (outputFile.tellp() == 0) {
                outputFile << "date,octree,point_type,encoder,npoints,operation,kernel,radius,num_searches,repeats,accumulated,mean,median,stdev,used_warmup,warmup_time,avg_result_size,tolerance_percentage\n";
            }
            // append the comment to the octree name if needed
            std::string octreeName;
            if constexpr (std::is_same_v<Octree_t<Point_t, Encoder_t>, LinearOctree<Point_t, Encoder_t>>) {
                octreeName = "LinearOctree";
            } else if constexpr (std::is_same_v<Octree_t<Point_t, Encoder_t>, Octree<Point_t, Encoder_t>>) {
                octreeName = "Octree";
            }

            std::string pointTypeName = getPointName<Point_t>();
            std::string encoderTypename = PointEncoding::getEncoderName<Encoder_t>();
            // if the comment, exists, append it to the op. name
            std::string fullOp = operation + ((comment != "") ? "_" + comment : "");
            outputFile << getCurrentDate() << ',' 
                << octreeName << ',' 
                << pointTypeName << ','
                << encoderTypename << ','
                << points.size() << ','
                << fullOp << ',' 
                << kernel << ',' 
                << radius << ','
                << searchSet->numSearches << ',' 
                << stats.size() << ','
                << stats.accumulated() << ',' 
                << stats.mean() << ',' 
                << stats.median() << ',' 
                << stats.stdev() << ','
                << stats.usedWarmup() << ','
                << stats.warmupValue() << ','
                << averageResultSize << ','
                << tolerancePercentage 
                << std::endl;
        }

    public:
        OctreeBenchmark(std::vector<Point_t>& points, std::optional<std::vector<PointMetadata>>& metadata = std::nullopt,
            size_t numSearches = 100, std::shared_ptr<const SearchSet> searchSet = nullptr, std::ofstream &file = std::ofstream(),
            std::string comment = "", bool checkResults = false, bool useWarmup = mainOptions.useWarmup, bool useParallel = true) :
            points(points), 
            oct(std::make_unique<Octree_t<Point_t, Encoder_t>>(points, metadata)),
            searchSet(searchSet ? searchSet : std::make_shared<const SearchSet>(numSearches, points)),
            outputFile(file),
            comment(comment),
            checkResults(checkResults),
            useWarmup(useWarmup),
            useParallel(useParallel),
            resultSet(std::make_shared<ResultSet<Point_t>>(searchSet)) { }
    

        template<Kernel_t kernel>
        void benchmarkSearchNeigh(size_t repeats, float radius) {
            const auto kernelStr = kernelToString(kernel);
            auto [stats, averageResultSize] = benchmarking::benchmark<size_t>(repeats, [&]() { return searchNeigh<kernel>(radius); }, useWarmup);
            appendToCsv("neighSearch", kernelStr, radius, stats, averageResultSize);
        }

        template<Kernel_t kernel>
        void benchmarkSearchNeighCopy(size_t repeats, float radius) {
            const auto kernelStr = kernelToString(kernel);
            auto [stats, averageResultSize] = benchmarking::benchmark<size_t>(repeats, [&]() { return searchNeighCopy<kernel>(radius); }, useWarmup);
            appendToCsv("neighSearchCopy", kernelStr, radius, stats, averageResultSize);
        }

        template<Kernel_t kernel>
        void benchmarkSearchNeighStruct(size_t repeats, float radius) {
            const auto kernelStr = kernelToString(kernel);
            auto [stats, averageResultSize] = benchmarking::benchmark<size_t>(repeats, [&]() { return searchNeighStruct<kernel>(radius); }, useWarmup);
            appendToCsv("neighSearchStruct", kernelStr, radius, stats, averageResultSize);
        }

        template<Kernel_t kernel>
        void benchmarkSearchNeighApprox(size_t repeats, float radius, double tolerancePercentage, bool upperBound) {
            const auto kernelStr = kernelToString(kernel);
            auto [stats, averageResultSize] = benchmarking::benchmark<size_t>(repeats, [&]() { return searchNeighApprox<kernel>(radius, tolerancePercentage, upperBound); }, useWarmup);
            appendToCsv(std::string("neighSearchApprox") + (upperBound ? "Upper" : "Lower"), kernelStr, radius, stats, averageResultSize, tolerancePercentage);
        }

        template<Kernel_t kernel>
        void benchmarkSearchNeighOld(size_t repeats, float radius) {
            const auto kernelStr = kernelToString(kernel);
            auto [stats, averageResultSize] = benchmarking::benchmark<size_t>(repeats, [&]() { return searchNeighOld<kernel>(radius); }, useWarmup);
            appendToCsv("neighOldSearch", kernelStr, radius, stats, averageResultSize);
        }

        template<Kernel_t kernel>
        void benchmarkNumNeigh(size_t repeats, float radius) {
            const auto kernelStr = kernelToString(kernel);
            auto [stats, averageResultSize] = benchmarking::benchmark<size_t>(repeats, [&]() { return searchNumNeigh<kernel>(radius); }, useWarmup);
            appendToCsv("numNeighSearch", kernelStr, radius, stats, averageResultSize);
        }

        template<Kernel_t kernel>
        void benchmarkNumNeighOld(size_t repeats, float radius) {
            const auto kernelStr = kernelToString(kernel);
            auto [stats, averageResultSize] = benchmarking::benchmark<size_t>(repeats, [&]() { return searchNumNeighOld<kernel>(radius); }, useWarmup);
            appendToCsv("numNeighOldSearch", kernelStr, radius, stats, averageResultSize);
        }


        void benchmarkKNN(size_t repeats) {
            auto [stats, averageResultSize] = benchmarking::benchmark(repeats, [&]() { KNN(); }, useWarmup);
            appendToCsv("KNN", "NA", -1.0, stats);
        }

        void benchmarkRingSearchNeigh(size_t repeats, Vector &innerRadii, Vector &outerRadii) {
            auto [stats, averageResultSize] = benchmarking::benchmark<size_t>(repeats, [&]() { return searchNeighRing(innerRadii, outerRadii); }, useWarmup);
            appendToCsv("ringNeighSearch", "NA", -1.0, stats, averageResultSize);
        }

        void printBenchmarkLog(const std::string &bench_name, const std::vector<float> &benchmarkRadii, const size_t repeats, const size_t numSearches) {
            // Displaying the basic information with formatting
            std::cout << std::fixed << std::setprecision(3);
            std::string octreeName;
            if constexpr (std::is_same_v<Octree_t<Point_t, Encoder_t>, LinearOctree<Point_t, Encoder_t>>) {
                octreeName = "LinearOctree";
            } else if constexpr (std::is_same_v<Octree_t<Point_t, Encoder_t>, Octree<Point_t, Encoder_t>>) {
                octreeName = "Octree";
            }
            std::cout << std::left << std::setw(LOG_FIELD_WIDTH) << "Running benchmark:"        << std::setw(LOG_FIELD_WIDTH) << bench_name                      << "\n";
            std::cout << std::left << std::setw(LOG_FIELD_WIDTH) << "Octree used:"              << std::setw(LOG_FIELD_WIDTH) << octreeName        << "\n";
            std::cout << std::left << std::setw(LOG_FIELD_WIDTH) << "Point type:"               << std::setw(LOG_FIELD_WIDTH) << getPointName<Point_t>()           << "\n";
            std::cout << std::left << std::setw(LOG_FIELD_WIDTH) << "Encoder:"                  << std::setw(LOG_FIELD_WIDTH) << PointEncoding::getEncoderName<Encoder_t>() << "\n";
            std::cout << std::left << std::setw(LOG_FIELD_WIDTH) << "Radii:";

            // Outputting radii values in a similar structured format
            for (size_t i = 0; i < benchmarkRadii.size(); ++i) {
                if(i == 0)
                    std::cout << "{";
                std::cout << benchmarkRadii[i];
                if (i != benchmarkRadii.size() - 1) {
                    std::cout << ", ";
                }
            }
            std::cout << "}" << std::endl;

            // Showing other parameters
            std::cout << std::left << std::setw(LOG_FIELD_WIDTH) << "Number of searches:"       << std::setw(LOG_FIELD_WIDTH) << numSearches                      << "\n";
            std::cout << std::left << std::setw(LOG_FIELD_WIDTH) << "Repeats:"                  << std::setw(LOG_FIELD_WIDTH) << repeats                           << "\n";
            std::cout << std::left << std::setw(LOG_FIELD_WIDTH) << "Warmup:"                   << std::setw(LOG_FIELD_WIDTH) << (useWarmup ? "enabled" : "disabled") << "\n";
            std::cout << std::left << std::setw(LOG_FIELD_WIDTH) << "Parallel execution:"       << std::setw(LOG_FIELD_WIDTH) << (useParallel ? "enabled" : "disabled") << "\n";
            std::cout << std::left << std::setw(LOG_FIELD_WIDTH) << "Search set distribution:"  << std::setw(LOG_FIELD_WIDTH) << (searchSet->isSequential ? "sequential" : "random") << "\n";
            
            std::cout << std::endl;
            std::cout << std::left << std::setw(LOG_FIELD_WIDTH/2) << "Progress"    << std::setw(LOG_FIELD_WIDTH/2) << "Completed at" 
                                   << std::setw(LOG_FIELD_WIDTH*1.5) << "Method"      << std::setw(LOG_FIELD_WIDTH/2) << "Radius" << std::endl;
        }

        static void printBenchmarkUpdate(const std::string &method, const size_t totalExecutions, size_t &currentExecution, const float radius) {
            const std::string progress_str = "(" + std::to_string(currentExecution) + "/" + std::to_string(totalExecutions) + ")";
            std::cout << std::left << std::setw(LOG_FIELD_WIDTH/2) << progress_str  << std::setw(LOG_FIELD_WIDTH/2) << getCurrentDate("[%H:%M:%S]") 
                                   << std::setw(LOG_FIELD_WIDTH*1.5) << method        << std::setw(LOG_FIELD_WIDTH/2) << radius << std::endl;
            currentExecution++;
        }

        void searchImplComparisonBench(const std::vector<float> &benchmarkRadii, const size_t repeats, const size_t numSearches) {
            printBenchmarkLog("neighSearch and numNeighSearch implementation comparison", benchmarkRadii, repeats, numSearches);
            size_t total = benchmarkRadii.size() * 4;
            size_t current = 1;
            for(size_t i = 0; i<benchmarkRadii.size(); i++) {
                benchmarkSearchNeighOld<Kernel_t::sphere>(repeats, benchmarkRadii[i]);
                benchmarkSearchNeighOld<Kernel_t::circle>(repeats, benchmarkRadii[i]);
                benchmarkSearchNeighOld<Kernel_t::cube>(repeats, benchmarkRadii[i]);
                benchmarkSearchNeighOld<Kernel_t::square>(repeats, benchmarkRadii[i]);
                printBenchmarkUpdate("Neighbor search - old impl.", total, current, benchmarkRadii[i]);

                benchmarkSearchNeigh<Kernel_t::sphere>(repeats, benchmarkRadii[i]);
                benchmarkSearchNeigh<Kernel_t::circle>(repeats, benchmarkRadii[i]);
                benchmarkSearchNeigh<Kernel_t::cube>(repeats, benchmarkRadii[i]);
                benchmarkSearchNeigh<Kernel_t::square>(repeats, benchmarkRadii[i]);
                printBenchmarkUpdate("Neighbor search - new impl.", total, current, benchmarkRadii[i]);

                benchmarkNumNeighOld<Kernel_t::sphere>(repeats, benchmarkRadii[i]);
                benchmarkNumNeighOld<Kernel_t::circle>(repeats, benchmarkRadii[i]);
                benchmarkNumNeighOld<Kernel_t::cube>(repeats, benchmarkRadii[i]);
                benchmarkNumNeighOld<Kernel_t::square>(repeats, benchmarkRadii[i]);
                printBenchmarkUpdate("Num. neighbor search - old impl.", total, current, benchmarkRadii[i]);

                benchmarkNumNeigh<Kernel_t::sphere>(repeats, benchmarkRadii[i]);
                benchmarkNumNeigh<Kernel_t::circle>(repeats, benchmarkRadii[i]);
                benchmarkNumNeigh<Kernel_t::cube>(repeats, benchmarkRadii[i]);
                benchmarkNumNeigh<Kernel_t::square>(repeats, benchmarkRadii[i]);
                printBenchmarkUpdate("Num. neighbor search - new impl.", total, current, benchmarkRadii[i]);
            }
            std::cout << std::endl;
        }

        void searchPtrVsCopyBench(const std::vector<float> &benchmarkRadii, const size_t repeats, const size_t numSearches) {
            printBenchmarkLog("neighbors vs copyNeighbors comparison", benchmarkRadii, repeats, numSearches);
            size_t total = benchmarkRadii.size() * 2;
            size_t current = 1;
            for(size_t i = 0; i<benchmarkRadii.size(); i++) {
                benchmarkSearchNeigh<Kernel_t::sphere>(repeats, benchmarkRadii[i]);
                benchmarkSearchNeigh<Kernel_t::circle>(repeats, benchmarkRadii[i]);
                benchmarkSearchNeigh<Kernel_t::cube>(repeats, benchmarkRadii[i]);
                benchmarkSearchNeigh<Kernel_t::square>(repeats, benchmarkRadii[i]);
                printBenchmarkUpdate("Neighbor search - pointer", total, current, benchmarkRadii[i]);
                benchmarkSearchNeighCopy<Kernel_t::sphere>(repeats, benchmarkRadii[i]);
                benchmarkSearchNeighCopy<Kernel_t::circle>(repeats, benchmarkRadii[i]);
                benchmarkSearchNeighCopy<Kernel_t::cube>(repeats, benchmarkRadii[i]);
                benchmarkSearchNeighCopy<Kernel_t::square>(repeats, benchmarkRadii[i]);
                printBenchmarkUpdate("Neighbor search - copy", total, current, benchmarkRadii[i]);
            }
            std::cout << std::endl;
        }

        void searchPtrVsStructBench(const std::vector<float> &benchmarkRadii, const size_t repeats, const size_t numSearches) {
            printBenchmarkLog("neighbors vs neighborsStruct comparison", benchmarkRadii, repeats, numSearches);
            size_t total = benchmarkRadii.size() * 2;
            size_t current = 1;
            for(size_t i = 0; i<benchmarkRadii.size(); i++) {
                benchmarkSearchNeigh<Kernel_t::sphere>(repeats, benchmarkRadii[i]);
                printBenchmarkUpdate("Neighbor search - pointer", total, current, benchmarkRadii[i]);
                benchmarkSearchNeighStruct<Kernel_t::sphere>(repeats, benchmarkRadii[i]);
                printBenchmarkUpdate("Neighbor search - struct", total, current, benchmarkRadii[i]);
            }
            std::cout << std::endl;
        }

        void searchBench(const std::vector<float> &benchmarkRadii, const size_t repeats, const size_t numSearches) {
            printBenchmarkLog("neighSearch and numNeighSearch", benchmarkRadii, repeats, numSearches);
            size_t total = benchmarkRadii.size() * 2;
            size_t current = 1;
            for(size_t i = 0; i<benchmarkRadii.size(); i++) {
                benchmarkSearchNeigh<Kernel_t::sphere>(repeats, benchmarkRadii[i]);
                benchmarkSearchNeigh<Kernel_t::circle>(repeats, benchmarkRadii[i]);
                benchmarkSearchNeigh<Kernel_t::cube>(repeats, benchmarkRadii[i]);
                benchmarkSearchNeigh<Kernel_t::square>(repeats, benchmarkRadii[i]);
                printBenchmarkUpdate("Neighbor search", total, current, benchmarkRadii[i]);

                benchmarkNumNeigh<Kernel_t::sphere>(repeats, benchmarkRadii[i]);
                benchmarkNumNeigh<Kernel_t::circle>(repeats, benchmarkRadii[i]);
                benchmarkNumNeigh<Kernel_t::cube>(repeats, benchmarkRadii[i]);
                benchmarkNumNeigh<Kernel_t::square>(repeats, benchmarkRadii[i]);
                printBenchmarkUpdate("Num. neighbor search", total, current, benchmarkRadii[i]);
            }
            std::cout << std::endl;
        }

        void approxSearchBench(const std::vector<float> &benchmarkRadii, const size_t repeats, const size_t numSearches, const std::vector<double> tolerancePercentages) {
            printBenchmarkLog("Approximate searches with low and high bounds", benchmarkRadii, repeats, numSearches);
            size_t total = benchmarkRadii.size() * tolerancePercentages.size();
            size_t current = 1;
            for(size_t i = 0; i<benchmarkRadii.size(); i++) {
                benchmarkSearchNeigh<Kernel_t::sphere>(repeats, benchmarkRadii[i]);
                for(size_t j = 0; j<tolerancePercentages.size(); j++) {
                    benchmarkSearchNeighApprox<Kernel_t::sphere>(repeats, benchmarkRadii[i], tolerancePercentages[j], false);
                    benchmarkSearchNeighApprox<Kernel_t::sphere>(repeats, benchmarkRadii[i], tolerancePercentages[j], true);
                    printBenchmarkUpdate(std::string("Approx. neighSearch with tol = ") + std::to_string(tolerancePercentages[j]) + std::string("%"), total, current, benchmarkRadii[i]);
                }
            }
        }

        std::shared_ptr<const SearchSet> getSearchSet() const { return searchSet; }
        std::shared_ptr<ResultSet<Point_t>> getResultSet() const { return resultSet; }
};