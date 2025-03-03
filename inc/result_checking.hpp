#pragma once
#include "search_set.hpp"
#include "neighbor_set.hpp"
#include <vector>
#include <set>
#include <concepts>

namespace ResultChecking {
    
    template <PointType Point_t>
    struct ResultSet {
        const std::shared_ptr<const SearchSet> searchSet;
        std::vector<std::vector<Point_t*>> resultsNeigh;
        std::vector<NeighborSet<Point_t>> resultsNeighStruct;
        std::vector<std::vector<Point_t*>> resultsNeighOld;
        std::vector<size_t> resultsNumNeigh;
        std::vector<size_t> resultsNumNeighOld;
        std::vector<std::vector<Point_t*>> resultsKNN;
        std::vector<std::vector<Point_t*>> resultsRingNeigh;
        std::vector<NeighborSet<Point_t>> resultsSearchApproxUpper;
        std::vector<NeighborSet<Point_t>> resultsSearchApproxLower;
        double tolerancePercentageUsed;
        
        ResultSet(const std::shared_ptr<const SearchSet> searchSet): searchSet(searchSet) {  }

        // Copy constructor
        ResultSet(const ResultSet& other)
        : searchSet(other.searchSet),
          resultsNeigh(other.resultsNeigh),
          resultsNeighStruct(other.resultsNeighStruct),
          resultsNeighOld(other.resultsNeighOld),
          resultsNumNeigh(other.resultsNumNeigh),
          resultsNumNeighOld(other.resultsNumNeighOld),
          resultsKNN(other.resultsKNN),
          resultsRingNeigh(other.resultsRingNeigh),
          resultsSearchApproxUpper(other.resultsSearchApproxUpper),
          resultsSearchApproxLower(other.resultsSearchApproxLower),
          tolerancePercentageUsed(other.tolerancePercentageUsed) { }
        
          // Move constructor
        ResultSet(ResultSet&& other)
        : searchSet(std::move(other.searchSet)),
          resultsNeigh(std::move(other.resultsNeigh)),
          resultsNeighStruct(std::move(other.resultsNeighStruct)),
          resultsNeighOld(std::move(other.resultsNeighOld)),
          resultsNumNeigh(std::move(other.resultsNumNeigh)),
          resultsNumNeighOld(std::move(other.resultsNumNeighOld)),
          resultsKNN(std::move(other.resultsKNN)),
          resultsRingNeigh(std::move(other.resultsRingNeigh)),
          resultsSearchApproxUpper(std::move(other.resultsSearchApproxUpper)),
          resultsSearchApproxLower(std::move(other.resultsSearchApproxLower)),
          tolerancePercentageUsed(other.tolerancePercentageUsed) { }
    };

    // Generic check for neighbor results
    template <typename Point_t>
    static std::vector<size_t> checkResults(
        const std::vector<std::vector<Point_t*>> &results1, 
        const std::vector<std::vector<Point_t*>> &results2)
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

    template <typename Point_t>
    static std::vector<size_t> checkResults(
        const std::vector<std::vector<Point_t*>>& results1, 
        const std::vector<NeighborSet<Point_t>>& results2)
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
            
            // Extract IDs from the NeighborSet using its iterator
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

    template <typename Point_t>
    static std::vector<size_t> checkResults(
        const std::vector<NeighborSet<Point_t>>& results1,
        const std::vector<std::vector<Point_t*>>& results2) 
    {
        return checkResults(results2, results1);
    }

    template <typename Point_t>
    static std::vector<size_t> checkResults(
        const std::vector<NeighborSet<Point_t>>& results1, 
        const std::vector<NeighborSet<Point_t>>& results2) {
        std::vector<size_t> wrongSearches;
        for (size_t i = 0; i < results1.size() && i < results2.size(); i++) {
            const auto& set1 = results1[i];
            const auto& set2 = results2[i];
            
            // Extract IDs from both NeighborSets
            std::set<int> ids1, ids2;
            for (const auto& point : set1) ids1.insert(point.id());
            for (const auto& point : set2) ids2.insert(point.id());
            
            // Compare sets directly
            if (ids1 != ids2) {
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
    static std::vector<size_t> checkResultsNumNeigh(
        const std::vector<size_t> &results1, 
        const std::vector<size_t> &results2)
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

    
    // Concept to check if the provided containers are exactly:
    // 1. std::vector<std::vector<Point_t*>> (for results1)
    // 2. std::vector<NeighborSet<Point_t>> (for results2)
    // Point_t is another template typename
    
    // Wrapper for checkResults with logging about what checks failed
    template <typename ResultsContainer1, typename ResultsContainer2>
    static void checkOperationNeigh(
        const ResultsContainer1 &results1,
        const ResultsContainer2 &results2,
        size_t printingLimit = 10)
    {
        std::vector<size_t> wrongSearches = checkResults(results1, results2);
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

    // Operation for checking number of neighbor results
    static void checkOperationNumNeigh(
        const std::vector<size_t> &results1,
        const std::vector<size_t> &results2,
        size_t printingLimit = 10)
    {
        std::vector<size_t> wrongSearches = checkResultsNumNeigh(results1, results2);
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

    // Check differences in access time between both containers
    template <typename Point_t>
    static void checkAcessTimesDifference(
        const std::vector<std::vector<Point_t*>>& results, 
        const std::vector<NeighborSet<Point_t>>& resultsStruct) 
    {
        if(results.empty() || resultsStruct.empty()){
            std::cout << "Vectors are empty, can't check access times!" << std::endl;
            return;
        }
        size_t N = std::min(results.size(), resultsStruct.size());

        volatile uint64_t acc1 = 0, acc2 = 0;
        auto start1 = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < N; i++) {
            for (const auto p : results[i]) {
                acc1 = acc1 + static_cast<uint64_t>(p->getX() + p->getY() + p->getZ());
            }
        }
        auto end1 = std::chrono::high_resolution_clock::now();
        double timeVector = std::chrono::duration<double, std::milli>(end1 - start1).count();
        
        auto start2 = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < N; i++) {
            for (const auto& p : resultsStruct[i]) {
                acc2 = acc2 + static_cast<uint64_t>(p.getX() + p.getY() + p.getZ());
            }
        }
        auto end2 = std::chrono::high_resolution_clock::now();
        double timeStruct = std::chrono::duration<double, std::milli>(end2 - start2).count();
        std::cout << "Access time for " << N << " instances of std::vector<Point_t*>: " << timeVector << " ms" << std::endl;
        std::cout << "Access time for " << N << " instances of NeighborSet<Point_t>: " << timeStruct << " ms" << std::endl;
    }

    // Check differences in linear vs pointer octree results
    template <typename Point_t>
    static void checkResultsLinearVsPointer(
        const ResultSet<Point_t>& resultSetLinear, 
        const ResultSet<Point_t>& resultSetPointer, 
        size_t printingLimit = 10) {
        if(!resultSetLinear.resultsNeigh.empty() && !resultSetPointer.resultsNeigh.empty()) {
            std::cout << "Checking neigh search results with linear neigh vector and pointer neigh vector\n";
            checkOperationNeigh(resultSetLinear.resultsNeigh, resultSetPointer.resultsNeigh, printingLimit);
        } else if(!resultSetLinear.resultsNeighStruct.empty() && !resultSetPointer.resultsNeigh.empty()) {
            std::cout << "Checking neigh search results with linear neigh struct and pointer neigh vector\n";
            checkOperationNeigh(resultSetLinear.resultsNeighStruct, resultSetPointer.resultsNeigh, printingLimit);
        } else {
            std::cout << "No neigh search results were computed!\n";
        }
        if(!resultSetLinear.resultsNumNeigh.empty() && !resultSetPointer.resultsNumNeigh.empty()) {
            std::cout << "Checking num neigh search results\n";
            checkOperationNumNeigh(resultSetLinear.resultsNumNeigh, resultSetPointer.resultsNumNeigh, printingLimit);
        } else {
            std::cout << "No num neigh search results were computed!\n";
        }
    }

    // Check differences in algorithm comparison results
    template <typename Point_t>
    static void checkResultsAlgoComp(
        const ResultSet<Point_t>& resultSet, 
        size_t printingLimit = 10) {
        if(!resultSet.resultsNeigh.empty() && !resultSet.resultsNeighOld.empty()) {
            std::cout << "Checking neigh search vs old neigh search results\n";
            checkOperationNeigh(resultSet.resultsNeigh, resultSet.resultsNeighOld, printingLimit);
        }
        if(!resultSet.resultsNeigh.empty() && !resultSet.resultsNeighStruct.empty()) {
            std::cout << "Checking neigh search vs struct neigh search results\n";
            checkOperationNeigh(resultSet.resultsNeigh, resultSet.resultsNeighStruct, printingLimit);
            checkAcessTimesDifference(resultSet.resultsNeigh, resultSet.resultsNeighStruct);
        }
        if(!resultSet.resultsNumNeigh.empty() && !resultSet.resultsNumNeighOld.empty()) {
            std::cout << "Checking num neigh search vs old num neigh search results\n";
            checkOperationNumNeigh(resultSet.resultsNumNeigh, resultSet.resultsNumNeighOld, printingLimit);
        }
    }

    // Check differences between approximate searches results
    template <typename Point_t>
    static void checkResultsApproxSearches(
        const ResultSet<Point_t>& resultSet, 
        size_t printingLimit = 10) {
        if (resultSet.resultsSearchApproxLower.empty() || resultSet.resultsSearchApproxUpper.empty() || resultSet.resultsNeighStruct.empty()) {
            std::cout << "Approximate searches results were not computed! Not checking approximation results.\n";
            return;
        }
    
        size_t printingOn = std::min(resultSet.searchSet->numSearches, printingLimit);
        std::cout << "Approximate searches results (printing " << printingOn 
                  << " searches of a total of " << resultSet.searchSet->numSearches << " searches performed):\n";
        std::cout << "Tolerance percentage used: " << resultSet.tolerancePercentageUsed << "%\n";
    
        // Column headers
        std::cout << std::left 
                  << std::setw(10) << "Search #" 
                  << std::setw(15) << "Lower bound" 
                  << std::setw(15) << "Exact search" 
                  << std::setw(15) << "Upper bound"
                  << "\n";
    
        double totalDiffLower = 0.0, totalDiffUpper = 0.0;
        size_t nnzSearches = resultSet.searchSet->numSearches;
    
        for (size_t i = 0; i < resultSet.searchSet->numSearches; i++) {
            // Avoid division by zero
            if (resultSet.resultsNeighStruct[i].empty()) {
                nnzSearches--;
                continue;
            }
            size_t upperSize = resultSet.resultsSearchApproxUpper[i].size(); 
            size_t lowerSize = resultSet.resultsSearchApproxLower[i].size();
            size_t exactSize = resultSet.resultsNeighStruct[i].size();
            // Compute percentage differences
            totalDiffLower += (static_cast<double>(exactSize - lowerSize) 
                / exactSize) * 100.0;
            totalDiffUpper += (static_cast<double>(upperSize - exactSize) 
                / exactSize) * 100.0;
    
            if (i < printingOn) {
                std::cout << std::left 
                          << std::setw(10) << (i + 1) 
                          << std::setw(15) << lowerSize 
                          << std::setw(15) << exactSize 
                          << std::setw(15) << upperSize
                          << "\n";
            }
        }
        if(nnzSearches > 0) {
            std::cout << "On average over all searches done, lower bound searches found " 
                    << (totalDiffLower / nnzSearches) << "% fewer points.\n";
            std::cout << "On average over all searches done, upper bound searches found " 
                    << (totalDiffUpper / nnzSearches) << "% more points.\n";
        } else {
            std::cout << "No non-empty approximate searches were found!\n";
        }
    }    
};
