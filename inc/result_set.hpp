#pragma once
#include "search_set.hpp"
#include "neighbor_set.hpp"
#include <vector>
#include <set>

template <PointType Point_t>
struct ResultSet {
    const std::shared_ptr<const SearchSet> searchSet;
    std::vector<std::vector<Point_t*>> resultsNeigh;
    std::vector<NeighborSet<Point_t>> resultsNeighStruct;
    std::vector<std::vector<Point_t>> resultsNeighCopy;
    std::vector<std::vector<Point_t*>> resultsNeighOld;
    std::vector<size_t> resultsNumNeigh;
    std::vector<size_t> resultsNumNeighOld;
    std::vector<std::vector<Point_t*>> resultsKNN;
    std::vector<std::vector<Point_t*>> resultsRingNeigh;
    std::vector<NeighborSet<Point_t>> resultsSearchApproxUpper;
    std::vector<NeighborSet<Point_t>> resultsSearchApproxLower;
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
        std::vector<NeighborSet<Point_t>>& resultsStruct,
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

        // Check difference in access times
        // Measure access times
        auto measureAccessTime = [](auto& container, auto accessor) {
            std::uintptr_t dummy = 0; // Prevent optimization
            auto start = std::chrono::high_resolution_clock::now();
            for (const auto& elem : container) {
                for (const auto point : accessor(elem)) {
                    dummy ^= reinterpret_cast<uintptr_t>(&point); // Prevent optimization
                }
            }
            asm volatile("" : "+r" (dummy)); // Prevent optimization
            auto end = std::chrono::high_resolution_clock::now();
            return std::chrono::duration<double, std::milli>(end - start).count();
        };
        auto start = std::chrono::high_resolution_clock::now();
        uint64_t random_stuff = 0;
        for(const auto p: results[0]) {
            random_stuff += (uint64_t) (p->getX() + p->getY() + p->getZ());
        }
        auto end = std::chrono::high_resolution_clock::now();
        double timeVector = std::chrono::duration<double, std::milli>(end - start).count();

        std::cout << random_stuff << std::endl;
        start = std::chrono::high_resolution_clock::now();
        uint64_t random_stuff_2 = 0;
        for(const auto p: resultsStruct[0]) {
            random_stuff_2 += (uint64_t) (p.getX() + p.getY() + p.getZ());
        }
        end = std::chrono::high_resolution_clock::now();
        std::cout << random_stuff_2 << std::endl;
        double timeStruct = std::chrono::duration<double, std::milli>(end - start).count();

        std::cout << "Access time for std::vector<Point_t*>: " << timeVector << " ms" << std::endl;
        std::cout << "Access time for NeighborSet<Point_t>: " << timeStruct << " ms" << std::endl;
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
