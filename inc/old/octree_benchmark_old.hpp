#pragma once

#include "benchmarking.hpp"
#include "octree.hpp"
#include "linear_octree.hpp"
#include "octree_linear_old.hpp"
#include "morton_encoder.hpp"
#include <random>
#include "point.hpp"
#include <omp.h>
#include "NeighborKernels/KernelFactory.hpp"


class OctreeBenchmarkOld {
    private:
        const size_t numSearches;

        constexpr static bool CHECK_RESULTS = true;
        constexpr static uint32_t MIN_KNN = 5;
        constexpr static uint32_t MAX_KNN = 100;

        const std::string startTimestamp;
        // Copy points for the linear octree because it reorders them and also to be more fair on comparisons
        // i.e. neither tree has points already in-cache when executing after the other
        std::vector<Lpoint> &points, &lOctPoints;

        Octree* pOct = nullptr;
        LinearOctreeOld* oldLOct = nullptr;
        LinearOctree *lOct = nullptr;
        std::mt19937 rng;
        
        std::vector<size_t> searchPointIndexes;
        std::vector<uint32_t> searchKNNLimits;

        std::vector<std::vector<Lpoint*>> searchResultsPointer;
        std::vector<std::vector<Lpoint*>> searchResultsOldLinear;
        std::vector<std::vector<Lpoint*>> searchResultsLinear;

        std::vector<size_t> numNeighResultsPointer;
        std::vector<size_t> numNeighResultsOldLinear;
        std::vector<size_t> numNeighResultsLinear;

        void allocateResultCheckingMemory() {
            searchResultsPointer.resize(numSearches);
            searchResultsOldLinear.resize(numSearches);
            searchResultsLinear.resize(numSearches);
            numNeighResultsPointer.resize(numSearches);
            numNeighResultsOldLinear.resize(numSearches);
            numNeighResultsLinear.resize(numSearches);
        }

        void generateSearchSet() {
            rng.seed(42);
            searchPointIndexes.resize(numSearches);
            searchKNNLimits.resize(numSearches);
            std::uniform_int_distribution<size_t> indexDist(0, points.size()-1);
            std::uniform_int_distribution<size_t> knnDist(MIN_KNN, MAX_KNN);
            
                for(int i = 0; i<numSearches; i++) {
                    searchPointIndexes[i] = indexDist(rng);
                    searchKNNLimits[i] = knnDist(rng);
                }
        }

        bool checkNeighSearchResults(float radii) {
            bool correct = true;
            for(int i = 0; i<numSearches; i++) {
                auto point = searchResultsPointer[i];
                auto oldLin = searchResultsLinear[i];
                if(point.size() != oldLin.size()) {
                            std::cout << "Wrong search result size in set " << i << 
                    "\n Pointer = " << point.size() << " Old linear = " << oldLin.size() << std::endl << 
                    "Search center: " << points[searchPointIndexes[i]] << " with radii " << radii << std::endl;
                    correct = false;
                } else {
                    sort(point.begin(), point.end(), [&] (Lpoint *p, Lpoint* q) -> bool {
                        return p->id() < q->id();
                    });
                    sort(oldLin.begin(), oldLin.end(), [&] (Lpoint *p, Lpoint* q) -> bool {
                        return p->id() < q->id();
                    });
                    for(int j = 0; j<point.size(); j++){
                        if(point[j]->id() != oldLin[j]->id()) {
                            std::cout << "Wrong search result point in set " << i << " at index " << j <<
                            "\n Pointer = " << point[j]->id() << " Old linear =  " << oldLin[j]->id() << std::endl;
                            correct = false;
                        }
                    }
                }
            }

            for(int i = 0; i<numSearches; i++) {
                auto point = searchResultsPointer[i];
                auto lin = searchResultsLinear[i];
                if(point.size() != lin.size()) {
                    std::cout << "Wrong search result size in set " << i << 
                    "\n Pointer = " << point.size() << " Linear = " << lin.size() << std::endl << 
                    "Search center: " << points[searchPointIndexes[i]] << " with radii " << radii << std::endl;
                    correct = false;
                } else {
                    sort(point.begin(), point.end(), [&] (Lpoint *p, Lpoint* q) -> bool {
                        return p->id() < q->id();
                    });
                    sort(lin.begin(), lin.end(), [&] (Lpoint *p, Lpoint* q) -> bool {
                        return p->id() < q->id();
                    });
                    for(int j = 0; j<point.size(); j++){
                        if(point[j]->id() != lin[j]->id()) {
                            std::cout << "Wrong search result point in set " << i << " at index " << j <<
                            "\n Pointer = " << point[j]->id() << " Linear =  " << lin[j]->id() << std::endl;
                            correct = false;
                        }
                    }
                }
            }
            return correct;
        }
        bool checkKNNSearchResults() {
                        bool correct = true;
            for(int i = 0; i<numSearches; i++) {
                auto point = searchResultsPointer[i];
                auto oldLin = searchResultsLinear[i];
                if(point.size() != oldLin.size()) {
                            std::cout << "Wrong search result size in set " << i << 
                    "\n Pointer = " << point.size() << " Old linear = " << oldLin.size() << std::endl << 
                    "Search center: " << points[searchPointIndexes[i]] << std::endl;
                    correct = false;
                } else {
                    sort(point.begin(), point.end(), [&] (Lpoint *p, Lpoint* q) -> bool {
                        return p->id() < q->id();
                    });
                    sort(oldLin.begin(), oldLin.end(), [&] (Lpoint *p, Lpoint* q) -> bool {
                        return p->id() < q->id();
                    });
                    for(int j = 0; j<point.size(); j++){
                        if(point[j]->id() != oldLin[j]->id()) {
                            std::cout << "Wrong search result point in set " << i << " at index " << j <<
                            "\n Pointer = " << point[j]->id() << " Old linear =  " << oldLin[j]->id() << std::endl;
                            correct = false;
                        }
                    }
                }
            }

            for(int i = 0; i<numSearches; i++) {
                auto point = searchResultsPointer[i];
                auto lin = searchResultsLinear[i];
                if(point.size() != lin.size()) {
                    std::cout << "Wrong search result size in set " << i << 
                    "\n Pointer = " << point.size() << " Linear = " << lin.size() << std::endl << 
                    "Search center: " << points[searchPointIndexes[i]] << std::endl;
                    correct = false;
                } else {
                    sort(point.begin(), point.end(), [&] (Lpoint *p, Lpoint* q) -> bool {
                        return p->id() < q->id();
                    });
                    sort(lin.begin(), lin.end(), [&] (Lpoint *p, Lpoint* q) -> bool {
                        return p->id() < q->id();
                    });
                    for(int j = 0; j<point.size(); j++){
                        if(point[j]->id() != lin[j]->id()) {
                            std::cout << "Wrong search result point in set " << i << " at index " << j <<
                            "\n Pointer = " << point[j]->id() << " Linear =  " << lin[j]->id() << std::endl;
                            correct = false;
                        }
                    }
                }
            }
            return correct;
        }

        bool checkNumNeighResults(float radii) {
            bool correct = true;
            for(int i = 0; i<numSearches; i++) {
                size_t point = numNeighResultsPointer[i];
                size_t oldLin = numNeighResultsLinear[i];
                if(point != oldLin) {
                    std::cout << "Wrong search result size in set " << i << 
                    "\n Pointer = " << point << " Old linear = " << oldLin << std::endl << 
                    "Search center: " << points[searchPointIndexes[i]] << " with radii " << radii << std::endl;
                    correct = false;
                }
            }
            for(int i = 0; i<numSearches; i++) {
                size_t point = numNeighResultsPointer[i];
                size_t lin = numNeighResultsLinear[i];
                if(point != lin) {
                    std::cout << "Wrong search result size in set " << i << 
                    "\n Pointer = " << point << " Linear = " << lin << std::endl << 
                    "Search center: " << points[searchPointIndexes[i]] << " with radii " << radii << std::endl;
                    correct = false;
                }
            }
            return correct;
        }

        bool checkNeighKNNResults() {
            bool correct = true;
            for(int i = 0; i<numSearches; i++) {
                auto point = searchResultsPointer[i];
                auto oldLin = searchResultsLinear[i];
                if(point.size() != oldLin.size()) {
                    std::cout << "Wrong search result size in set " << i << 
                    "\n Pointer = " << point.size() << " Old linear = " << oldLin.size() << std::endl << 
                    "Search center: " << points[searchPointIndexes[i]] << " with max KNN " << searchKNNLimits[i] << std::endl;
                    correct = false;
                } else {
                    sort(point.begin(), point.end(), [&] (Lpoint *p, Lpoint* q) -> bool {
                        return p->id() < q->id();
                    });
                    sort(oldLin.begin(), oldLin.end(), [&] (Lpoint *p, Lpoint* q) -> bool {
                        return p->id() < q->id();
                    });
                    for(int j = 0; j<point.size(); j++){
                        if(point[j]->id() != oldLin[j]->id()) {
                            std::cout << "Wrong search result point in set " << i << " at index " << j <<
                            "\n Pointer = " << point[j]->id() << " Old linear =  " << oldLin[j]->id() << std::endl;
                            correct = false;
                        }
                    }
                }
            }

            for(int i = 0; i<numSearches; i++) {
                auto point = searchResultsPointer[i];
                auto lin = searchResultsLinear[i];
                if(point.size() != lin.size()) {
                    std::cout << "Wrong search result size in set " << i << 
                    "\n Pointer = " << point.size() << " Linear = " << lin.size() << std::endl << 
                    "Search center: " << points[searchPointIndexes[i]] << " with max KNN " << searchKNNLimits[i] << std::endl;
                    correct = false;
                } else {
                    sort(point.begin(), point.end(), [&] (Lpoint *p, Lpoint* q) -> bool {
                        return p->id() < q->id();
                    });
                    sort(lin.begin(), lin.end(), [&] (Lpoint *p, Lpoint* q) -> bool {
                        return p->id() < q->id();
                    });
                    for(int j = 0; j<point.size(); j++){
                        if(point[j]->id() != lin[j]->id()) {
                            std::cout << "Wrong search result point in set " << i << " at index " << j <<
                            "\n Pointer = " << point[j]->id() << " Linear =  " << lin[j]->id() << std::endl;
                            correct = false;
                        }
                    }
                }
            }
            return correct;
        }

        bool checkRingNeighSearchResults(Vector& innerRadii, Vector& outerRadii) {
            bool correct = true;
            for(int i = 0; i<numSearches; i++) {
                auto point = searchResultsPointer[i];
                auto oldLin = searchResultsLinear[i];
                if(point.size() != oldLin.size()) {
                    std::cout << "Wrong search result size in set " << i << 
                    "\n Pointer = " << point.size() << " Old linear = " << oldLin.size() << "\n" << 
                    "  Search center: " << points[searchPointIndexes[i]] << "\n" <<
                    "  Inner radii of the ring " << innerRadii << "\n" <<
                    "   Outer radii of the ring " << outerRadii << std::endl;
                    correct = false;
                } else {
                    sort(point.begin(), point.end(), [&] (Lpoint *p, Lpoint* q) -> bool {
                        return p->id() < q->id();
                    });
                    sort(oldLin.begin(), oldLin.end(), [&] (Lpoint *p, Lpoint* q) -> bool {
                        return p->id() < q->id();
                    });
                    for(int j = 0; j<point.size(); j++){
                        if(point[j]->id() != oldLin[j]->id()) {
                            std::cout << "Wrong search result point in set " << i << " at index " << j <<
                            "\n Pointer = " << point[j]->id() << " Old linear =  " << oldLin[j]->id() << std::endl;
                            correct = false;
                        }
                    }
                }
            }
            for(int i = 0; i<numSearches; i++) {
                auto point = searchResultsPointer[i];
                auto lin = searchResultsLinear[i];
                if(point.size() != lin.size()) {
                    std::cout << "Wrong search result size in set " << i << 
                    "\n Pointer = " << point.size() << " Linear = " << lin.size() << "\n" << 
                    "  Search center: " << points[searchPointIndexes[i]] << "\n" <<
                    "  Inner radii of the ring " << innerRadii << "\n" <<
                    "   Outer radii of the ring " << outerRadii << std::endl;
                    correct = false;
                } else {
                    sort(point.begin(), point.end(), [&] (Lpoint *p, Lpoint* q) -> bool {
                        return p->id() < q->id();
                    });
                    sort(lin.begin(), lin.end(), [&] (Lpoint *p, Lpoint* q) -> bool {
                        return p->id() < q->id();
                    });
                    for(int j = 0; j<point.size(); j++){
                        if(point[j]->id() != lin[j]->id()) {
                            std::cout << "Wrong search result point in set " << i << " at index " << j <<
                            "\n Pointer = " << point[j]->id() << " Linear =  " << lin[j]->id() << std::endl;
                            correct = false;
                        }
                    }
                }
            }
            return correct;
        }


        void octreePointerBuild() {
            // Pointer based octree
            if(pOct != nullptr)
                delete pOct;
            pOct = new Octree(points);
        }
        void octreeOldLinearBuild() {
            // Old linear octree impl.
            if(oldLOct != nullptr)
                delete oldLOct;
            oldLOct = new LinearOctreeOld(points);
        }
        void octreeLinearBuild() {
            // Linear octree impl.
            if(lOct != nullptr)
                delete lOct;
            lOct = new LinearOctree(lOctPoints, true);
        }

        template<Kernel_t kernel>
        void pointerOctreeSearchNeigh(float radii) {
            #pragma omp parallel for schedule(static)
                for(int i = 0; i<numSearches; i++) {
                    if(CHECK_RESULTS){
                        searchResultsPointer[i] = pOct->searchNeighbors<kernel>(points[searchPointIndexes[i]], radii);
                    } else{
                        (void) pOct->searchNeighbors<kernel>(points[searchPointIndexes[i]], radii);
                        
                    }
                }
        }

        template<Kernel_t kernel>
        void oldLinearSearchNeigh(float radii) {
            #pragma omp parallel for schedule(static)
                for(int i = 0; i<numSearches; i++) {
                    if(CHECK_RESULTS){
                        searchResultsOldLinear[i] = oldLOct->searchNeighbors<kernel>(points[searchPointIndexes[i]], radii);
                    }else{
                        (void) oldLOct->searchNeighbors<kernel>(points[searchPointIndexes[i]], radii);
                        
                    }
                }
        }

        template<Kernel_t kernel>
        void linearOctreeNeighborSearch(float radii) {
            #pragma omp parallel for schedule(static)
                for(int i = 0; i<numSearches; i++) {
                    if(CHECK_RESULTS){
                        searchResultsLinear[i] = lOct->searchNeighbors<kernel>(points[searchPointIndexes[i]], radii);
                    } else {
                        (void) lOct->searchNeighbors<kernel>(points[searchPointIndexes[i]], radii);
                    }
                }
        }

        template<Kernel_t kernel>
        void pointerOctreeNumNeigh(float radii) {
            #pragma omp parallel for schedule(static)
                for(int i = 0; i<numSearches; i++) {
                    if(CHECK_RESULTS) {
                        numNeighResultsPointer[i] = pOct->numNeighbors<kernel>(points[searchPointIndexes[i]], radii);
                    } else {
                        (void) pOct->numNeighbors<kernel>(points[searchPointIndexes[i]], radii);
                    }
                }
        }

        template<Kernel_t kernel>
        void oldLinearOctreeNumNeigh(float radii) {
            #pragma omp parallel for schedule(static)
                for(int i = 0; i<numSearches; i++) {
                    if(CHECK_RESULTS) {
                        numNeighResultsOldLinear[i] = oldLOct->numNeighbors<kernel>(points[searchPointIndexes[i]], radii);
                    } else {
                        (void) oldLOct->numNeighbors<kernel>(points[searchPointIndexes[i]], radii);
                    }
                }
        }

        template<Kernel_t kernel>
        void linearOctreeNumNeigh(float radii) {
            #pragma omp parallel for schedule(static)
                for(int i = 0; i<numSearches; i++) {
                    if(CHECK_RESULTS) {
                        numNeighResultsLinear[i] = lOct->numNeighbors<kernel>(points[searchPointIndexes[i]], radii);
                    } else {
                        (void) lOct->numNeighbors<kernel>(points[searchPointIndexes[i]], radii);
                    }
                }
        }

        void pointerOctreeKNN() {
            #pragma omp parallel for schedule(static)
                for(int i = 0; i<numSearches; i++) {
                    if(CHECK_RESULTS) {
                        searchResultsPointer[i] = pOct->KNN(points[searchPointIndexes[i]], searchKNNLimits[i], searchKNNLimits[i]);
                    } else {
                        (void) pOct->KNN(points[searchPointIndexes[i]], searchKNNLimits[i], searchKNNLimits[i]);
                    }
                }
        }

        void oldLinearOctreeKNN() {
            #pragma omp parallel for schedule(static)
                for(int i = 0; i<numSearches; i++) {
                    if(CHECK_RESULTS) {
                        searchResultsOldLinear[i] = oldLOct->KNN(points[searchPointIndexes[i]], searchKNNLimits[i], searchKNNLimits[i]);
                    } else {
                        (void) oldLOct->KNN(points[searchPointIndexes[i]], searchKNNLimits[i], searchKNNLimits[i]);
                    }
                }
        }

        void linearOctreeKNN() {
            #pragma omp parallel for schedule(static)
                for(int i = 0; i<numSearches; i++) {
                    if(CHECK_RESULTS) {
                        searchResultsLinear[i] = lOct->KNN(points[searchPointIndexes[i]], searchKNNLimits[i], searchKNNLimits[i]);
                    } else {
                        (void) lOct->KNN(points[searchPointIndexes[i]], searchKNNLimits[i], searchKNNLimits[i]);
                    }
                }
        }

        void pointerOctreeRingSearchNeigh(Vector &innerRadii, Vector &outerRadii) {
            #pragma omp parallel for schedule(static)
                for(int i = 0; i<numSearches; i++) {
                    if(CHECK_RESULTS) {    
                        searchResultsPointer[i] = pOct->searchNeighborsRing(points[searchPointIndexes[i]], 
                            innerRadii, outerRadii);
                    } else {
                        (void) pOct->searchNeighborsRing(points[searchPointIndexes[i]], 
                            innerRadii, outerRadii);
                    }
                }
        }

        void oldLinearOctreeRingSearchNeigh(Vector &innerRadii, Vector &outerRadii) {
            #pragma omp parallel for schedule(static)
                for(int i = 0; i<numSearches; i++) {
                    if(CHECK_RESULTS) {
                        searchResultsOldLinear[i] = oldLOct->searchNeighborsRing(points[searchPointIndexes[i]], 
                            innerRadii, outerRadii);
                    } else {
                        (void) oldLOct->searchNeighborsRing(points[searchPointIndexes[i]], 
                            innerRadii, outerRadii);
                    }
                }
        }

        void linearOctreeRingSearchNeigh(Vector &innerRadii, Vector &outerRadii) {
            #pragma omp parallel for schedule(static)
                for(int i = 0; i<numSearches; i++) {
                    if(CHECK_RESULTS) {
                        searchResultsLinear[i] = lOct->searchNeighborsRing(points[searchPointIndexes[i]], 
                            innerRadii, outerRadii);
                    } else {
                        (void) lOct->searchNeighborsRing(points[searchPointIndexes[i]], 
                            innerRadii, outerRadii);
                    }
                }
        }

        inline std::string getCurrentDate() {
            auto t = std::time(nullptr);
            auto tm = *std::localtime(&t);
            std::ostringstream oss;
            oss << std::put_time(&tm, "%Y-%m-%d-%H:%M:%S");
            return oss.str();
        }

        inline void appendToCsv(const std::string& octree, const std::string& operation, 
                            const std::string& kernel, const float radius, const benchmarking::Stats<>& stats) {
            // Open the file in append mode
            std::string csvFilename = mainOptions.inputFileName + "-" + startTimestamp + ".csv";
            std::filesystem::path csvPath = mainOptions.outputDirName / csvFilename;
            std::ofstream file(csvPath, std::ios::app);
            if (!file.is_open()) {
                throw std::ios_base::failure(std::string("Failed to open benchmark output file: ") + csvPath.string());
            }

            // Check if the file is empty and append header if it is
            if (file.tellp() == 0) {
                file << "date,octree,operation,kernel,radius,num_searches,repeats,accumulated,mean,median,stdev,used_warmup\n";
            }

            // Append the benchmark data
            file << getCurrentDate() << ',' 
                << octree << ',' 
                << operation << ',' 
                << kernel << ',' 
                << radius << ','
                << numSearches << ',' 
                << stats.size() << ','
                << stats.accumulated() << ',' 
                << stats.mean() << ',' 
                << stats.median() << ',' 
                << stats.stdev() << ','
                << stats.usedWarmup() << '\n';
        }
    
    public:
        OctreeBenchmarkOld(std::vector<Lpoint> &points, std::vector<Lpoint> &lsOctreePoints, size_t numSearches = 100) : 
            points(points), lOctPoints(lsOctreePoints), numSearches(numSearches), startTimestamp(getCurrentDate()) {
            generateSearchSet();

            if(CHECK_RESULTS)
                allocateResultCheckingMemory();
            
            octreePointerBuild();
            octreeLinearBuild();
        }

        void benchmarkBuild(size_t repeats) {
            auto statsPointer = benchmarking::benchmark(repeats, [&]() { octreePointerBuild(); });
            appendToCsv("pointer", "build", "NA", -1.0, statsPointer);

            auto statsLinear = benchmarking::benchmark(repeats, [&]() { octreeLinearBuild(); });
            appendToCsv("linear", "build", "NA", -1.0, statsLinear);
        }

        template<Kernel_t kernel>
        void benchmarkSearchNeigh(size_t repeats, float radius) {
            const auto kernelStr = kernelToString(kernel);

            auto statsPointer = benchmarking::benchmark(repeats, [&]() { pointerOctreeSearchNeigh<kernel>(radius); });
            appendToCsv("pointer", "neighSearch", kernelStr, radius, statsPointer);

            auto statsLinear = benchmarking::benchmark(repeats, [&]() { linearOctreeNeighborSearch<kernel>(radius); });
            appendToCsv("linear", "neighSearch", kernelStr, radius, statsLinear);
            
            if (CHECK_RESULTS && checkNeighSearchResults(radius))
                std::cout << "All neighbors search methods yield the same results!" << std::endl;
        }

        template<Kernel_t kernel>
        void benchmarkNumNeigh(size_t repeats, float radius) {
            const auto kernelStr = kernelToString(kernel);

            auto statsPointer = benchmarking::benchmark(repeats, [&]() { pointerOctreeNumNeigh<kernel>(radius); });
            appendToCsv("pointer", "numNeighSearch", kernelStr, radius, statsPointer);

            auto statsLinear = benchmarking::benchmark(repeats, [&]() { linearOctreeNumNeigh<kernel>(radius); });
            appendToCsv("linear", "numNeighSearch", kernelStr, radius, statsLinear);
            
            if(CHECK_RESULTS && checkNumNeighResults(radius))
                std::cout << "All count neighbors methods yield the same results!" << std::endl;
        }

        void benchmarkKNN(size_t repeats) {
            auto statsPointer = benchmarking::benchmark(repeats, [&]() { pointerOctreeKNN(); });
            appendToCsv("pointer", "KNN", "NA", -1.0, statsPointer);

            auto statsLinear = benchmarking::benchmark(repeats, [&]() { linearOctreeKNN(); });
            appendToCsv("linear", "KNN", "NA", -1.0, statsLinear);

            if(CHECK_RESULTS && checkKNNSearchResults())
                std::cout << "All KNN search methods yield the same results!" << std::endl;
        }

        void benchmarkRingSearchNeigh(size_t repeats, Vector &innerRadii, Vector &outerRadii) {
            auto statsPointer = benchmarking::benchmark(repeats, [&]() { pointerOctreeRingSearchNeigh(innerRadii, outerRadii); });
            appendToCsv("pointer", "ringNeighSearch", "NA", -1.0, statsPointer);

            auto statsLinear = benchmarking::benchmark(repeats, [&]() { linearOctreeRingSearchNeigh(innerRadii, outerRadii); });
            appendToCsv("linear", "ringNeighSearch", "NA", -1.0, statsLinear);

            if(CHECK_RESULTS && checkRingNeighSearchResults(innerRadii, outerRadii))
                std::cout << "All KNN search methods yield the same results!" << std::endl; 
        }
};
