#pragma once

#include "benchmarking.hpp"
#include <random>
#include <omp.h>
#include "NeighborKernels/KernelFactory.hpp"
#include "octree_factory.hpp"
#include "search_set.hpp"

template <OctreeType Octree_t, PointType Point_t>
class OctreeBenchmark {
    private:

        const std::unique_ptr<Octree_t> oct;
        const std::string comment;

        std::vector<Point_t>& points;
        std::ofstream &outputFile;
        
        void rebuild() {
            oct = std::make_unique<Octree_t>(points); 
        }

        template<Kernel_t kernel>
        void searchNeighParallel(float radii) {
            if(check && resultsNeigh.empty())
                resultsNeigh.resize(searchSet->numSearches);
            #pragma omp parallel for schedule(static)
                for(size_t i = 0; i<searchSet->numSearches; i++) {
                    if(check){
                        resultsNeigh[i] = oct->template searchNeighbors<kernel>(searchSet->searchPoints[i], radii);
                    } else{
                        volatile auto result = oct->template searchNeighbors<kernel>(searchSet->searchPoints[i], radii);
                    }
                }
        }

        template<Kernel_t kernel>
        void numNeighParallel(float radii) {
            if(check && resultsNumNeigh.empty())
                resultsNumNeigh.resize(searchSet->numSearches);
            #pragma omp parallel for schedule(static)
                for(size_t i = 0; i<searchSet->numSearches; i++) {
                    if(check) {
                        resultsNumNeigh[i] = oct->template numNeighbors<kernel>(searchSet->searchPoints[i], radii);
                    } else {
                        volatile auto result =  oct->template numNeighbors<kernel>(searchSet->searchPoints[i], radii);
                    }
                }
        }

        void KNNParallel() {
            if(check && resultsKNN.empty())
                resultsKNN.resize(searchSet->numSearches);
            #pragma omp parallel for schedule(static)
                for(size_t i = 0; i<searchSet->numSearches; i++) {
                    if(check) {
                        resultsKNN[i] = oct->template KNN(searchSet->searchPoints[i], searchSet->searchKNNLimits[i], searchSet->searchKNNLimits[i]);
                    } else {
                        volatile auto result =  oct->template KNN(searchSet->searchPoints[i], searchSet->searchKNNLimits[i], searchSet->searchKNNLimits[i]);
                    }
                }
        }

        void ringNeighSearchParallel(Vector &innerRadii, Vector &outerRadii) {
            if(check && resultsRingNeigh.empty())
                resultsRingNeigh.resize(searchSet->numSearches);
            #pragma omp parallel for schedule(static)
                for(size_t i = 0; i<searchSet->numSearches; i++) {
                    if(check) {    
                        resultsRingNeigh[i] = oct->template searchNeighborsRing(searchSet->searchPoints[i], 
                            innerRadii, outerRadii);
                    } else {
                        volatile auto result = oct->template searchNeighborsRing(searchSet->searchPoints[i], 
                            innerRadii, outerRadii);
                    }
                }
        }

        inline void appendToCsv(const std::string& operation, 
                            const std::string& kernel, const float radius, const benchmarking::Stats<>& stats) {
            // Check if the file is empty and append header if it is
            if (outputFile.tellp() == 0) {
                outputFile << "date,octree,npoints,operation,kernel,radius,num_searches,repeats,accumulated,mean,median,stdev,used_warmup\n";
            }
            std::string octreeName = getOctreeName<Octree_t, Point_t>() + " " + comment;
            // Append the benchmark data
            outputFile << getCurrentDate() << ',' 
                << octreeName << ',' 
                << points.size() << ','
                << operation << ',' 
                << kernel << ',' 
                << radius << ','
                << searchSet->numSearches << ',' 
                << stats.size() << ','
                << stats.accumulated() << ',' 
                << stats.mean() << ',' 
                << stats.median() << ',' 
                << stats.stdev() << ','
                << stats.usedWarmup() << '\n';
        }

        // Generic check for neighbor results
        template <typename Point_t2>
        static std::vector<size_t> checkNeighResults(std::vector<std::vector<Point_t2*>> &results1, std::vector<std::vector<Point_t2*>> &results2)
        {
            std::vector<size_t> wrongSearches;
            for (size_t i = 0; i < results1.size(); i++) {
                auto v1 = results1[i];
                auto v2 = results2[i];
                if (v1.size() != v2.size()) {
                    wrongSearches.push_back(i);
                } else {
                    std::sort(v1.begin(), v1.end(), [](Point_t2 *p, Point_t2* q) -> bool {
                        return p->id() < q->id();
                    });
                    std::sort(v2.begin(), v2.end(), [](Point_t2 *p, Point_t2* q) -> bool {
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

        // Generic check for the number of neighbors
        static std::vector<size_t> checkNumNeighResults(std::vector<size_t> &results1, std::vector<size_t> &results2)
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

        // Operation for checking neighbor results
        template <typename Point_t2>
        static void checkOperationNeigh(
            std::vector<std::vector<Point_t2*>> &results1,
            std::vector<std::vector<Point_t2*>> &results2,
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

        // Operation for checking number of neighbor results
        static void checkOperationNumNeigh(
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

    public:
        using PointType = Point_t;
        const bool check = false;
        const std::shared_ptr<const SearchSet> searchSet;
        std::vector<std::vector<Point_t*>> resultsNeigh;
        std::vector<size_t> resultsNumNeigh;
        std::vector<std::vector<Point_t*>> resultsKNN;
        std::vector<std::vector<Point_t*>> resultsRingNeigh;

        OctreeBenchmark(std::vector<Point_t>& points, size_t numSearches = 100, std::shared_ptr<const SearchSet> searchSet = nullptr, std::ofstream &file = std::ofstream(), bool check = false,
            std::string comment = "") :
            points(points), 
            oct(std::make_unique<Octree_t>(points)),
            searchSet(searchSet ? searchSet : std::make_shared<const SearchSet>(numSearches, points)),
            outputFile(file),
            check(check),
            comment(comment) { }

        void benchmarkBuild(size_t repeats) {
            auto stats = benchmarking::benchmark(repeats, [&]() { rebuild(); });
            appendToCsv("build", "NA", -1.0, stats);
        }

        template<Kernel_t kernel>
        void benchmarkSearchNeigh(size_t repeats, float radius) {
            const auto kernelStr = kernelToString(kernel);
            auto stats = benchmarking::benchmark(repeats, [&]() { searchNeighParallel<kernel>(radius); });
            appendToCsv("neighSearch", kernelStr, radius, stats);
        }

        template<Kernel_t kernel>
        void benchmarkNumNeigh(size_t repeats, float radius) {
            const auto kernelStr = kernelToString(kernel);
            auto stats = benchmarking::benchmark(repeats, [&]() { numNeighParallel<kernel>(radius); });
            appendToCsv("numNeighSearch", kernelStr, radius, stats);
        }

        void benchmarkKNN(size_t repeats) {
            auto stats = benchmarking::benchmark(repeats, [&]() { KNNParallel(); });
            appendToCsv("KNN", "NA", -1.0, stats);
        }

        void benchmarkRingSearchNeigh(size_t repeats, Vector &innerRadii, Vector &outerRadii) {
            auto stats = benchmarking::benchmark(repeats, [&]() { ringNeighSearchParallel(innerRadii, outerRadii); });
            appendToCsv("ringNeighSearch", "NA", -1.0, stats);
        }

        static void runFullBenchmark(OctreeBenchmark &ob, const std::vector<float> &benchmarkRadii, const size_t repeats, const size_t numSearches) {
            std::cout << "Running octree benchmark on " << getOctreeName<Octree_t, Point_t>() << " with parameters:" << std::endl;
            std::cout << "  Search radii: {";
            for(int i = 0; i<benchmarkRadii.size(); i++) {
                std::cout << benchmarkRadii[i];
                if(i != benchmarkRadii.size()-1) {
                std::cout << ", ";
                }
            }
            std::cout << "}" << std::endl;
            std::cout << "  Number of searches: " << numSearches << std::endl;
            std::cout << "  Repeats: " << repeats << std::endl << std::endl;

            size_t total = benchmarkRadii.size() * 2;
            for(int i = 0; i<benchmarkRadii.size(); i++) {
                ob.benchmarkSearchNeigh<Kernel_t::sphere>(repeats, benchmarkRadii[i]);
                ob.benchmarkSearchNeigh<Kernel_t::circle>(repeats, benchmarkRadii[i]);
                ob.benchmarkSearchNeigh<Kernel_t::cube>(repeats, benchmarkRadii[i]);
                ob.benchmarkSearchNeigh<Kernel_t::square>(repeats, benchmarkRadii[i]);
                std::cout << "(" << (i+1) << "/" << total << ") Benchmark search neighbors with radii " << benchmarkRadii[i] << " completed" << std::endl;
            }

            for(int i = 0; i<benchmarkRadii.size(); i++) {
                ob.benchmarkNumNeigh<Kernel_t::sphere>(repeats, benchmarkRadii[i]);
                ob.benchmarkNumNeigh<Kernel_t::circle>(repeats, benchmarkRadii[i]);
                ob.benchmarkNumNeigh<Kernel_t::cube>(repeats, benchmarkRadii[i]);
                ob.benchmarkNumNeigh<Kernel_t::square>(repeats, benchmarkRadii[i]);
                std::cout << "(" << (i+1+benchmarkRadii.size()) << "/" << total << ") Benchmark number of neighbors with radii " << benchmarkRadii[i] << " completed" << std::endl;
            }

            // TODO: fix the implementation of this other two benchmarks
            // ob.benchmarkKNN(5);
            // ob.benchmarkRingSearchNeigh(5);

            std::cout << "Benchmark done!" << std::endl << std::endl;
        }
    
    template <typename Octree_t1, typename Octree_t2>
    static void checkResults(
        OctreeBenchmark<Octree_t1, typename Octree_t1::PointType> &bench1,
        OctreeBenchmark<Octree_t2, typename Octree_t2::PointType> &bench2,
        size_t printingLimit = 10) {
            // Ensure the search sets are the same
            assert(bench1.searchSet == bench2.searchSet && "The search sets of the benchmarks are not the same");
            assert(bench1.check && bench2.check && "The octrees did not save results for checking");
            
            // Check neighbor search results if available
            if (!bench1.resultsNeigh.empty() && !bench2.resultsNeigh.empty()) {
                std::cout << "Checking search results for neighbor searches..." << std::endl;
                checkOperationNeigh<typename Octree_t1::PointType>(bench1.resultsNeigh, bench2.resultsNeigh, printingLimit);
            }

            // Check number of neighbor search results if available
            if (!bench1.resultsNumNeigh.empty() && !bench2.resultsNumNeigh.empty()) {
                std::cout << "Checking search results for number of neighbor searches..." << std::endl;
                checkOperationNumNeigh(bench1.resultsNumNeigh, bench2.resultsNumNeigh, printingLimit);
            }
    }
};
