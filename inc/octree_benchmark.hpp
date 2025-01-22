#pragma once

#include "benchmarking.hpp"
#include <random>
#include <omp.h>
#include "NeighborKernels/KernelFactory.hpp"
#include "type_names.hpp"

struct SearchSet {
    const size_t numSearches;
    std::vector<Point> searchPoints;
    std::vector<uint32_t> searchKNNLimits;
    constexpr static uint32_t MIN_KNN = 5;
    constexpr static uint32_t MAX_KNN = 100;
    std::mt19937 rng;

    // Random subset of size numSearches (may have repeated points)
    template <PointType Point_t>
    SearchSet(size_t numSearches, const std::vector<Point_t>& points)
        : numSearches(numSearches) {

        rng.seed(42);
        searchPoints.resize(numSearches);
        searchKNNLimits.resize(numSearches);

        std::uniform_int_distribution<size_t> indexDist(0, points.size() - 1);
        std::uniform_int_distribution<size_t> knnDist(MIN_KNN, MAX_KNN);

        for (size_t i = 0; i < numSearches; ++i) {
            searchPoints[i] = points[indexDist(rng)];
            searchKNNLimits[i] = knnDist(rng);
        }
    }

    // Either straight indexing (0, 1, 2, ...) or shuffled permutation
    template <PointType Point_t>
    SearchSet(const std::vector<Point_t>& points, bool sequential)
         : numSearches(points.size()) {
        if(points.size() > 2*1e7) {
            std::cout << "Warning: doing a neighbor search over every point might be too expensive! (" << points.size() << " points!)" << std::endl;
        }
        rng.seed(42);
        searchPoints.resize(numSearches);
        searchKNNLimits.resize(numSearches);

        std::uniform_int_distribution<size_t> knnDist(MIN_KNN, MAX_KNN);
        std::vector<size_t> indices(numSearches);
        std::iota(indices.begin(), indices.end(), 0); // Fill indices with 0, 1, 2, ...

        if (sequential) {
            // Sequential order: use points in original order
            for (size_t i = 0; i < numSearches; ++i) {
                searchPoints[i] = points[indices[i]];
                searchKNNLimits[i] = knnDist(rng);
            }
        } else {
            // Random permutation: shuffle indices
            std::shuffle(indices.begin(), indices.end(), rng);
            for (size_t i = 0; i < numSearches; ++i) {
                searchPoints[i] = points[indices[i]];
                searchKNNLimits[i] = knnDist(rng);
            }
        }
    }
};

template <PointType Point_t>
struct ResultSet {
    const std::shared_ptr<const SearchSet> searchSet;
    std::vector<std::vector<Point_t*>> resultsNeigh;
    std::vector<std::vector<Point_t*>> resultsNeighOld;
    std::vector<size_t> resultsNumNeigh;
    std::vector<size_t> resultsNumNeighOld;
    std::vector<std::vector<Point_t*>> resultsKNN;
    std::vector<std::vector<Point_t*>> resultsRingNeigh;

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
};


template <template <typename, typename> class Octree_t, PointType Point_t, typename Encoder_t>
class OctreeBenchmark {
    private:
        const std::unique_ptr<Octree_t<Point_t, Encoder_t>> oct;
        const std::string comment;

        std::vector<Point_t>& points;
        std::ofstream &outputFile;
        
        const bool check;
        const std::shared_ptr<const SearchSet> searchSet;
        std::shared_ptr<ResultSet<Point_t>> resultSet;

        #pragma GCC push_options
        #pragma GCC optimize("O0")
        void preventOptimization(size_t value) {
            volatile size_t* dummy = &value;
            (void) *dummy;
        }
        #pragma GCC pop_options
        
        // Here we add some logging since this is used on the benchmark which goes through every point, which takes a while in big datasets
        // It will make times less accurate since we flush but it doesn't matter that much since we do it for both sequential and shuffled result sets
        template<Kernel_t kernel>
        size_t searchNeighSeq(float radii) {
            if(check && resultSet->resultsNeigh.empty())
                resultSet->resultsNeigh.resize(searchSet->numSearches);
            size_t averageResultSize = 0;
            int last_prog = 0;
            for(size_t i = 0; i<searchSet->numSearches; i++) {
                float progress =  ((float) i/searchSet->numSearches) * 100.0f;
                if(progress > last_prog){
                    last_prog++;
                    std::cout << last_prog << "% done" << std::endl;
                }
                auto result = oct->template searchNeighbors<kernel>(searchSet->searchPoints[i], radii);
                averageResultSize += result.size();
                if(check)
                    resultSet->resultsNeigh[i] = result;
            }
            
            averageResultSize = averageResultSize / searchSet->numSearches;
            return averageResultSize;
        }

        template<Kernel_t kernel>
        size_t searchNeighParallel(float radii) {
            if(check && resultSet->resultsNeigh.empty())
                resultSet->resultsNeigh.resize(searchSet->numSearches);
            size_t averageResultSize = 0;
            #pragma omp parallel for schedule(static) reduction(+:averageResultSize)
                for(size_t i = 0; i<searchSet->numSearches; i++) {
                    auto result = oct->template searchNeighbors<kernel>(searchSet->searchPoints[i], radii);
                    averageResultSize += result.size();
                    if(check)
                        resultSet->resultsNeigh[i] = result;
                }
            
            averageResultSize = averageResultSize / searchSet->numSearches;
            return averageResultSize;
        }

        template<Kernel_t kernel>
        size_t searchNeighOldParallel(float radii) {
            if(check && resultSet->resultsNeighOld.empty())
                resultSet->resultsNeighOld.resize(searchSet->numSearches);
            size_t averageResultSize = 0;
            #pragma omp parallel for schedule(static) reduction(+:averageResultSize)
                for(size_t i = 0; i<searchSet->numSearches; i++) {
                    auto result = oct->template searchNeighborsOld<kernel>(searchSet->searchPoints[i], radii);
                    averageResultSize += result.size();
                    if(check)
                        resultSet->resultsNeighOld[i] = result;
                }
            averageResultSize = averageResultSize / searchSet->numSearches;
            return averageResultSize;
        }

        template<Kernel_t kernel>
        size_t numNeighParallel(float radii) {
            if(check && resultSet->resultsNumNeigh.empty())
                resultSet->resultsNumNeigh.resize(searchSet->numSearches);
            size_t averageResultSize = 0;
            #pragma omp parallel for schedule(static) reduction(+:averageResultSize)
                for(size_t i = 0; i<searchSet->numSearches; i++) {
                    auto result = oct->template numNeighbors<kernel>(searchSet->searchPoints[i], radii);
                    averageResultSize += result;
                    if(check)
                        resultSet->resultsNumNeigh[i] = result;
                }
            averageResultSize = averageResultSize / searchSet->numSearches;
            return averageResultSize;
        }

        template<Kernel_t kernel>
        size_t numNeighOldParallel(float radii) {
            if(check && resultSet->resultsNumNeighOld.empty())
                resultSet->resultsNumNeighOld.resize(searchSet->numSearches);
            size_t averageResultSize = 0;
            #pragma omp parallel for schedule(static) reduction(+:averageResultSize)
                for(size_t i = 0; i<searchSet->numSearches; i++) {
                    auto result = oct->template numNeighborsOld<kernel>(searchSet->searchPoints[i], radii);
                    averageResultSize += result;
                    if(check)
                        resultSet->resultsNumNeighOld[i] = result;
                }
            averageResultSize = averageResultSize / searchSet->numSearches;
            return averageResultSize;
        }

        void KNNParallel() {
            if(check && resultSet->resultsKNN.empty())
                resultSet->resultsKNN.resize(searchSet->numSearches);
            #pragma omp parallel for schedule(static)
                for(size_t i = 0; i<searchSet->numSearches; i++) {
                    if(check) {
                        resultSet->resultsKNN[i] = oct->template KNN(searchSet->searchPoints[i], searchSet->searchKNNLimits[i], searchSet->searchKNNLimits[i]);
                    } else {
                        preventOptimization(oct->template KNN(searchSet->searchPoints[i], searchSet->searchKNNLimits[i], searchSet->searchKNNLimits[i]));
                    }
                }
        }

        size_t ringNeighSearchParallel(Vector &innerRadii, Vector &outerRadii) {
            if(check && resultSet->resultsRingNeigh.empty())
                resultSet->resultsRingNeigh.resize(searchSet->numSearches);
            size_t averageResultSize = 0;
            #pragma omp parallel for schedule(static) reduction(+:averageResultSize)
                for(size_t i = 0; i<searchSet->numSearches; i++) {
                    auto result = oct->template searchNeighborsRing(searchSet->searchPoints[i], innerRadii, outerRadii);
                    averageResultSize += result.size();
                    if(check)
                        resultSet->resultsRingNeigh[i] = result;
                }
            averageResultSize = averageResultSize / searchSet->numSearches;
            return averageResultSize;
        }

        inline void appendToCsv(const std::string& operation, 
                            const std::string& kernel, const float radius, const benchmarking::Stats<>& stats, size_t averageResultSize = 0) {
            // Check if the file is empty and append header if it is
            if (outputFile.tellp() == 0) {
                outputFile << "date,octree,point_type,encoder,npoints,operation,kernel,radius,num_searches,repeats,accumulated,mean,median,stdev,used_warmup,warmup_time,avg_result_size\n";
            }
            std::string octreeName = getOctreeName<Octree_t>();
            std::string pointTypeName = getPointName<Point_t>();
            std::string encoderTypename = PointEncoding::getEncoderName<Encoder_t>();
            // Append the benchmark data
            outputFile << getCurrentDate() << ',' 
                << octreeName << ',' 
                << pointTypeName << ','
                << encoderTypename << ','
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
                << stats.usedWarmup() << ','
                << stats.warmupValue() << ','
                << averageResultSize << std::endl;
        }

    public:
        OctreeBenchmark(std::vector<Point_t>& points, size_t numSearches = 100, std::shared_ptr<const SearchSet> searchSet = nullptr, std::ofstream &file = std::ofstream(), bool check = false,
            std::string comment = "") :
            points(points), 
            oct(std::make_unique<Octree_t<Point_t, Encoder_t>>(points)),
            searchSet(searchSet ? searchSet : std::make_shared<const SearchSet>(numSearches, points)),
            outputFile(file),
            check(check),
            comment(comment),
            resultSet(std::make_shared<ResultSet<Point_t>>(searchSet)) { }
        
        template<Kernel_t kernel>
        void benchmarkSearchNeighSeq(size_t repeats, float radius) {
            const auto kernelStr = kernelToString(kernel);
            // note: the seq on searchNeighSeq means that execution is not parallelized, but the searchSets here are either sequential or 
            // shuffled meaning that indexes are 0,1,2,3... or a random permutation (see 2nd constructor of SearchSet)
            auto [stats, averageResultSize] = benchmarking::benchmark<size_t>(repeats, [&]() { return searchNeighSeq<kernel>(radius); });
            // here the field comment is used to differentiate between sequential point order and shuffled order execution
            appendToCsv("neighSearch " + comment, kernelStr, radius, stats, averageResultSize);
        }

        template<Kernel_t kernel>
        void benchmarkSearchNeigh(size_t repeats, float radius) {
            const auto kernelStr = kernelToString(kernel);
            auto [stats, averageResultSize] = benchmarking::benchmark<size_t>(repeats, [&]() { return searchNeighParallel<kernel>(radius); });
            appendToCsv("neighSearch", kernelStr, radius, stats, averageResultSize);
        }

        template<Kernel_t kernel>
        void benchmarkSearchNeighOld(size_t repeats, float radius) {
            const auto kernelStr = kernelToString(kernel);
            auto [stats, averageResultSize] = benchmarking::benchmark<size_t>(repeats, [&]() { return searchNeighOldParallel<kernel>(radius); });
            appendToCsv("neighOldSearch", kernelStr, radius, stats, averageResultSize);
        }

        template<Kernel_t kernel>
        void benchmarkNumNeigh(size_t repeats, float radius) {
            const auto kernelStr = kernelToString(kernel);
            auto [stats, averageResultSize] = benchmarking::benchmark<size_t>(repeats, [&]() { return numNeighParallel<kernel>(radius); });
            appendToCsv("numNeighSearch", kernelStr, radius, stats, averageResultSize);
        }

        template<Kernel_t kernel>
        void benchmarkNumNeighOld(size_t repeats, float radius) {
            const auto kernelStr = kernelToString(kernel);
            auto [stats, averageResultSize] = benchmarking::benchmark<size_t>(repeats, [&]() { return numNeighOldParallel<kernel>(radius); });
            appendToCsv("numNeighOldSearch", kernelStr, radius, stats, averageResultSize);
        }


        void benchmarkKNN(size_t repeats) {
            auto [stats, averageResultSize] = benchmarking::benchmark(repeats, [&]() { KNNParallel(); });
            appendToCsv("KNN", "NA", -1.0, stats);
        }

        void benchmarkRingSearchNeigh(size_t repeats, Vector &innerRadii, Vector &outerRadii) {
            auto [stats, averageResultSize] = benchmarking::benchmark<size_t>(repeats, [&]() { return ringNeighSearchParallel(innerRadii, outerRadii); });
            appendToCsv("ringNeighSearch", "NA", -1.0, stats, averageResultSize);
        }

        static void printBenchmarkLog(const std::string &bench_name, const std::vector<float> &benchmarkRadii, const size_t repeats, const size_t numSearches) {
            std::cout << "Running " << bench_name << " benchmark on " << getOctreeName<Octree_t>() << " with points " << getPointName<Point_t>() << 
                " and encoder " << PointEncoding::getEncoderName<Encoder_t>() << std::endl << "Parameters:" << std::endl;
            for(int i = 0; i<benchmarkRadii.size(); i++) {
                std::cout << benchmarkRadii[i];
                if(i != benchmarkRadii.size()-1) {
                std::cout << ", ";
                }
            }
            std::cout << "}" << std::endl;
            std::cout << "  Number of searches: " << numSearches << std::endl;
            std::cout << "  Repeats: " << repeats << std::endl << std::endl;
        }

        static void printBenchmarkUpdate(const std::string &method, const size_t totalExecutions, size_t &currentExecution, const float radius) {
            std::cout << getCurrentDate("[%H:%M:%S]") << " (" << currentExecution << "/" << totalExecutions << ") " << method << " with radius " << radius << " done" << std::endl;
            currentExecution++;
        }

        void searchImplComparisonBench(const std::vector<float> &benchmarkRadii, const size_t repeats, const size_t numSearches) {
            printBenchmarkLog("search and numsearch implementation comparison", benchmarkRadii, repeats, numSearches);
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
        }

        void searchBench(const std::vector<float> &benchmarkRadii, const size_t repeats, const size_t numSearches) {
            printBenchmarkLog("search and numsearch", benchmarkRadii, repeats, numSearches);
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
        }

        void benchSearchNeighSeq(const std::vector<float> &benchmarkRadii, const size_t repeats, const size_t numSearches) {
            printBenchmarkLog("search neighbors sequential", benchmarkRadii, repeats, numSearches);

            size_t total = benchmarkRadii.size() * 2;
            size_t current = 1;
            for(size_t i = 0; i<benchmarkRadii.size(); i++) {
                benchmarkSearchNeighSeq<Kernel_t::sphere>(repeats, benchmarkRadii[i]);
                benchmarkSearchNeighSeq<Kernel_t::circle>(repeats, benchmarkRadii[i]);
                benchmarkSearchNeighSeq<Kernel_t::cube>(repeats, benchmarkRadii[i]);
                benchmarkSearchNeighSeq<Kernel_t::square>(repeats, benchmarkRadii[i]);
                printBenchmarkUpdate("Neighbor search sequential", total, current, benchmarkRadii[i]);
            }
        }

        std::shared_ptr<const SearchSet> getSearchSet() const { return searchSet; }
        std::shared_ptr<ResultSet<Point_t>> getResultSet() const { return resultSet; }
};