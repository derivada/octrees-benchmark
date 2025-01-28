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
                            const std::string& kernel, const float radius, const benchmarking::Stats<>& stats, size_t averageResultSize = 0) {
            // Check if the file is empty and append header if it is
            if (outputFile.tellp() == 0) {
                outputFile << "date,octree,point_type,encoder,npoints,operation,kernel,radius,num_searches,repeats,accumulated,mean,median,stdev,used_warmup,warmup_time,avg_result_size\n";
            }
            // append the comment to the octree name if needed
            std::string octreeName = getOctreeName<Octree_t>();
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
                << averageResultSize << std::endl;
        }

    public:
        OctreeBenchmark(std::vector<Point_t>& points, std::optional<std::reference_wrapper<std::vector<PointMetadata>>> metadata = std::nullopt,
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
            std::cout << "Running " << bench_name << " benchmark on " << getOctreeName<Octree_t>() << " with points " << getPointName<Point_t>() << 
                " and encoder " << PointEncoding::getEncoderName<Encoder_t>() << "\nParameters:\n";
            std::cout << "  Radii: {";
            for(int i = 0; i<benchmarkRadii.size(); i++) {
                std::cout << benchmarkRadii[i];
                if(i != benchmarkRadii.size()-1) {
                    std::cout << ", ";
                }
            }
            std::cout << "}\n";
            std::cout << "  Number of searches: " << numSearches << "\n";
            std::cout << "  Repeats: " << repeats << "\n";
            std::cout << "  Warmup: " << (useWarmup ? "enabled" : "disabled") << "\n";
            std::cout << "  Parallel execution: " << (useParallel ? "enabled" : "disabled") << "\n";
            std::cout << "  Search set point distribution: " << (searchSet->isSequential ? "sequential" : "random") << "\n";
            std::cout << std::endl;
        }

        static void printBenchmarkUpdate(const std::string &method, const size_t totalExecutions, size_t &currentExecution, const float radius) {
            std::cout << getCurrentDate("[%H:%M:%S]") << " (" << currentExecution << "/" << totalExecutions << ") " << method << " with radius " << radius << " done" << std::endl;
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
        }

        std::shared_ptr<const SearchSet> getSearchSet() const { return searchSet; }
        std::shared_ptr<ResultSet<Point_t>> getResultSet() const { return resultSet; }
};