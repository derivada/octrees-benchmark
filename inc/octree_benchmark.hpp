#pragma once
#include <omp.h>
#include <type_traits>
#include "benchmarking.hpp"
#include "NeighborKernels/KernelFactory.hpp"
#include "octree.hpp"
#include "linear_octree.hpp"
#include "type_names.hpp"
#include "TimeWatcher.hpp"
#include "result_set.hpp"
#include "search_set.hpp"

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
            size_t total = benchmarkRadii.size() * 5;
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

                benchmarkSearchNeighStruct<Kernel_t::sphere>(repeats, benchmarkRadii[i]);
                benchmarkSearchNeighStruct<Kernel_t::circle>(repeats, benchmarkRadii[i]);
                benchmarkSearchNeighStruct<Kernel_t::cube>(repeats, benchmarkRadii[i]);
                benchmarkSearchNeighStruct<Kernel_t::square>(repeats, benchmarkRadii[i]);
                printBenchmarkUpdate("Neighbor search - struct impl.", total, current, benchmarkRadii[i]);
                
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
                benchmarkSearchNeighStruct<Kernel_t::sphere>(repeats, benchmarkRadii[i]);
                benchmarkSearchNeighStruct<Kernel_t::circle>(repeats, benchmarkRadii[i]);
                benchmarkSearchNeighStruct<Kernel_t::cube>(repeats, benchmarkRadii[i]);
                benchmarkSearchNeighStruct<Kernel_t::square>(repeats, benchmarkRadii[i]);
                printBenchmarkUpdate("Neighbor search - struct impl.", total, current, benchmarkRadii[i]);
                for(size_t j = 0; j<tolerancePercentages.size(); j++) {
                    benchmarkSearchNeighApprox<Kernel_t::sphere>(repeats, benchmarkRadii[i], tolerancePercentages[j], false);
                    benchmarkSearchNeighApprox<Kernel_t::sphere>(repeats, benchmarkRadii[i], tolerancePercentages[j], true);
                    benchmarkSearchNeighApprox<Kernel_t::circle>(repeats, benchmarkRadii[i], tolerancePercentages[j], false);
                    benchmarkSearchNeighApprox<Kernel_t::circle>(repeats, benchmarkRadii[i], tolerancePercentages[j], true);
                    benchmarkSearchNeighApprox<Kernel_t::cube>(repeats, benchmarkRadii[i], tolerancePercentages[j], false);
                    benchmarkSearchNeighApprox<Kernel_t::cube>(repeats, benchmarkRadii[i], tolerancePercentages[j], true);
                    benchmarkSearchNeighApprox<Kernel_t::square>(repeats, benchmarkRadii[i], tolerancePercentages[j], false);
                    benchmarkSearchNeighApprox<Kernel_t::square>(repeats, benchmarkRadii[i], tolerancePercentages[j], true);
                    printBenchmarkUpdate(std::string("Neighbor search - Approximated with tol. = ") 
                            + std::to_string(tolerancePercentages[j]) + std::string("%"),
                            total, current, benchmarkRadii[i]);
                }
                
            }
        }

        std::shared_ptr<const SearchSet> getSearchSet() const { return searchSet; }
        std::shared_ptr<ResultSet<Point_t>> getResultSet() const { return resultSet; }
};