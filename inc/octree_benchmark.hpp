#pragma once
#include <omp.h>
#include <type_traits>
#include "benchmarking.hpp"
#include "NeighborKernels/KernelFactory.hpp"
#include "octree.hpp"
#include "linear_octree.hpp"
#include "type_names.hpp"
#include "TimeWatcher.hpp"
#include "result_checking.hpp"
#include "search_set.hpp"
#include "main_options.hpp"

using namespace ResultChecking;

template <template <typename> class Octree_t, typename Point_t>
class OctreeBenchmark {
    private:
        using PointEncoder = PointEncoding::PointEncoder;
        using key_t = PointEncoding::key_t;
        using coords_t = PointEncoding::coords_t;
        std::unique_ptr<Octree_t<Point_t>> oct;
        PointEncoder& enc;
        const std::string comment;

        std::vector<Point_t>& points;
        std::ofstream &outputFile;
        
        const bool checkResults, useWarmup;
        const SearchSet<Point_t> &searchSet;
        ResultSet<Point_t> resultSet;

        #pragma GCC push_options
        #pragma GCC optimize("O0")
        void preventOptimization(size_t value) {
            volatile size_t* dummy = &value;
            (void) *dummy;
        }
        #pragma GCC pop_options
        
        template<Kernel_t kernel>
        size_t searchNeigh(float radii) {
            size_t averageResultSize = 0;
            #pragma omp parallel for schedule(runtime) reduction(+:averageResultSize)
                for(size_t i = 0; i<searchSet.numSearches; i++) {
                    auto result = oct->template searchNeighbors<kernel>(points[searchSet.searchPoints[i]], radii);
                    averageResultSize += result.size();
                    if(checkResults)
                        resultSet.resultsNeigh[i] = std::move(result);
                }
            
            averageResultSize = averageResultSize / searchSet.numSearches;
            return averageResultSize;
        }

        template<Kernel_t kernel>
        size_t searchNeighStruct(float radii) {
            size_t averageResultSize = 0;
            #pragma omp parallel for schedule(runtime) reduction(+:averageResultSize)
                for(size_t i = 0; i<searchSet.numSearches; i++) {
                    auto result = oct->template searchNeighborsStruct<kernel>(points[searchSet.searchPoints[i]], radii);
                    averageResultSize += result.size();
                    if(checkResults)
                        resultSet.resultsNeighStruct[i] = std::move(result);
                }
            averageResultSize = averageResultSize / searchSet.numSearches;
            return averageResultSize;
        }

        template<Kernel_t kernel>
        size_t searchNeighApprox(float radii, double tolerancePercentage, bool upperBound) {
            resultSet.tolerancePercentageUsed = tolerancePercentage;
            size_t averageResultSize = 0;
            #pragma omp parallel for schedule(runtime) reduction(+:averageResultSize)
                for(size_t i = 0; i<searchSet.numSearches; i++) {
                    auto result = oct->template searchNeighborsApprox<kernel>(points[searchSet.searchPoints[i]], radii, tolerancePercentage, upperBound);
                    averageResultSize += result.size();
                    if(checkResults) {
                        if(upperBound)
                            resultSet.resultsSearchApproxUpper[i] = std::move(result);
                        else
                            resultSet.resultsSearchApproxLower[i] = std::move(result);
                    }
                }
            
            averageResultSize = averageResultSize / searchSet.numSearches;
            return averageResultSize;
        }

        template<Kernel_t kernel>
        size_t searchNeighOld(float radii) {
            size_t averageResultSize = 0;
            #pragma omp parallel for schedule(runtime) reduction(+:averageResultSize)
                for(size_t i = 0; i<searchSet.numSearches; i++) {
                    auto result = oct->template searchNeighborsOld<kernel>(points[searchSet.searchPoints[i]], radii);
                    averageResultSize += result.size();
                    if(checkResults)
                        resultSet.resultsNeighOld[i] = std::move(result);
                }
            averageResultSize = averageResultSize / searchSet.numSearches;
            return averageResultSize;
        }

        template<Kernel_t kernel>
        size_t searchNumNeigh(float radii) {
            size_t averageResultSize = 0;
            #pragma omp parallel for schedule(runtime) reduction(+:averageResultSize)
                for(size_t i = 0; i<searchSet.numSearches; i++) {
                    auto result = oct->template numNeighbors<kernel>(points[searchSet.searchPoints[i]], radii);
                    averageResultSize += result;
                    if(checkResults)
                        resultSet.resultsNumNeigh[i] = result;
                }
            averageResultSize = averageResultSize / searchSet.numSearches;
            return averageResultSize;
        }

        template<Kernel_t kernel>
        size_t searchNumNeighOld(float radii) {
            size_t averageResultSize = 0;
            #pragma omp parallel for schedule(runtime) reduction(+:averageResultSize)
                for(size_t i = 0; i<searchSet.numSearches; i++) {
                    auto result = oct->template numNeighborsOld<kernel>(points[searchSet.searchPoints[i]], radii);
                    averageResultSize += result;
                    if(checkResults)
                        resultSet.resultsNumNeighOld[i] = result;
                }
            averageResultSize = averageResultSize / searchSet.numSearches;
            return averageResultSize;
        }


        inline void appendToCsv(const std::string& operation, const std::string& kernel, const float radius, const benchmarking::Stats<>& stats, 
                                size_t averageResultSize = 0, int numThreads = omp_get_max_threads(), double tolerancePercentage = 0.0) {
            // Check if the file is empty and append header if it is
            if (outputFile.tellp() == 0) {
                outputFile <<   "date,octree,point_type,encoder,npoints,operation,kernel,radius,num_searches,sequential_searches,repeats,"
                                "accumulated,mean,median,stdev,used_warmup,warmup_time,avg_result_size,tolerance_percentage,"
                                "openmp_threads,openmp_schedule\n";
            }
            // append the comment to the octree name if needed
            std::string octreeName;
            if constexpr (std::is_same_v<Octree_t<Point_t>, LinearOctree<Point_t>>) {
                octreeName = "LinearOctree";
            } else if constexpr (std::is_same_v<Octree_t<Point_t>, Octree<Point_t>>) {
                octreeName = "Octree";
            }

            std::string pointTypeName = getPointName<Point_t>();
            std::string encoderName = enc.getEncoderName();
            // if the comment, exists, append it to the op. name
            std::string fullOp = operation + ((comment != "") ? "_" + comment : "");

            // Get OpenMP runtime information
            omp_sched_t openmpSchedule;
            int openmpChunkSize;
            omp_get_schedule(&openmpSchedule, &openmpChunkSize);
            std::string openmpScheduleName;
            switch (openmpSchedule) {
                case omp_sched_static: openmpScheduleName = "static"; break;
                case omp_sched_dynamic: openmpScheduleName = "dynamic"; break;
                case omp_sched_guided: openmpScheduleName = "guided"; break;
                default: openmpScheduleName = "unknown"; break;
            }
            std::string sequentialSearches;
            if(searchSet.sequential) {
                sequentialSearches = "sequential";
            } else {
                sequentialSearches = "random";
            }
            outputFile << getCurrentDate() << ',' 
                << octreeName << ',' 
                << pointTypeName << ','
                << enc.getEncoderName() << ','
                << points.size() << ','
                << fullOp << ',' 
                << kernel << ',' 
                << radius << ','
                << searchSet.numSearches << ',' 
                << sequentialSearches << ','
                << stats.size() << ','
                << stats.accumulated() << ',' 
                << stats.mean() << ',' 
                << stats.median() << ',' 
                << stats.stdev() << ','
                << stats.usedWarmup() << ','
                << stats.warmupValue() << ','
                << averageResultSize << ','
                << tolerancePercentage << ','
                << numThreads << ','
                << openmpScheduleName
                << std::endl;
        }

    public:
        OctreeBenchmark(std::vector<Point_t>& points, std::vector<key_t>& codes, Box box, PointEncoder& enc, const SearchSet<Point_t>& searchSet, 
            std::ofstream &file, bool checkResults = mainOptions.checkResults, bool useWarmup = mainOptions.useWarmup) :
            points(points), 
            enc(enc),
            searchSet(searchSet),
            outputFile(file),
            checkResults(checkResults),
            useWarmup(useWarmup),
            resultSet(searchSet) { 

            // Conditional initialization of oct based on the type of Octree_t<Point_t>
            if constexpr (std::is_same_v<Octree_t<Point_t>, LinearOctree<Point_t>>) {
                // Initialize for LinearOctree
                oct = std::make_unique<LinearOctree<Point_t>>(points, codes, box, enc);
            } else if constexpr (std::is_same_v<Octree_t<Point_t>, Octree<Point_t>>) {
                // Initialize for Octree
                oct = std::make_unique<Octree<Point_t>>(points, box);
            }
        }
    

        template<Kernel_t kernel>
        void benchmarkSearchNeigh(size_t repeats, float radius, int numThreads = omp_get_max_threads()) {
            omp_set_num_threads(numThreads);
            const auto kernelStr = kernelToString(kernel);
            auto [stats, averageResultSize] = benchmarking::benchmark<size_t>(repeats, [&]() { return searchNeigh<kernel>(radius); }, useWarmup);
            appendToCsv("neighSearch", kernelStr, radius, stats, averageResultSize, numThreads);
        }

        template<Kernel_t kernel>
        void benchmarkSearchNeighStruct(size_t repeats, float radius, int numThreads = omp_get_max_threads()) {
            omp_set_num_threads(numThreads);
            const auto kernelStr = kernelToString(kernel);
            auto [stats, averageResultSize] = benchmarking::benchmark<size_t>(repeats, [&]() { return searchNeighStruct<kernel>(radius); }, useWarmup);
            appendToCsv("neighSearchStruct", kernelStr, radius, stats, averageResultSize, numThreads);
        }

        template<Kernel_t kernel>
        void benchmarkSearchNeighApprox(size_t repeats, float radius, double tolerancePercentage, int numThreads = omp_get_max_threads()) {
            omp_set_num_threads(numThreads);
            const auto kernelStr = kernelToString(kernel);
            // first lower bound, then upper bound
            auto [statsLower, averageResultSizeLower] = benchmarking::benchmark<size_t>(repeats, [&]() { return searchNeighApprox<kernel>(radius, tolerancePercentage, false); }, useWarmup);
            appendToCsv("neighSearchApproxLower", kernelStr, radius, statsLower, averageResultSizeLower, numThreads, tolerancePercentage);
            auto [statsUpper, averageResultSizeUpper] = benchmarking::benchmark<size_t>(repeats, [&]() { return searchNeighApprox<kernel>(radius, tolerancePercentage, true); }, useWarmup);
            appendToCsv("neighSearchApproxUpper", kernelStr, radius, statsUpper, averageResultSizeUpper, numThreads, tolerancePercentage);
        }

        template<Kernel_t kernel>
        void benchmarkSearchNeighOld(size_t repeats, float radius, int numThreads = omp_get_max_threads()) {
            omp_set_num_threads(numThreads);
            const auto kernelStr = kernelToString(kernel);
            auto [stats, averageResultSize] = benchmarking::benchmark<size_t>(repeats, [&]() { return searchNeighOld<kernel>(radius); }, useWarmup);
            appendToCsv("neighOldSearch", kernelStr, radius, stats, averageResultSize, numThreads);
        }

        template<Kernel_t kernel>
        void benchmarkNumNeigh(size_t repeats, float radius, int numThreads = omp_get_max_threads()) {
            omp_set_num_threads(numThreads);
            const auto kernelStr = kernelToString(kernel);
            auto [stats, averageResultSize] = benchmarking::benchmark<size_t>(repeats, [&]() { return searchNumNeigh<kernel>(radius); }, useWarmup);
            appendToCsv("numNeighSearch", kernelStr, radius, stats, averageResultSize, numThreads);
        }

        template<Kernel_t kernel>
        void benchmarkNumNeighOld(size_t repeats, float radius, int numThreads = omp_get_max_threads()) {
            omp_set_num_threads(numThreads);
            const auto kernelStr = kernelToString(kernel);
            auto [stats, averageResultSize] = benchmarking::benchmark<size_t>(repeats, [&]() { return searchNumNeighOld<kernel>(radius); }, useWarmup);
            appendToCsv("numNeighOldSearch", kernelStr, radius, stats, averageResultSize, numThreads);
        }

        void printBenchmarkLog(const std::string &bench_name, const std::vector<float> &benchmarkRadii, const size_t repeats) {
            // Displaying the basic information with formatting
            std::cout << std::fixed << std::setprecision(3);
            std::string octreeName;
            if constexpr (std::is_same_v<Octree_t<Point_t>, LinearOctree<Point_t>>) {
                octreeName = "LinearOctree";
            } else if constexpr (std::is_same_v<Octree_t<Point_t>, Octree<Point_t>>) {
                octreeName = "Octree";
            }
            std::cout << std::left << std::setw(LOG_FIELD_WIDTH) << "Running benchmark:"        << std::setw(LOG_FIELD_WIDTH) << bench_name                      << "\n";
            std::cout << std::left << std::setw(LOG_FIELD_WIDTH) << "Octree used:"              << std::setw(LOG_FIELD_WIDTH) << octreeName        << "\n";
            std::cout << std::left << std::setw(LOG_FIELD_WIDTH) << "Point type:"               << std::setw(LOG_FIELD_WIDTH) << getPointName<Point_t>()           << "\n";
            std::cout << std::left << std::setw(LOG_FIELD_WIDTH) << "Encoder:"                  << std::setw(LOG_FIELD_WIDTH) << enc.getEncoderName() << "\n";
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
            std::cout << std::left << std::setw(LOG_FIELD_WIDTH) << "Number of searches:"       << std::setw(LOG_FIELD_WIDTH) << searchSet.numSearches        << "\n";
            std::cout << std::left << std::setw(LOG_FIELD_WIDTH) << "Kernels:"                  << std::setw(2*LOG_FIELD_WIDTH) << getKernelListString()      << "\n";
            if constexpr (std::is_same_v<Octree_t<Point_t>, LinearOctree<Point_t>>) {
                std::cout << std::left << std::setw(LOG_FIELD_WIDTH) << "Search algorithms:"    << std::setw(2*LOG_FIELD_WIDTH) << getSearchAlgoListString() << "\n";
            } else if constexpr (std::is_same_v<Octree_t<Point_t>, Octree<Point_t>>) {
                std::cout << std::left << std::setw(LOG_FIELD_WIDTH) << "Search algorithms:"    << std::setw(2*LOG_FIELD_WIDTH) << "neighbors" << "\n";
            }            
            std::cout << std::left << std::setw(LOG_FIELD_WIDTH) << "Repeats:"                  << std::setw(LOG_FIELD_WIDTH) << repeats                           << "\n";
            std::cout << std::left << std::setw(LOG_FIELD_WIDTH) << "Warmup:"                   << std::setw(LOG_FIELD_WIDTH) << (useWarmup ? "enabled" : "disabled") << "\n";
            std::cout << std::left << std::setw(LOG_FIELD_WIDTH) << "Search set distribution:"  << std::setw(LOG_FIELD_WIDTH) << (searchSet.sequential ? "sequential" : "random") << "\n";
            
            std::cout << std::endl;
            std::cout << std::left << std::setw(LOG_FIELD_WIDTH/2) << "Progress"    << std::setw(LOG_FIELD_WIDTH/2) << "Completed at" 
                                   << std::setw(LOG_FIELD_WIDTH*2) << "Method"      << std::setw(LOG_FIELD_WIDTH/2) << "Radius" << std::endl;
        }

        static void printBenchmarkUpdate(const std::string &method, const size_t totalExecutions, size_t currentExecution, const float radius) {
            const std::string progress_str = "(" + std::to_string(currentExecution) + "/" + std::to_string(totalExecutions) + ")";
            std::cout << std::left << std::setw(LOG_FIELD_WIDTH/2) << progress_str  << std::setw(LOG_FIELD_WIDTH/2) << getCurrentDate("[%H:%M:%S]") 
                                   << std::setw(LOG_FIELD_WIDTH*2) << method        << std::setw(LOG_FIELD_WIDTH/2) << radius << std::endl;
        }

        template <typename Func>
        void runBenchOverKernels(Func&& benchFunc) {
            // Use the kernels directly from mainOptions
            for (const auto& kernel : mainOptions.kernels) {
                switch (kernel) {
                    case Kernel_t::sphere:
                        benchFunc(Kernel_t::sphere);
                        break;
                    case Kernel_t::circle:
                        benchFunc(Kernel_t::circle);
                        break;
                    case Kernel_t::cube:
                        benchFunc(Kernel_t::cube);
                        break;
                    case Kernel_t::square:
                        benchFunc(Kernel_t::square);
                        break;
                }
            }
        }
        
        /// @brief Main benchmarking function
        void searchBench() {
            // Some aliases
            const auto& benchmarkRadii = mainOptions.benchmarkRadii;
            const auto& tolerances = mainOptions.approximateTolerances;
            const auto& algos = mainOptions.searchAlgos;
            const auto& kernels = mainOptions.kernels;
            const size_t repeats = mainOptions.repeats;

            if (checkResults) {
                resultSet.resultsNeighOld.resize(searchSet.numSearches);
                resultSet.resultsNeigh.resize(searchSet.numSearches);
                resultSet.resultsNeighStruct.resize(searchSet.numSearches);
                resultSet.resultsSearchApproxLower.resize(searchSet.numSearches);
                resultSet.resultsSearchApproxUpper.resize(searchSet.numSearches);
            }
        
            printBenchmarkLog("neighSearch benchmark", benchmarkRadii, repeats);
        
            const size_t total = benchmarkRadii.size();
            for (size_t i = 0; i < benchmarkRadii.size(); ++i) {
                // Linear octree
                if constexpr (std::is_same_v<Octree_t<Point_t>, LinearOctree<Point_t>>) {
                    if (algos.contains(SearchAlgo::NEIGHBORS)) {
                        for (const auto& kernel : kernels) {
                            switch (kernel) {
                                case Kernel_t::sphere:
                                    benchmarkSearchNeighOld<Kernel_t::sphere>(repeats, benchmarkRadii[i]);
                                    break;
                                case Kernel_t::circle:
                                    benchmarkSearchNeighOld<Kernel_t::circle>(repeats, benchmarkRadii[i]);
                                    break;
                                case Kernel_t::cube:
                                    benchmarkSearchNeighOld<Kernel_t::cube>(repeats, benchmarkRadii[i]);
                                    break;
                                case Kernel_t::square:
                                    benchmarkSearchNeighOld<Kernel_t::square>(repeats, benchmarkRadii[i]);
                                    break;
                            }
                        }
                    }
                    if (algos.contains(SearchAlgo::NEIGHBORS_V2)) {
                        for (const auto& kernel : kernels) {
                            switch (kernel) {
                                case Kernel_t::sphere:
                                    benchmarkSearchNeigh<Kernel_t::sphere>(repeats, benchmarkRadii[i]);
                                    break;
                                case Kernel_t::circle:
                                    benchmarkSearchNeigh<Kernel_t::circle>(repeats, benchmarkRadii[i]);
                                    break;
                                case Kernel_t::cube:
                                    benchmarkSearchNeigh<Kernel_t::cube>(repeats, benchmarkRadii[i]);
                                    break;
                                case Kernel_t::square:
                                    benchmarkSearchNeigh<Kernel_t::square>(repeats, benchmarkRadii[i]);
                                    break;
                            }
                        }
                    }
                    if (algos.contains(SearchAlgo::NEIGHBORS_STRUCT)) {
                        for (const auto& kernel : kernels) {
                            switch (kernel) {
                                case Kernel_t::sphere:
                                    benchmarkSearchNeighStruct<Kernel_t::sphere>(repeats, benchmarkRadii[i]);
                                    break;
                                case Kernel_t::circle:
                                    benchmarkSearchNeighStruct<Kernel_t::circle>(repeats, benchmarkRadii[i]);
                                    break;
                                case Kernel_t::cube:
                                    benchmarkSearchNeighStruct<Kernel_t::cube>(repeats, benchmarkRadii[i]);
                                    break;
                                case Kernel_t::square:
                                    benchmarkSearchNeighStruct<Kernel_t::square>(repeats, benchmarkRadii[i]);
                                    break;
                            }
                        }
                    }
                    if(algos.contains(SearchAlgo::NEIGHBORS_APPROX)) {
                        for(size_t j = 0; j<tolerances.size(); j++) {
                            for (const auto& kernel : kernels) {
                                if (kernel == Kernel_t::sphere) {
                                    benchmarkSearchNeighApprox<Kernel_t::sphere>(repeats, benchmarkRadii[i], tolerances[j]);
                                } else if (kernel == Kernel_t::circle) {
                                    benchmarkSearchNeighApprox<Kernel_t::circle>(repeats, benchmarkRadii[i], tolerances[j]);
                                } else if (kernel == Kernel_t::cube) {
                                    benchmarkSearchNeighApprox<Kernel_t::cube>(repeats, benchmarkRadii[i], tolerances[j]);
                                } else if (kernel == Kernel_t::square) {
                                    benchmarkSearchNeighApprox<Kernel_t::square>(repeats, benchmarkRadii[i], tolerances[j]);
                                }
                            }
                        }
                    }
                }
                // Pointer-based octree
                else if constexpr (std::is_same_v<Octree_t<Point_t>, Octree<Point_t>>) {
                    if (algos.contains(SearchAlgo::NEIGHBORS_PTR)) {
                        for (const auto& kernel : kernels) {
                            switch (kernel) {
                                case Kernel_t::sphere:
                                    benchmarkSearchNeigh<Kernel_t::sphere>(repeats, benchmarkRadii[i]);
                                    break;
                                case Kernel_t::circle:
                                    benchmarkSearchNeigh<Kernel_t::circle>(repeats, benchmarkRadii[i]);
                                    break;
                                case Kernel_t::cube:
                                    benchmarkSearchNeigh<Kernel_t::cube>(repeats, benchmarkRadii[i]);
                                    break;
                                case Kernel_t::square:
                                    benchmarkSearchNeigh<Kernel_t::square>(repeats, benchmarkRadii[i]);
                                    break;
                            }
                        }
                    }
                }
                printBenchmarkUpdate("Neighbor search", total, i+1, benchmarkRadii[i]);
            }
            std::cout << std::endl;
        }
        
        void parallelScalabilityBenchmark() {
            // Some aliases
            const auto& benchmarkRadii = mainOptions.benchmarkRadii;
            const auto& tolerances = mainOptions.approximateTolerances;
            const auto& numThreads = mainOptions.numThreads;
            const auto& kernels = mainOptions.kernels;
            const size_t repeats = mainOptions.repeats;

            // neighbors vs neighborsV2
            printBenchmarkLog("parallelism benchmark", benchmarkRadii, repeats);
            size_t total = numThreads.size() * benchmarkRadii.size();
            omp_set_schedule(omp_sched_dynamic, 0); // We always use OpenMP default chunk size by passing 0
            for (size_t j = 0; j < numThreads.size(); j++) {                    
                for (size_t i = 0; i < benchmarkRadii.size(); i++) {
                    for (const auto& kernel : kernels) {
                        if (kernel == Kernel_t::sphere) {
                            benchmarkSearchNeigh<Kernel_t::sphere>(repeats, benchmarkRadii[i], numThreads[j]);
                        } else if (kernel == Kernel_t::circle) {
                            benchmarkSearchNeigh<Kernel_t::circle>(repeats, benchmarkRadii[i], numThreads[j]);
                        } else if (kernel == Kernel_t::cube) {
                            benchmarkSearchNeigh<Kernel_t::cube>(repeats, benchmarkRadii[i], numThreads[j]);
                        } else if (kernel == Kernel_t::square) {
                            benchmarkSearchNeigh<Kernel_t::square>(repeats, benchmarkRadii[i], numThreads[j]);
                        }
                    }
                    printBenchmarkUpdate("Neighbor search - dynamic schedule, threads: ", 
                                            total, i+1, benchmarkRadii[i]);
                }
            }
        }

        void deleteOctree() {
            oct.reset();
        }

        const SearchSet<Point_t>& getSearchSet() const { return searchSet; }
        ResultSet<Point_t> getResultSet() const { return resultSet; }
};