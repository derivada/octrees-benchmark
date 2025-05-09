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

        size_t getTotalAmountOfRuns() {
            size_t availableAlgos = 0;
            size_t execPerAlgo = mainOptions.numThreads.size() * mainOptions.benchmarkRadii.size();

            if constexpr (std::is_same_v<Octree_t<Point_t>, LinearOctree<Point_t>>) {
                if(mainOptions.searchAlgos.contains(SearchAlgo::NEIGHBORS_PTR)) availableAlgos = 1;
            } else if constexpr (std::is_same_v<Octree_t<Point_t>, Octree<Point_t>>) {
                availableAlgos = mainOptions.searchAlgos.size();
                if(mainOptions.searchAlgos.contains(SearchAlgo::NEIGHBORS_PTR)) availableAlgos--;
            }
            // Calculate total number of benchmark functionc alls
            size_t total = execPerAlgo * availableAlgos;
            if(mainOptions.searchAlgos.contains(SearchAlgo::NEIGHBORS_APPROX)) {
                // approx searches do an innermost loop, add remaining tol.size - 1 executions
                total += (mainOptions.approximateTolerances.size() - 1) * execPerAlgo;
            }
            return total;
        }

        bool isParallelismBenchmark() {
            if(mainOptions.numThreads.size() == 0) return false;
            return mainOptions.numThreads.size() > 1 || mainOptions.numThreads[0] != omp_get_max_threads();
        }


        void printBenchmarkLog() {
            const auto& benchmarkRadii = mainOptions.benchmarkRadii;
            const auto& repeats = mainOptions.repeats;

            // Displaying the basic information with formatting
            std::cout << std::fixed << std::setprecision(3);
            std::string octreeName;
            if constexpr (std::is_same_v<Octree_t<Point_t>, LinearOctree<Point_t>>) {
                octreeName = "LinearOctree";
            } else if constexpr (std::is_same_v<Octree_t<Point_t>, Octree<Point_t>>) {
                octreeName = "Octree";
            }
            std::cout << std::left << "Starting neighbor search benchmark!\n";
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
            std::cout << std::left << std::setw(LOG_FIELD_WIDTH) << "Number of runs:"             << std::setw(LOG_FIELD_WIDTH)   << getTotalAmountOfRuns()          << "\n";
            std::cout << std::left << std::setw(LOG_FIELD_WIDTH) << "Number of searches per run:" << std::setw(LOG_FIELD_WIDTH)   << searchSet.numSearches                              << "\n";
            std::cout << std::left << std::setw(LOG_FIELD_WIDTH) << "Kernels:"                    << std::setw(2*LOG_FIELD_WIDTH) << getKernelListString()                              << "\n";
            if constexpr (std::is_same_v<Octree_t<Point_t>, LinearOctree<Point_t>>) {
                std::cout << std::left << std::setw(LOG_FIELD_WIDTH) << "Search algorithms:"      << std::setw(2*LOG_FIELD_WIDTH) << getSearchAlgoListString()                          << "\n";
            } else if constexpr (std::is_same_v<Octree_t<Point_t>, Octree<Point_t>>) {
                std::cout << std::left << std::setw(LOG_FIELD_WIDTH) << "Search algorithms:"      << std::setw(2*LOG_FIELD_WIDTH) << "neighbors"                                        << "\n";
            }            
            std::cout << std::left << std::setw(LOG_FIELD_WIDTH) << "Repeats:"                    << std::setw(LOG_FIELD_WIDTH)   << repeats                                            << "\n";
            std::cout << std::left << std::setw(LOG_FIELD_WIDTH) << "Warmup:"                     << std::setw(LOG_FIELD_WIDTH)   << (useWarmup ? "enabled" : "disabled")               << "\n";
            std::cout << std::left << std::setw(LOG_FIELD_WIDTH) << "Search set distribution:"    << std::setw(LOG_FIELD_WIDTH)   << (searchSet.sequential ? "sequential" : "random")   << "\n";
            std::cout << std::endl;
            
            std::cout << std::left << std::setw(LOG_FIELD_WIDTH/2) << "Progress"    << std::setw(LOG_FIELD_WIDTH/2) << "Completed at" 
                                   << std::setw(LOG_FIELD_WIDTH*1.5) << "Method"      << std::setw(LOG_FIELD_WIDTH/2) << "Radius";
                                   
            if(isParallelismBenchmark()) 
                std::cout << std::setw(LOG_FIELD_WIDTH/2) << "Num. threads";
            std::cout << std::endl;
        }

        void printBenchmarkUpdate(const std::string &method, size_t currentExecution, float radius, int numThreads) {
            const std::string progress_str = "(" + std::to_string(currentExecution) + "/" + std::to_string(getTotalAmountOfRuns()) + ")";
            std::cout << std::left << std::setw(LOG_FIELD_WIDTH/2) << progress_str  << std::setw(LOG_FIELD_WIDTH/2) << getCurrentDate("[%H:%M:%S]") 
                                   << std::setw(LOG_FIELD_WIDTH*1.5) << method        << std::setw(LOG_FIELD_WIDTH/2) << radius;
            
            if(isParallelismBenchmark()) 
                std::cout << std::setw(LOG_FIELD_WIDTH/2) << numThreads;
            std::cout << std::endl;
        }

        /// @brief Main benchmarking function
        void searchBench() {
            // Some aliases
            const auto& benchmarkRadii = mainOptions.benchmarkRadii;
            const auto& tolerances = mainOptions.approximateTolerances;
            const auto& numThreads = mainOptions.numThreads;
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

            printBenchmarkLog();
            size_t current = 0;
            for (size_t th = 0; th < numThreads.size(); th++) {                    
                for (size_t r = 0; r < benchmarkRadii.size(); ++r) {
                    // Linear octree
                    if constexpr (std::is_same_v<Octree_t<Point_t>, LinearOctree<Point_t>>) {
                        if (algos.contains(SearchAlgo::NEIGHBORS)) {
                            for (const auto& kernel : kernels) {
                                switch (kernel) {
                                    case Kernel_t::sphere:
                                        benchmarkSearchNeighOld<Kernel_t::sphere>(repeats, benchmarkRadii[r], numThreads[th]);
                                        break;
                                    case Kernel_t::circle:
                                        benchmarkSearchNeighOld<Kernel_t::circle>(repeats, benchmarkRadii[r], numThreads[th]);
                                        break;
                                    case Kernel_t::cube:
                                        benchmarkSearchNeighOld<Kernel_t::cube>(repeats, benchmarkRadii[r], numThreads[th]);
                                        break;
                                    case Kernel_t::square:
                                        benchmarkSearchNeighOld<Kernel_t::square>(repeats, benchmarkRadii[r], numThreads[th]);
                                        break;
                                }
                            }
                            printBenchmarkUpdate("neighbors", ++current, benchmarkRadii[r], numThreads[th]);
                        }
                        if (algos.contains(SearchAlgo::NEIGHBORS_V2)) {
                            for (const auto& kernel : kernels) {
                                switch (kernel) {
                                    case Kernel_t::sphere:
                                        benchmarkSearchNeigh<Kernel_t::sphere>(repeats, benchmarkRadii[r], numThreads[th]);
                                        break;
                                    case Kernel_t::circle:
                                        benchmarkSearchNeigh<Kernel_t::circle>(repeats, benchmarkRadii[r], numThreads[th]);
                                        break;
                                    case Kernel_t::cube:
                                        benchmarkSearchNeigh<Kernel_t::cube>(repeats, benchmarkRadii[r], numThreads[th]);
                                        break;
                                    case Kernel_t::square:
                                        benchmarkSearchNeigh<Kernel_t::square>(repeats, benchmarkRadii[r], numThreads[th]);
                                        break;
                                }
                            }
                            printBenchmarkUpdate("neighborsV2", ++current, benchmarkRadii[r], numThreads[th]);
                        }
                        if (algos.contains(SearchAlgo::NEIGHBORS_STRUCT)) {
                            for (const auto& kernel : kernels) {
                                switch (kernel) {
                                    case Kernel_t::sphere:
                                        benchmarkSearchNeighStruct<Kernel_t::sphere>(repeats, benchmarkRadii[r], numThreads[th]);
                                        break;
                                    case Kernel_t::circle:
                                        benchmarkSearchNeighStruct<Kernel_t::circle>(repeats, benchmarkRadii[r], numThreads[th]);
                                        break;
                                    case Kernel_t::cube:
                                        benchmarkSearchNeighStruct<Kernel_t::cube>(repeats, benchmarkRadii[r], numThreads[th]);
                                        break;
                                    case Kernel_t::square:
                                        benchmarkSearchNeighStruct<Kernel_t::square>(repeats, benchmarkRadii[r], numThreads[th]);
                                        break;
                                }
                            }
                            printBenchmarkUpdate("neighborsStruct", ++current, benchmarkRadii[r], numThreads[th]);
                        }
                        if(algos.contains(SearchAlgo::NEIGHBORS_APPROX)) {
                            for(size_t tol = 0; tol<tolerances.size(); tol++) {
                                for (const auto& kernel : kernels) {
                                    if (kernel == Kernel_t::sphere) {
                                        benchmarkSearchNeighApprox<Kernel_t::sphere>(repeats, benchmarkRadii[r], tolerances[tol], numThreads[th]);
                                    } else if (kernel == Kernel_t::circle) {
                                        benchmarkSearchNeighApprox<Kernel_t::circle>(repeats, benchmarkRadii[r], tolerances[tol], numThreads[th]);
                                    } else if (kernel == Kernel_t::cube) {
                                        benchmarkSearchNeighApprox<Kernel_t::cube>(repeats, benchmarkRadii[r], tolerances[tol], numThreads[th]);
                                    } else if (kernel == Kernel_t::square) {
                                        benchmarkSearchNeighApprox<Kernel_t::square>(repeats, benchmarkRadii[r], tolerances[tol], numThreads[th]);
                                    }
                                }
                                std::string updateStr = "neighborsApprox, tol = " + std::to_string(tolerances[tol]) + "%";
                                printBenchmarkUpdate(updateStr, ++current, benchmarkRadii[r], numThreads[th]);
                            }
                        }
                    }
                    // Pointer-based octree
                    else if constexpr (std::is_same_v<Octree_t<Point_t>, Octree<Point_t>>) {
                        if (algos.contains(SearchAlgo::NEIGHBORS_PTR)) {
                            for (const auto& kernel : kernels) {
                                switch (kernel) {
                                    case Kernel_t::sphere:
                                        benchmarkSearchNeigh<Kernel_t::sphere>(repeats, benchmarkRadii[r], numThreads[th]);
                                        break;
                                    case Kernel_t::circle:
                                        benchmarkSearchNeigh<Kernel_t::circle>(repeats, benchmarkRadii[r], numThreads[th]);
                                        break;
                                    case Kernel_t::cube:
                                        benchmarkSearchNeigh<Kernel_t::cube>(repeats, benchmarkRadii[r], numThreads[th]);
                                        break;
                                    case Kernel_t::square:
                                        benchmarkSearchNeigh<Kernel_t::square>(repeats, benchmarkRadii[r], numThreads[th]);
                                        break;
                                }
                            }
                        }
                        printBenchmarkUpdate("neighborsPtr", ++current, benchmarkRadii[r], numThreads[th]);
                    }
                }
            }
            std::cout << std::endl;
        }

        void deleteOctree() {
            oct.reset();
        }

        const SearchSet<Point_t>& getSearchSet() const { return searchSet; }
        ResultSet<Point_t> getResultSet() const { return resultSet; }
};