#pragma once
#include <omp.h>
#include <type_traits>
#include "benchmarking.hpp"
#include "NeighborKernels/KernelFactory.hpp"
#include "octree.hpp"
#include "linear_octree.hpp"
#include "TimeWatcher.hpp"
#include "result_checking.hpp"
#include "search_set.hpp"
#include "main_options.hpp"
#include "unibnOctree.hpp"
#ifdef HAVE_PCL
#include <pcl/point_cloud.h>
#include <pcl/octree/octree_search.h>
#include <pcl/kdtree/kdtree_flann.h>
#endif
#include "nanoflann.hpp"
#include "nanoflann_wrappers.hpp"
#include "papi.h"

using namespace ResultChecking;

template <typename Point_t>
class NeighborsBenchmark {
    private:
        using PointEncoder = PointEncoding::PointEncoder;
        using key_t = PointEncoding::key_t;
        using coords_t = PointEncoding::coords_t;
        PointEncoder& enc;
        std::vector<key_t>& codes;
        Box box;
        const std::string comment;
        std::vector<Point_t>& points;
        size_t currentBenchmarkExecution = 0;
        std::ofstream &outputFile;
        const bool checkResults, useWarmup;
        SearchSet &searchSet;
        ResultSet<Point_t> resultSet;

        /**
         * main_parameter might be radius (fixed-radius searches) or k (knn searches)
         * in any case, we write it on radius column in the csv, for simplicity
         * kernel is "kNN" or one of the 4 kernels for radius searches
         */
        template <typename ParameterType>
        inline void appendToCsv(SearchAlgo algo, std::string_view kernel, ParameterType main_parameter, const benchmarking::Stats<>& stats, 
                                size_t averageResultSize = 0, int numThreads = omp_get_max_threads(), double tolerancePercentage = 0.0) {
            // Check if the file is empty and append header if it is
            if (outputFile.tellp() == 0) {
                outputFile <<   "date,octree,point_type,encoder,npoints,operation,kernel,radius,num_searches,sequential_searches,repeats,"
                                "accumulated,mean,median,stdev,used_warmup,warmup_time,avg_result_size,tolerance_percentage,"
                                "openmp_threads,openmp_schedule\n";
            }

            // if the comment, exists, append it to the op. name
            std::string fullAlgoName = std::string(searchAlgoToString(algo)) + ((comment != "") ? "_" + comment : "");
            
            // Get OpenMP runtime information
            omp_sched_t openmpSchedule;
            int openmpChunkSize;
            omp_get_schedule(&openmpSchedule, &openmpChunkSize);
            std::string openmpScheduleName;
            switch (openmpSchedule) {
                case omp_sched_static: openmpScheduleName = "static"; break;
                case omp_sched_dynamic: openmpScheduleName = "dynamic"; break;
                case omp_sched_guided: openmpScheduleName = "guided"; break;
                default: openmpScheduleName; break;
            }
            std::string sequentialSearches;
            if(searchSet.sequential) {
                sequentialSearches = "sequential";
            } else {
                sequentialSearches = "random";
            }
            outputFile << getCurrentDate() << ',' 
                << searchStructureToString(algoToStructure(algo)) << ',' 
                << "Point" << ','
                << enc.getEncoderName() << ','
                << points.size() <<  ','
                << fullAlgoName << ',' 
                << kernel << ',' 
                << main_parameter << ','
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
        template <typename ParameterType>
        inline void appendToCsv(SearchAlgo algo, std::string_view kernel, ParameterType main_parameter, const benchmarking::Stats<>& stats, 
                                std::vector<long long> &eventValues, size_t averageResultSize = 0, 
                                int numThreads = omp_get_max_threads(), double tolerancePercentage = 0.0
                                ) {
            // Check if the file is empty and append header if it is
            if (outputFile.tellp() == 0) {
                outputFile <<   "date,octree,point_type,encoder,npoints,operation,kernel,radius,num_searches,sequential_searches,repeats,"
                                "accumulated,mean,median,stdev,used_warmup,warmup_time,avg_result_size,tolerance_percentage,"
                                "openmp_threads,openmp_schedule,l1d_miss,l2d_miss,l3_miss\n";
            }

            // if the comment, exists, append it to the op. name
            std::string fullAlgoName = std::string(searchAlgoToString(algo)) + ((comment != "") ? "_" + comment : "");
            
            // Get OpenMP runtime information
            omp_sched_t openmpSchedule;
            int openmpChunkSize;
            omp_get_schedule(&openmpSchedule, &openmpChunkSize);
            std::string openmpScheduleName;
            switch (openmpSchedule) {
                case omp_sched_static: openmpScheduleName = "static"; break;
                case omp_sched_dynamic: openmpScheduleName = "dynamic"; break;
                case omp_sched_guided: openmpScheduleName = "guided"; break;
                default: openmpScheduleName; break;
            }
            std::string sequentialSearches;
            if(searchSet.sequential) {
                sequentialSearches = "sequential";
            } else {
                sequentialSearches = "random";
            }
            outputFile << getCurrentDate() << ',' 
                << searchStructureToString(algoToStructure(algo)) << ',' 
                << "Point" << ','
                << enc.getEncoderName() << ','
                << points.size() <<  ','
                << fullAlgoName << ',' 
                << kernel << ',' 
                << main_parameter << ','
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
                << openmpScheduleName << ','
                << eventValues[0] << ','
                << eventValues[1] << ','
                << eventValues[2] << ','
                << std::endl;
        }


    public:
        NeighborsBenchmark(std::vector<Point_t>& points, std::vector<key_t>& codes, Box box, PointEncoder& enc, SearchSet& searchSet, 
            std::ofstream &file, bool checkResults = mainOptions.checkResults, bool useWarmup = mainOptions.useWarmup) :
            points(points), 
            codes(codes),
            box(box),
            enc(enc),
            searchSet(searchSet),
            outputFile(file),
            checkResults(checkResults),
            useWarmup(useWarmup),
            resultSet(searchSet) {}

        void printBenchmarkInfo() {
            std::cout << std::fixed << std::setprecision(3);
            std::cout << std::left << "Starting neighbor search benchmark!\n";
            std::cout << std::left << std::setw(LOG_FIELD_WIDTH) << "Encoder:"                    << std::setw(LOG_FIELD_WIDTH)   << enc.getEncoderName()                               << "\n";
            std::cout << std::left << std::setw(LOG_FIELD_WIDTH) << "Number of searches per run:" << std::setw(LOG_FIELD_WIDTH)   << searchSet.numSearches                              << "\n";
            std::cout << std::left << std::setw(LOG_FIELD_WIDTH) << "Repeats:"                    << std::setw(LOG_FIELD_WIDTH)   << mainOptions.repeats                                << "\n";
            std::cout << std::left << std::setw(LOG_FIELD_WIDTH) << "Warmup:"                     << std::setw(LOG_FIELD_WIDTH)   << (useWarmup ? "enabled" : "disabled")               << "\n";
            std::cout << std::left << std::setw(LOG_FIELD_WIDTH) << "Search set distribution:"    << std::setw(LOG_FIELD_WIDTH)   << (searchSet.sequential ? "sequential" : "random")   << "\n";
            std::cout << std::endl;
        }


        static constexpr int CACHE_EVENTS[] = {
            PAPI_L1_DCM, // l1d cache misses
            PAPI_L2_DCM, // l2d cache misses
            PAPI_L3_TCM, // l3 cache misses
            //PAPI_L3_DCA // l3 accesses
        };
        
        // Native event names (compile-time known strings)
        static constexpr std::pair<const char*, const char*> NATIVE_EVENTS[] = {
            //{"perf::L1-DCACHE-LOADS", "L1D cache load accesses"},
            //{"perf::L1-DCACHE-STORES", "L1D store accesses"}
        };

        static constexpr int NUM_NATIVE_EVENTS = sizeof(NATIVE_EVENTS) / sizeof(NATIVE_EVENTS[0]);

        // At runtime, convert native names to codes and combine
        std::pair<std::vector<int>, std::vector<std::string>> buildCombinedEventList() {
            std::vector<int> fullEventList;
            std::vector<std::string> fullEventListDescs;

            PAPI_event_info_t info;

            // Add standard PAPI events first
            for (auto ev : CACHE_EVENTS) {
                fullEventList.push_back(ev);

                if (PAPI_get_event_info(ev, &info) == PAPI_OK) {
                    fullEventListDescs.emplace_back(info.short_descr);
                } else {
                    fullEventListDescs.emplace_back("Unknown PAPI event");
                }
            }

            // Convert native event names to PAPI codes and add
            for (int i = 0; i < NUM_NATIVE_EVENTS; ++i) {
                int code;
                if (PAPI_event_name_to_code(const_cast<char*>(NATIVE_EVENTS[i].first), &code) == PAPI_OK) {
                    fullEventList.push_back(code);
                    fullEventListDescs.push_back(NATIVE_EVENTS[i].second);
                } else {
                    std::cerr << "Failed to convert native event name: " << NATIVE_EVENTS[i].first << "\n";
                }
            }

            return {fullEventList, fullEventListDescs};
        }

        void printPapiResults(const std::vector<int> &events, const std::vector<std::string> &descs, const std::vector<long long>& values) {
            // Save original formatting
            std::ios_base::fmtflags originalFlags = std::cout.flags();
            std::streamsize originalPrecision = std::cout.precision();
            
            // Switch to scientific notation
            std::cout << std::scientific << std::setprecision(3);
            std::cout << "PAPI Performance Counters:\n";
            for (size_t i = 0; i < events.size(); ++i) {
                std::cout << "  " << descs[i] << ": "
                        << static_cast<double>(values[i]) << "\n";
            }

            // Restore original formatting
            std::cout.flags(originalFlags);
            std::cout.precision(originalPrecision);
        }

        int initPapiEventSet(std::vector<int> &events) {
            int eventSet = PAPI_NULL;

            if (PAPI_create_eventset(&eventSet) != PAPI_OK) {
                std::cerr << "Failed to create PAPI event set." << std::endl;
                return PAPI_NULL;
            }

            for (int i = 0; i < events.size(); ++i) {
                if (PAPI_add_event(eventSet, events[i]) != PAPI_OK) {
                    std::cerr << "Failed to add event " << events[i] << " at index " << i << std::endl;
                    PAPI_cleanup_eventset(eventSet);
                    PAPI_destroy_eventset(&eventSet);
                    return PAPI_NULL;
                }
            }
            return eventSet;
        }


        void executeBenchmark(const std::function<size_t(double)>& searchCallback, std::string_view kernelName, SearchAlgo algo) {
            std::cout << "  Running " << searchAlgoToString(algo) << " on kernel " << kernelName << std::endl;
            const auto& radii = mainOptions.benchmarkRadii;
            const size_t repeats = mainOptions.repeats;
            const auto& numThreads = mainOptions.numThreads;         
            for (size_t th = 0; th < numThreads.size(); th++) {    
                size_t numberOfThreads = numThreads[th];                
                omp_set_num_threads(numberOfThreads);
                for (size_t r = 0; r < radii.size(); r++) {
                    double radius = radii[r];
                    if(mainOptions.cacheProfiling) {
                        auto [events, descriptions] = buildCombinedEventList();
                        int eventSet = initPapiEventSet(events);
                        std::vector<long long> eventValues(events.size());
                        if (eventSet == PAPI_NULL) {
                            std::cout << "Failed to initialize PAPI event set." << std::endl;
                            exit(1);
                        }
                        auto [stats, averageResultSize] = benchmarking::benchmark<size_t>(repeats, [&]() { 
                            return searchCallback(radius); 
                        }, useWarmup, eventSet, eventValues.data());
                        printPapiResults(events, descriptions, eventValues);
                        appendToCsv(algo, kernelName, radius, stats, eventValues, averageResultSize, numberOfThreads);
                    } else {
                        auto [stats, averageResultSize] = benchmarking::benchmark<size_t>(repeats, [&]() { 
                            return searchCallback(radius); 
                        }, useWarmup);
                        appendToCsv(algo, kernelName, radius, stats, averageResultSize, numberOfThreads);
                    }
                    searchSet.reset();
                    std::cout << std::setprecision(2);
                    std::cout << "    (" << r + th*numThreads.size() + 1 << "/" << numThreads.size() * radii.size() << ") " 
                        << "Radius  " << std::setw(8) << radius 
                        << "Threads " << std::setw(8) << numberOfThreads
                        << std::endl;
                }
            }
        }

        void executeKNNBenchmark(const std::function<size_t(size_t)>& knnSearchCallback, SearchAlgo algo) {
            std::cout << "  Running k-NN searches with " << searchAlgoToString(algo) << std::endl;
            const auto& kValues = mainOptions.benchmarkKValues;
            const size_t repeats = mainOptions.repeats;
            const auto& numThreads = mainOptions.numThreads;         
            for (size_t th = 0; th < numThreads.size(); th++) {    
                size_t numberOfThreads = numThreads[th];                
                omp_set_num_threads(numberOfThreads);
                for (size_t i = 0; i < kValues.size(); i++) {
                    size_t k = kValues[i];
                    if(mainOptions.cacheProfiling) {
                        auto [events, descriptions] = buildCombinedEventList();
                        int eventSet = initPapiEventSet(events);
                        std::vector<long long> eventValues(events.size());
                        if (eventSet == PAPI_NULL) {
                            std::cout << "Failed to initialize PAPI event set." << std::endl;
                            exit(1);
                        }
                        auto [stats, averageResultSize] = benchmarking::benchmark<size_t>(repeats, [&]() { 
                            return knnSearchCallback(k); 
                        }, useWarmup, eventSet, eventValues.data());
                        printPapiResults(events, descriptions, eventValues);
                        appendToCsv(algo, "KNN", k, stats, eventValues, averageResultSize, numberOfThreads);
                    } else {
                        auto [stats, averageResultSize] = benchmarking::benchmark<size_t>(repeats, [&]() { 
                            return knnSearchCallback(k); 
                        }, useWarmup);
                        appendToCsv(algo, "KNN", k, stats, averageResultSize, numberOfThreads);
                    }
                    searchSet.reset();
                    std::cout << std::setprecision(2);
                    std::cout << "    (" << i + th*numThreads.size() + 1 << "/" << numThreads.size() * kValues.size() << ") " 
                        << "k  " << std::setw(8) << k 
                        << "Threads " << std::setw(8) << numberOfThreads
                        << std::endl;
                }
            }
        }

        void benchmarkNanoflannKDTree(NanoFlannKDTree<Point_t> &kdtree, std::string_view kernelName) {
            if(mainOptions.searchAlgos.contains(SearchAlgo::NEIGHBORS_NANOFLANN)) {
                auto neighborsNanoflannKDTree = [&](double radius) -> size_t {
                    size_t averageResultSize = 0;
                    std::vector<size_t> &searchIndexes = searchSet.searchPoints[searchSet.currentRepeat];
                    #pragma omp parallel for schedule(runtime) reduction(+:averageResultSize)
                        for (size_t i = 0; i < searchSet.numSearches; i++) {
                            std::vector<nanoflann::ResultItem<size_t, double>> ret_matches;
                            const double pt[3] = {points[searchIndexes[i]].getX(), points[searchIndexes[i]].getY(), points[searchIndexes[i]].getZ()};
                            // nanoflann expects squared radius
                            const size_t nMatches = kdtree.template radiusSearch(pt, radius*radius, ret_matches);
                            averageResultSize += nMatches;
                        }
                    averageResultSize /= searchSet.numSearches;
                    searchSet.nextRepeat();
                    return averageResultSize;
                };
                executeBenchmark(neighborsNanoflannKDTree, kernelName, SearchAlgo::NEIGHBORS_NANOFLANN);
            }
        }

        void benchmarkNanoflannKDTreeKNN(NanoFlannKDTree<Point_t> &kdtree) {
            if(mainOptions.searchAlgos.contains(SearchAlgo::KNN_NANOFLANN)) {
                auto neighborsKNNNanoflann = [&](size_t k) -> size_t {
                    size_t averageResultSize = 0;
                    std::vector<size_t> &searchIndexes = searchSet.searchPoints[searchSet.currentRepeat];
                    #pragma omp parallel for schedule(runtime) reduction(+:averageResultSize)
                        for (size_t i = 0; i < searchSet.numSearches; i++) {
                            std::vector<size_t> indexes(k);
                            std::vector<double> distances(k);
                            const double pt[3] = {points[searchIndexes[i]].getX(), points[searchIndexes[i]].getY(), points[searchIndexes[i]].getZ()};
                            // nanoflann expects squared radius
                            const size_t nMatches = kdtree.template knnSearch(pt, k, &indexes[0], &distances[0]);
                            averageResultSize += nMatches; // only here so the call is not optimized
                        }
                    averageResultSize /= searchSet.numSearches;
                    searchSet.nextRepeat();
                    return averageResultSize;
                };
                executeKNNBenchmark(neighborsKNNNanoflann, SearchAlgo::KNN_NANOFLANN);
            }
        }

        template <Kernel_t Kernel>
        void benchmarkUnibnOctree(unibn::Octree<Point_t> &oct, std::string_view kernelName) {
            if(mainOptions.searchAlgos.contains(SearchAlgo::NEIGHBORS_UNIBN)) {
                auto neighborsUnibn = [&](double radius) -> size_t {
                    size_t averageResultSize = 0;
                    std::vector<size_t> &searchIndexes = searchSet.searchPoints[searchSet.currentRepeat];

                    #pragma omp parallel for schedule(runtime) reduction(+:averageResultSize)
                        for (size_t i = 0; i < searchSet.numSearches; i++) {
                            std::vector<uint32_t> results;
                            if constexpr (Kernel == Kernel_t::sphere) {
                                oct.template radiusNeighbors<unibn::L2Distance<Point_t>>(points[searchIndexes[i]], radius, results);
                            } else if constexpr (Kernel == Kernel_t::cube) {
                                oct.template radiusNeighbors<unibn::MaxDistance<Point_t>>(points[searchIndexes[i]], radius, results);
                            } else {
                                static_assert(Kernel == Kernel_t::sphere || Kernel == Kernel_t::cube,
                                            "Unsupported kernel for unibn octree");
                            }
                            averageResultSize += results.size();
                        }

                    averageResultSize /= searchSet.numSearches;
                    searchSet.nextRepeat();
                    return averageResultSize;
                };
                executeBenchmark(neighborsUnibn, kernelName, SearchAlgo::NEIGHBORS_UNIBN);
            }
        }
        
#ifdef HAVE_PCL
        static pcl::PointCloud<pcl::PointXYZ> convertCloudToPCL(std::vector<Point_t> &points) {
            pcl::PointCloud<pcl::PointXYZ> pclCloud;
            pclCloud.width = points.size();
            pclCloud.height = 1;
            pclCloud.points.resize(points.size());
            #pragma omp parallel for schedule(runtime)
                for (size_t i = 0; i < points.size(); ++i) {
                    pclCloud.points[i].x = points[i].getX();
                    pclCloud.points[i].y = points[i].getY();
                    pclCloud.points[i].z = points[i].getZ();
                }
            return pclCloud;
        }

        void benchmarkPCLOctreeKNN(pcl::octree::OctreePointCloudSearch<pcl::PointXYZ> &octree, pcl::PointCloud<pcl::PointXYZ> &pclCloud) {
            if(mainOptions.searchAlgos.contains(SearchAlgo::KNN_PCLOCT)) {
                 auto KNN_PCLOCT = [&](size_t k) -> size_t {
                    size_t averageResultSize = 0;
                    std::vector<size_t> &searchIndexes = searchSet.searchPoints[searchSet.currentRepeat];
                    #pragma omp parallel for schedule(runtime) reduction(+:averageResultSize)
                        for(size_t i = 0; i<searchSet.numSearches; i++) {
                            pcl::PointXYZ searchPoint = pclCloud[searchIndexes[i]];
                            std::vector<int> indexes(k);
                            std::vector<float> distances(k);
                            averageResultSize += octree.nearestKSearch(searchPoint, k, indexes, distances);
                        }
                    averageResultSize = averageResultSize / searchSet.numSearches;
                    searchSet.nextRepeat();
                    return averageResultSize;
                };
                
                executeKNNBenchmark(KNN_PCLOCT, SearchAlgo::KNN_PCLOCT);
            }
        }


        void benchmarkPCLOctree(pcl::octree::OctreePointCloudSearch<pcl::PointXYZ> &octree, pcl::PointCloud<pcl::PointXYZ> &pclCloud, std::string_view kernelName) {
            if(mainOptions.searchAlgos.contains(SearchAlgo::NEIGHBORS_PCLOCT)) {
                auto neighborsPCLOct = [&](double radius) -> size_t {
                    size_t averageResultSize = 0;
                    std::vector<size_t> &searchIndexes = searchSet.searchPoints[searchSet.currentRepeat];
                    #pragma omp parallel for schedule(runtime) reduction(+:averageResultSize)
                        for(size_t i = 0; i<searchSet.numSearches; i++) {
                            pcl::PointXYZ searchPoint = pclCloud[searchIndexes[i]];
                            std::vector<int> indexes;
                            std::vector<float> distances;
                            averageResultSize += octree.radiusSearch(searchPoint, radius, indexes, distances);
                        }
                    averageResultSize = averageResultSize / searchSet.numSearches;
                    searchSet.nextRepeat();
                    return averageResultSize;
                };
                
                executeBenchmark(neighborsPCLOct, kernelName, SearchAlgo::NEIGHBORS_PCLOCT);
            }
        }

        void benchmarkPCLKDTreeKNN(pcl::KdTreeFLANN<pcl::PointXYZ> &kdtree, pcl::PointCloud<pcl::PointXYZ> &pclCloud) {
            if(mainOptions.searchAlgos.contains(SearchAlgo::KNN_PCLKD)) {
                 auto KNN_PCLKD = [&](size_t k) -> size_t {
                    size_t averageResultSize = 0;
                    std::vector<size_t> &searchIndexes = searchSet.searchPoints[searchSet.currentRepeat];
                    #pragma omp parallel for schedule(runtime) reduction(+:averageResultSize)
                        for(size_t i = 0; i<searchSet.numSearches; i++) {
                            pcl::PointXYZ searchPoint = pclCloud[searchIndexes[i]];
                            std::vector<int> indexes(k);
                            std::vector<float> distances(k);
                            averageResultSize += kdtree.nearestKSearch(searchPoint, k, indexes, distances);
                        }
                    averageResultSize = averageResultSize / searchSet.numSearches;
                    searchSet.nextRepeat();
                    return averageResultSize;
                };
                
                executeKNNBenchmark(KNN_PCLKD, SearchAlgo::NEIGHBORS_PCLKD);
            }
        }

        void benchmarkPCLKDTree(pcl::KdTreeFLANN<pcl::PointXYZ> &kdtree, pcl::PointCloud<pcl::PointXYZ> &pclCloud, std::string_view kernelName) {
            if(mainOptions.searchAlgos.contains(SearchAlgo::NEIGHBORS_PCLKD)) {
                auto neighborsPCLKD = [&](double radius) -> size_t {
                    size_t averageResultSize = 0;
                    std::vector<size_t> &searchIndexes = searchSet.searchPoints[searchSet.currentRepeat];
                    #pragma omp parallel for schedule(runtime) reduction(+:averageResultSize)
                        for(size_t i = 0; i<searchSet.numSearches; i++) {
                            pcl::PointXYZ searchPoint = pclCloud[searchIndexes[i]];
                            std::vector<int> indexes;
                            std::vector<float> distances;
                            averageResultSize += kdtree.radiusSearch(searchPoint, radius, indexes, distances);
                        }
                    averageResultSize = averageResultSize / searchSet.numSearches;
                    searchSet.nextRepeat();
                    return averageResultSize;
                };
                
                executeBenchmark(neighborsPCLKD, kernelName, SearchAlgo::NEIGHBORS_PCLKD);
            }
        }

#endif

        template <Kernel_t kernel>
        void benchmarkLinearOctree(LinearOctree<Point_t>& oct, const std::string_view& kernelName) {
            if(mainOptions.searchAlgos.contains(SearchAlgo::NEIGHBORS)) {
                auto neighborsSearch = [&](double radius) -> size_t {
                    size_t averageResultSize = 0;
                    std::vector<size_t> &searchIndexes = searchSet.searchPoints[searchSet.currentRepeat];
                    #pragma omp parallel for schedule(runtime) reduction(+:averageResultSize)
                        for(size_t i = 0; i<searchSet.numSearches; i++) {
                            auto result = oct.template searchNeighborsOld<kernel>(points[searchIndexes[i]], radius);
                            averageResultSize += result.size();
                        }
                    averageResultSize /= searchSet.numSearches;
                    searchSet.nextRepeat();
                    return averageResultSize;
                };
                executeBenchmark(neighborsSearch, kernelName, SearchAlgo::NEIGHBORS);
            }
            if(mainOptions.searchAlgos.contains(SearchAlgo::NEIGHBORS_PRUNE)) {
                auto neighborsSearchPrune = [&](double radius) -> size_t {
                    size_t averageResultSize = 0;
                    std::vector<size_t> &searchIndexes = searchSet.searchPoints[searchSet.currentRepeat];
                    #pragma omp parallel for schedule(runtime) reduction(+:averageResultSize)
                        for(size_t i = 0; i<searchSet.numSearches; i++) {
                            auto result = oct.template searchNeighbors<kernel>(points[searchIndexes[i]], radius);
                            averageResultSize += result.size();
                        }
                    averageResultSize /= searchSet.numSearches;
                    searchSet.nextRepeat();
                    return averageResultSize;
                };
                executeBenchmark(neighborsSearchPrune, kernelName, SearchAlgo::NEIGHBORS_PRUNE);
            }
            if(mainOptions.searchAlgos.contains(SearchAlgo::NEIGHBORS_STRUCT)) {
                auto neighborsSearchStruct = [&](double radius) -> size_t {
                    size_t averageResultSize = 0;
                    std::vector<size_t> &searchIndexes = searchSet.searchPoints[searchSet.currentRepeat];
                    #pragma omp parallel for schedule(runtime) reduction(+:averageResultSize)
                        for(size_t i = 0; i<searchSet.numSearches; i++) {
                            auto result = oct.template searchNeighborsStruct<kernel>(points[searchIndexes[i]], radius);
                            averageResultSize += result.size();
                        }
                    averageResultSize /= searchSet.numSearches;
                    searchSet.nextRepeat();
                    return averageResultSize;
                };
                executeBenchmark(neighborsSearchStruct, kernelName, SearchAlgo::NEIGHBORS_STRUCT);
            }
        }

        void benchmarkLinearOctreeKNN(LinearOctree<Point_t>& oct) {
            if(mainOptions.searchAlgos.contains(SearchAlgo::KNN_V2)) {
                auto neighborsKNNV2 = [&](size_t k) -> size_t {
                    size_t averageResultSize = 0;
                    std::vector<size_t> &searchIndexes = searchSet.searchPoints[searchSet.currentRepeat];
                    #pragma omp parallel for schedule(runtime) reduction(+:averageResultSize)
                        for(size_t i = 0; i<searchSet.numSearches; i++) {
                            std::vector<size_t> indexes(k);
                            std::vector<double> distances(k);
                            const size_t nMatches = oct.template knnV2 (points[searchIndexes[i]], k, indexes, distances);
                            averageResultSize += nMatches; // only here so the call is not optimized
                        }
                    averageResultSize /= searchSet.numSearches;
                    searchSet.nextRepeat();
                    return averageResultSize;
                };
                executeKNNBenchmark(neighborsKNNV2, SearchAlgo::KNN_V2);
            }
        }

        template <Kernel_t kernel>
        void benchmarkPtrOctree(Octree<Point_t> &oct, std::string_view kernelName) {
            if(mainOptions.searchAlgos.contains(SearchAlgo::NEIGHBORS_PTR)) {
                auto neighborsPtrSearch = [&](double radius) -> size_t {
                    size_t averageResultSize = 0;
                    std::vector<size_t> &searchIndexes = searchSet.searchPoints[searchSet.currentRepeat];
                    #pragma omp parallel for schedule(runtime) reduction(+:averageResultSize)
                        for(size_t i = 0; i<searchSet.numSearches; i++) {
                            auto result = oct.template searchNeighbors<kernel>(points[searchIndexes[i]], radius);
                            averageResultSize += result.size();
                        }
                    averageResultSize /= searchSet.numSearches;
                    searchSet.nextRepeat();
                    return averageResultSize;
                };
                executeBenchmark(neighborsPtrSearch, kernelName, SearchAlgo::NEIGHBORS_PTR);
            }
        }


        void initializeBenchmarkNanoflannKDTree() {
            NanoflannPointCloud<Point_t> npc(points);

            // Build nanoflann kd-tree and run searches
            NanoFlannKDTree<Point_t> kdtree(3, npc, {mainOptions.maxPointsLeaf});
            for (const auto& kernel : mainOptions.kernels) {
                switch (kernel) {
                    case Kernel_t::sphere:
                        benchmarkNanoflannKDTree(kdtree, kernelToString(kernel));
                        break;
                }
            }
            benchmarkNanoflannKDTreeKNN(kdtree);
        }

        void initializeBenchmarkUnibnOctree() {
            unibn::Octree<Point_t> oct;
            unibn::OctreeParams params;
            params.bucketSize = mainOptions.maxPointsLeaf;
            oct.initialize(points, params);
            for (const auto& kernel : mainOptions.kernels) {
                switch (kernel) {
                    case Kernel_t::sphere:
                        benchmarkUnibnOctree<Kernel_t::sphere>(oct, kernelToString(kernel));
                        break;
                    case Kernel_t::cube:
                        benchmarkUnibnOctree<Kernel_t::cube>(oct, kernelToString(kernel));
                        break;
                }
            }
        }

#ifdef HAVE_PCL
        void initializeBenchmarkPCLOctree() {
            // Convert cloud to PCL cloud
            auto pclCloud = convertCloudToPCL(points);
            
            // Build the PCL Octree
            pcl::octree::OctreePointCloudSearch<pcl::PointXYZ> oct(mainOptions.pclOctResolution);
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloudPtr = pclCloud.makeShared();
            oct.setInputCloud(cloudPtr);
            oct.addPointsFromInputCloud();;
            
            for (const auto& kernel : mainOptions.kernels) {
                switch (kernel) {
                    case Kernel_t::sphere:
                        benchmarkPCLOctree(oct, pclCloud, kernelToString(kernel));
                        break;
                }
            }
            benchmarkPCLOctreeKNN(oct, pclCloud);
        }
        
        void initializeBenchmarkPCLKDTree() {
            // Convert cloud to PCL cloud
            auto pclCloud = convertCloudToPCL(points);
            
            // Build the PCL Kd-tree
            pcl::KdTreeFLANN<pcl::PointXYZ> kdtree = pcl::KdTreeFLANN<pcl::PointXYZ>();
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloudPtr = pclCloud.makeShared();
            kdtree.setInputCloud(cloudPtr);
            
            for (const auto& kernel : mainOptions.kernels) {
                switch (kernel) {
                    case Kernel_t::sphere:
                        benchmarkPCLKDTree(kdtree, pclCloud, kernelToString(kernel));
                        break;
                }
            }
            benchmarkPCLKDTreeKNN(kdtree, pclCloud);
        }
#endif

        void initializeBenchmarkLinearOctree() {
            LinearOctree<Point_t> oct(points, codes, box, enc);
            for (const auto& kernel : mainOptions.kernels) {
                switch (kernel) {
                    case Kernel_t::sphere:
                        benchmarkLinearOctree<Kernel_t::sphere>(oct, kernelToString(kernel));
                        break;
                    case Kernel_t::circle:
                        benchmarkLinearOctree<Kernel_t::circle>(oct, kernelToString(kernel));
                        break;
                    case Kernel_t::cube:
                        benchmarkLinearOctree<Kernel_t::cube>(oct, kernelToString(kernel));
                        break;
                    case Kernel_t::square:
                        benchmarkLinearOctree<Kernel_t::square>(oct, kernelToString(kernel));
                        break;
                }
            }
            benchmarkLinearOctreeKNN(oct);
        }

        void initializeBenchmarkPtrOctree() {
            Octree<Point_t> oct(points, box);
            for (const auto& kernel : mainOptions.kernels) {
                switch (kernel) {
                    case Kernel_t::sphere:
                        benchmarkPtrOctree<Kernel_t::sphere>(oct, kernelToString(kernel));
                        break;
                    case Kernel_t::circle:
                        benchmarkPtrOctree<Kernel_t::circle>(oct, kernelToString(kernel));
                        break;
                    case Kernel_t::cube:
                        benchmarkPtrOctree<Kernel_t::cube>(oct, kernelToString(kernel));
                        break;
                    case Kernel_t::square:
                        benchmarkPtrOctree<Kernel_t::square>(oct, kernelToString(kernel));
                        break;
                }
            }
        }


        /// @brief Main benchmarking function
        void runAllBenchmarks() {
            printBenchmarkInfo();
            int currentStructureBenchmark = 1;
            int totalStructureBenchmarks = mainOptions.searchStructures.size();
            for(SearchStructure structure: mainOptions.searchStructures) {
                std::cout << "Starting benchmarks over structure " << searchStructureToString(structure) 
                    << " (" << currentStructureBenchmark << " out of " << totalStructureBenchmarks << " structures)" << std::endl; 
                switch(structure) {
                    case SearchStructure::PTR_OCTREE:
                        initializeBenchmarkPtrOctree();
                    break;
                    case SearchStructure::LINEAR_OCTREE:
                        if(enc.getShortEncoderName() == "none") {
                            std::cout << "Skipping Linear Octree since point cloud was not reordered!" << std::endl;
                        } else {
                            initializeBenchmarkLinearOctree();
                        }
                    break;
#ifdef HAVE_PCL
                    case SearchStructure::PCL_KDTREE:
                        initializeBenchmarkPCLKDTree();
                    break;
                    case SearchStructure::PCL_OCTREE:
                        initializeBenchmarkPCLOctree();
                    break;
#endif
                    case SearchStructure::UNIBN_OCTREE:
                        initializeBenchmarkUnibnOctree();
                    break;
                    case SearchStructure::NANOFLANN_KDTREE:
                        initializeBenchmarkNanoflannKDTree();
                    break;
                }
                currentStructureBenchmark++;
            }
        }

        SearchSet& getSearchSet() const { return searchSet; }
        ResultSet<Point_t> getResultSet() const { return resultSet; }
};
