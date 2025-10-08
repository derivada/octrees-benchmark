#pragma once
#include <omp.h>
#include <type_traits>
#include "benchmarking.hpp"
#include "NeighborKernels/KernelFactory.hpp"
#include "octree.hpp"
#include "linear_octree.hpp"
#include "TimeWatcher.hpp"
#include "search_set.hpp"
#include "main_options.hpp"
#include "unibnOctree.hpp"
#ifdef HAVE_PCL
#include <pcl/point_cloud.h>
#include <pcl/octree/octree_search.h>
#include <pcl/kdtree/kdtree_flann.h>
#include "pcl_wrappers.hpp"
#endif
#include "nanoflann.hpp"
#include "nanoflann_wrappers.hpp"
#include "papi_events.hpp"
#include "point_containers.hpp"


template <PointContainer Container>
class NeighborsBenchmark {
    private:
        using PointEncoder = PointEncoding::PointEncoder;
        using key_t = PointEncoding::key_t;
        using coords_t = PointEncoding::coords_t;
        PointEncoder& enc;
        std::vector<key_t>& codes;
        Box box;
        const std::string comment;
        Container& points;
        size_t currentBenchmarkExecution = 0;
        std::ofstream &outputFile;
        SearchSet &searchSet;
        
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
                << containerTypeToString(mainOptions.containerType) << ','
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
                << containerTypeToString(mainOptions.containerType) << ','
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
                << eventValues[2]
                << std::endl;
        }


    public:
        NeighborsBenchmark(Container& points, std::vector<key_t>& codes, Box box, PointEncoder& enc, SearchSet& searchSet, 
            std::ofstream &file) :
            points(points), 
            codes(codes),
            box(box),
            enc(enc),
            searchSet(searchSet),
            outputFile(file) {}

        void printBenchmarkInfo() {
            std::cout << std::fixed << std::setprecision(3);
            std::cout << std::left << "Starting neighbor search benchmark!\n";
            std::cout << std::left << std::setw(LOG_FIELD_WIDTH) << "Encoder:"                    << std::setw(LOG_FIELD_WIDTH)   << enc.getEncoderName()                               << "\n";
            std::cout << std::left << std::setw(LOG_FIELD_WIDTH) << "Number of searches per run:" << std::setw(LOG_FIELD_WIDTH)   << searchSet.numSearches                              << "\n";
            std::cout << std::left << std::setw(LOG_FIELD_WIDTH) << "Repeats:"                    << std::setw(LOG_FIELD_WIDTH)   << mainOptions.repeats                                << "\n";
            std::cout << std::left << std::setw(LOG_FIELD_WIDTH) << "Warmup:"                     << std::setw(LOG_FIELD_WIDTH)   << (mainOptions.useWarmup ? "enabled" : "disabled")               << "\n";
            std::cout << std::left << std::setw(LOG_FIELD_WIDTH) << "Search set distribution:"    << std::setw(LOG_FIELD_WIDTH)   << (searchSet.sequential ? "sequential" : "random")   << "\n";
            std::cout << std::endl;
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
                        }, mainOptions.useWarmup, eventSet, eventValues.data());
                        printPapiResults(events, descriptions, eventValues);
                        appendToCsv(algo, kernelName, radius, stats, eventValues, averageResultSize, numberOfThreads);
                    } else {
                        auto [stats, averageResultSize] = benchmarking::benchmark<size_t>(repeats, [&]() { 
                            return searchCallback(radius); 
                        }, mainOptions.useWarmup);
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
                        }, mainOptions.useWarmup, eventSet, eventValues.data());
                        printPapiResults(events, descriptions, eventValues);
                        appendToCsv(algo, "KNN", k, stats, eventValues, averageResultSize, numberOfThreads);
                    } else {
                        auto [stats, averageResultSize] = benchmarking::benchmark<size_t>(repeats, [&]() { 
                            return knnSearchCallback(k); 
                        }, mainOptions.useWarmup);
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

        void benchmarkNanoflannKDTree(NanoFlannKDTree<Container> &kdtree, std::string_view kernelName) {
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

        void benchmarkNanoflannKDTreeKNN(NanoFlannKDTree<Container> &kdtree) {
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
        void benchmarkUnibnOctree(unibn::Octree<Point, Container> &oct, std::string_view kernelName) {
            if(mainOptions.searchAlgos.contains(SearchAlgo::NEIGHBORS_UNIBN)) {
                auto neighborsUnibn = [&](double radius) -> size_t {
                    size_t averageResultSize = 0;
                    std::vector<size_t> &searchIndexes = searchSet.searchPoints[searchSet.currentRepeat];

                    #pragma omp parallel for schedule(runtime) reduction(+:averageResultSize)
                        for (size_t i = 0; i < searchSet.numSearches; i++) {
                            std::vector<uint32_t> results;
                            if constexpr (Kernel == Kernel_t::sphere) {
                                oct.template radiusNeighbors<unibn::L2Distance<Point>>(points[searchIndexes[i]], radius, results);
                            } else if constexpr (Kernel == Kernel_t::cube) {
                                oct.template radiusNeighbors<unibn::MaxDistance<Point>>(points[searchIndexes[i]], radius, results);
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
        void benchmarkLinearOctree(LinearOctree<Container>& oct, const std::string_view& kernelName) {
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

        void benchmarkLinearOctreeKNN(LinearOctree<Container>& oct) {
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
        void benchmarkPtrOctree(Octree<Container> &oct, std::string_view kernelName) {
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
            NanoflannPointCloud<Container> npc(points);

            // Build nanoflann kd-tree and run searches
            NanoFlannKDTree<Container> kdtree(3, npc, {mainOptions.maxPointsLeaf});
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
            unibn::Octree<Point, Container> oct;
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
            LinearOctree oct(points, codes, box, enc);
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
            Octree oct(points, box);
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
                        if(enc.getShortEncoderName() == encoderTypeToString(EncoderType::NO_ENCODING)) {
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
};
