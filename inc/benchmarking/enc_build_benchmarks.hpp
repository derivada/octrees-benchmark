#pragma once

#include <algorithm>
#include <memory>
#include <optional>
#include <omp.h>
#include <papi.h>
#include <string>
#include <type_traits>
#include <vector>

#include "geometry/point_containers.hpp"
#include "encoding/point_encoder.hpp"
#include "encoding/point_encoder_factory.hpp"
#include "structures/linear_octree.hpp"
#include "structures/nanoflann.hpp"
#include "structures/nanoflann_wrappers.hpp"

#include "structures/octree.hpp"
#include "structures/unibn_octree.hpp"

#ifdef HAVE_PICOTREE
#include "structures/picotree_profiler.hpp"
#include "structures/picotree_wrappers.hpp"
#endif

#ifdef HAVE_PCL
#include <pcl/point_cloud.h>
#include <pcl/octree/octree_search.h>
#include <pcl/kdtree/kdtree_flann.h>
#include "structures/pcl_wrappers.hpp"
#endif

#include "benchmarking_stats.hpp"
#include "build_log.hpp"
#include "encoding_log.hpp"
#include "gb_wrappers.hpp"
#include "main_options.hpp"
#include "papi_events.hpp"

template <PointContainer Container>
class EncodingBuildBenchmarks {
    using key_t = size_t;
    private:
        template <typename F>
        double runTimedEncodingBenchmark(const std::string& benchmarkName, F&& function) {
            gb_wrappers::initializeGoogleBenchmark();

            const size_t measuredRepeats = mainOptions.repeats;
            double capturedMilliseconds = 0.0;

            benchmark::ClearRegisteredBenchmarks();

            auto* benchmarkHandle = benchmark::RegisterBenchmark(benchmarkName.c_str(),
                [function = std::forward<F>(function)](benchmark::State& state) mutable {
                    state.counters["omp_threads"] = 1.0;
                    state.counters["l1d_miss"] = 0.0;
                    state.counters["l2d_miss"] = 0.0;
                    state.counters["l3_miss"] = 0.0;
                    for (auto _ : state) {
                        (void)_;
                        function();
                    }
                    benchmark::DoNotOptimize(state.iterations());
                }
            );

            benchmarkHandle->Iterations(measuredRepeats);
            if (mainOptions.useWarmup) {
                benchmarkHandle->MinWarmUpTime(0.1);
            }
            benchmarkHandle->Unit(benchmark::kMillisecond);
            static benchmark::BenchmarkReporter* baseReporter = benchmark::CreateDefaultDisplayReporter();
            gb_wrappers::ContextOnceReporter contextOnceReporter(baseReporter, &capturedMilliseconds, nullptr, false);
            benchmark::RunSpecifiedBenchmarks(&contextOnceReporter, benchmarkName);
            benchmark::ClearRegisteredBenchmarks();
            return capturedMilliseconds / 1000.0;
        }

        template <typename F>
        double runTimedBuildBenchmark(const std::string& benchmarkName, F&& function,
            int eventSet = PAPI_NULL, long long *eventValues = nullptr, int ompThreads = 1) {
            gb_wrappers::initializeGoogleBenchmark();

            const size_t measuredRepeats = mainOptions.repeats;
            double capturedMilliseconds = 0.0;

            benchmark::ClearRegisteredBenchmarks();

            auto* benchmarkHandle = benchmark::RegisterBenchmark(benchmarkName.c_str(),
                [function = std::forward<F>(function), eventSet, eventValues, measuredRepeats, ompThreads](benchmark::State& state) mutable {
                    state.counters["omp_threads"] = static_cast<double>(ompThreads);
                    state.counters["l1d_miss"] = 0.0;
                    state.counters["l2d_miss"] = 0.0;
                    state.counters["l3_miss"] = 0.0;
                    size_t currentIteration = 0;
                    for (auto _ : state) {
                        (void)_;
                        if (eventSet != PAPI_NULL && eventValues != nullptr && currentIteration == measuredRepeats - 1) {
                            if (PAPI_start(eventSet) != PAPI_OK) {
                                std::cout << "Failed to start PAPI." << std::endl;
                                exit(1);
                            }
                        }
                        function();
                        if (eventSet != PAPI_NULL && eventValues != nullptr && currentIteration == measuredRepeats - 1) {
                            if (PAPI_stop(eventSet, eventValues) != PAPI_OK) {
                                std::cout << "Failed to stop PAPI." << std::endl;
                                exit(1);
                            }
                            state.counters["l1d_miss"] = static_cast<double>(eventValues[0]);
                            state.counters["l2d_miss"] = static_cast<double>(eventValues[1]);
                            state.counters["l3_miss"] = static_cast<double>(eventValues[2]);
                        }

                        currentIteration++;
                    }
                    benchmark::DoNotOptimize(currentIteration);
                }
            );

            benchmarkHandle->Iterations(measuredRepeats);
            if (mainOptions.useWarmup) {
                benchmarkHandle->MinWarmUpTime(0.1);
            }
            benchmarkHandle->Unit(benchmark::kMillisecond);
            static benchmark::BenchmarkReporter* baseReporter = benchmark::CreateDefaultDisplayReporter();
            gb_wrappers::ContextOnceReporter contextOnceReporter(baseReporter, &capturedMilliseconds);
            benchmark::RunSpecifiedBenchmarks(&contextOnceReporter, benchmarkName);
            benchmark::ClearRegisteredBenchmarks();
            return capturedMilliseconds / 1000.0;
        }

        Container& points;
        std::optional<std::vector<PointMetadata>> &metadata;
        std::vector<key_t> codes;
        std::ostream& outputEncoding;
        std::ostream& outputBuild;
        Box box;

    public:
        EncodingBuildBenchmarks(Container &points, std::optional<std::vector<PointMetadata>> &metadata, 
            std::ostream &outputEncoding, std::ostream& outputBuild): 
            points(points), metadata(metadata), outputEncoding(outputEncoding), outputBuild(outputBuild) {}

        void runBuildBenchmark(SearchStructure structure, EncoderType encoding) {
            int eventSet = PAPI_NULL;
            auto [events, descriptions] = buildCombinedEventList();
            std::vector<long long> eventValues(events.size());
            auto &enc = PointEncoding::getEncoder(encoding);
            const std::string encName = std::string(encoderTypeToString(encoding));
            const std::string structureName = std::string(searchStructureToString(structure));
            const std::string benchmarkName = std::string("BUILD") + "-" + encName + "-" + structureName;
            std::shared_ptr<BuildLog> log = std::make_shared<BuildLog>();
            log->encoding = encoding;
            log->cloudSize = points.size();
            log->maxLeafPoints = mainOptions.maxPointsLeaf;
            log->structure = structure;
            log->numThreads = 1; // by default not parallelized, will be updated for linear octree if run in parallel

            // Initialize PAPI
            if(mainOptions.cacheProfiling) {
                auto [events, descriptions] = buildCombinedEventList();
                eventSet = initPapiEventSet(events);
                if (eventSet == PAPI_NULL) {
                    std::cout << "Failed to initialize PAPI event set." << std::endl;
                    exit(1);
                }
            }

            // Run the build for the chosen structure
            switch(structure) {
                case SearchStructure::PTR_OCTREE: {
                    size_t currRepeat = 0;
                    log->buildTime = runTimedBuildBenchmark(benchmarkName, [&]() {
                        Octree oct(points, box);
                    }, eventSet, eventValues.data(), 1);
                    // extra build for logging (not counted towards total time)
                    Octree oct(points, box);
                    oct.logOctreeData(log);
                    break;
                }
                case SearchStructure::LINEAR_OCTREE: {
                    if(!enc.is3D()) {
                        // Skip linear octree if encoding is not 3D
                        return;
                    } else {
                        const int prevOmpThreads = omp_get_max_threads();
                        int parallelThreads = 1;
                        if (!mainOptions.numThreads.empty()) {
                            parallelThreads = *std::max_element(mainOptions.numThreads.begin(), mainOptions.numThreads.end());
                        }

                        auto runLinearVariant = [&](int threadCount, const std::string& modeSuffix) {
                            omp_set_num_threads(threadCount);
                            double totalLeaf = 0.0;
                            double totalInternal = 0.0;

                            std::fill(eventValues.begin(), eventValues.end(), 0);
                            runTimedBuildBenchmark(benchmarkName + "-" + modeSuffix, [&]() {
                                LinearOctree oct(points, codes, box, enc, log);
                                totalLeaf += log->linearOctreeLeafTime;
                                totalInternal += log->linearOctreeInternalTime;
                            }, eventSet, eventValues.data(), threadCount);

                            log->linearOctreeLeafTime = totalLeaf / mainOptions.repeats;
                            log->linearOctreeInternalTime = totalInternal / mainOptions.repeats;
                            log->buildTime = log->linearOctreeLeafTime + log->linearOctreeInternalTime;
                            if(mainOptions.cacheProfiling) {
                                log->l1dMisses = eventValues[0];
                                log->l2dMisses = eventValues[1];
                                log->l3Misses = eventValues[2];
                            }
                            log->toCSV(outputBuild);
                            log->numThreads = threadCount;
                        };

                        runLinearVariant(1, "seq");
                        if (parallelThreads > 1) {
                            runLinearVariant(parallelThreads, "par");
                        }

                        omp_set_num_threads(prevOmpThreads);
                        return;
                    }
                    break;
                }
#ifdef HAVE_PCL
                case SearchStructure::PCL_KDTREE: {
                    auto pclCloud = convertCloudToPCL(points);
                    // Build the PCL Kd-tree
                    log->buildTime = runTimedBuildBenchmark(benchmarkName, [&]() {
                        pcl::KdTreeFLANN<pcl::PointXYZ> kdtree = pcl::KdTreeFLANN<pcl::PointXYZ>();
                        pcl::PointCloud<pcl::PointXYZ>::Ptr cloudPtr = pclCloud.makeShared();
                        kdtree.setInputCloud(cloudPtr);
                    }, eventSet, eventValues.data(), 1);
                    break;
                }
                case SearchStructure::PCL_OCTREE: {
                    auto pclCloud = convertCloudToPCL(points);
                    log->buildTime = runTimedBuildBenchmark(benchmarkName, [&]() {
                            // Build the PCL Octree
                            pcl::octree::OctreePointCloudSearch<pcl::PointXYZ> oct(mainOptions.pclOctResolution);
                            pcl::PointCloud<pcl::PointXYZ>::Ptr cloudPtr = pclCloud.makeShared();
                            oct.setInputCloud(cloudPtr);
                            oct.addPointsFromInputCloud();
                    }, eventSet, eventValues.data(), 1);
                    pcl::octree::OctreePointCloudSearch<pcl::PointXYZ> oct(mainOptions.pclOctResolution);
                    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudPtr = pclCloud.makeShared();
                    oct.setInputCloud(cloudPtr);
                    oct.addPointsFromInputCloud();
                    log->memoryUsed = estimatePCLOctMemory(oct);
                    break;
                }
#endif
                case SearchStructure::UNIBN_OCTREE: {
                    unibn::OctreeParams params;
                    log->buildTime = runTimedBuildBenchmark(benchmarkName, [&]() {
                        unibn::Octree<Point, Container> oct;
                        params.bucketSize = mainOptions.maxPointsLeaf;
                        oct.initialize(points, params);
                    }, eventSet, eventValues.data(), 1);
                    unibn::Octree<Point, Container> oct; // rebuild so logOctreeData doesnt affect measured time
                    oct.initialize(points, params);
                    oct.logOctreeData(log);
                    break;
                }
                case SearchStructure::NANOFLANN_KDTREE: {
                    NanoflannPointCloud<Container> npc(points);
                    log->buildTime = runTimedBuildBenchmark(benchmarkName, [&]() {
                        NanoFlannKDTree<Container> kdtree(3, npc, {mainOptions.maxPointsLeaf});
                        log->memoryUsed = kdtree.usedMemory(kdtree); // record memory used
                    }, eventSet, eventValues.data(), 1);
                    break;
                }
#ifdef HAVE_PICOTREE
                case SearchStructure::PICOTREE: {
                    if constexpr (std::is_same_v<Container, PointsAoS>) {
                        log->buildTime = runTimedBuildBenchmark(benchmarkName, [&]() {
                            pico_tree::kd_tree<Container> tree(points, pico_tree::max_leaf_size_t(mainOptions.maxPointsLeaf));
                        }, eventSet, eventValues.data(), 1);
                        pico_tree::kd_tree<PointsAoS> tree(points, pico_tree::max_leaf_size_t(mainOptions.maxPointsLeaf));
                        log->memoryUsed = pico_tree_profiler::get_memory_usage(tree);
                    } else {
                        std::cout << "WARNING: Skipping pico_tree build benchmark: Container is not PointsAoS." << std::endl;
                        return;
                    }
                    break;
                }
#endif
            }
            if(mainOptions.cacheProfiling) {
                log->l1dMisses = eventValues[0];
                log->l2dMisses = eventValues[1];
                log->l3Misses = eventValues[2];
            }
            log->toCSV(outputBuild);
        }

        void runEncodingBenchmark(EncoderType encoding) {
            double totalBbox = 0.0, totalEnc = 0.0, totalSort = 0.0;
            std::shared_ptr<EncodingLog> log = std::make_shared<EncodingLog>();
            auto& enc = PointEncoding::getEncoder(encoding);
            const std::string encName = std::string(encoderTypeToString(encoding));
            const std::string benchmarkName = std::string("ENCOD") + "-" + encName;

            if (encoding == EncoderType::NO_ENCODING) {
                runTimedEncodingBenchmark(benchmarkName, [&]() {
                    Vector radii;
                    Point center = mbb(points, radii);
                    box = Box(center, radii);
                });

                codes.clear();
                log->cloudSize = points.size();
                log->encoding = encoding;
                log->boundingBoxTime = 0.0;
                log->encodingTime = 0.0;
                log->sortingTime = 0.0;
                log->toCSV(outputEncoding);
                return;
            }

            size_t measuredIteration = 0;
            runTimedEncodingBenchmark(benchmarkName, [&]() {
                auto pointsCopy = points;
                auto [codesRepeat, boxRepeat] = enc.sortPoints(pointsCopy, metadata, log);
                totalBbox += log->boundingBoxTime;
                totalEnc += log->encodingTime;
                totalSort += log->sortingTime;
                // move point cloud
                if(measuredIteration == mainOptions.repeats - 1) {
                    points = pointsCopy;
                    codes = codesRepeat;
                    box = boxRepeat;
                }
                measuredIteration++;
            });
            log->cloudSize = points.size();
            log->encoding = encoding;
            log->boundingBoxTime = totalBbox / mainOptions.repeats;
            log->encodingTime = totalEnc / mainOptions.repeats;
            log->sortingTime = totalSort / mainOptions.repeats;
            log->toCSV(outputEncoding);
        }

        /// @brief Main benchmarking function
        void runEncodingBuildBenchmarks() {
            BuildLog::writeCSVHeader(outputBuild);
            EncodingLog::writeCSVHeader(outputEncoding);
            int currentEncoder = 1;
            int totalStructureBenchmarks = mainOptions.searchStructures.size();

            // First, we do unencoded points, since we don't reorder here and later the points are altered. 
            // If NO_ENCODING was done after Hilbert per example, the array would still be on Hilbert order.
            if(mainOptions.encodings.contains(EncoderType::NO_ENCODING)) {
                runEncodingBenchmark(EncoderType::NO_ENCODING);
                int currentStructure = 1;
                for(SearchStructure structure: mainOptions.searchStructures) {
                        runBuildBenchmark(structure, EncoderType::NO_ENCODING);
                }
            }

            for(EncoderType encoding: mainOptions.encodings) {
                // Skip NO_ENCODING, already done
                if(encoding == EncoderType::NO_ENCODING)
                    continue;
                runEncodingBenchmark(encoding);
                int currentStructure = 1;
                for(SearchStructure structure: mainOptions.searchStructures) {
                        runBuildBenchmark(structure, encoding);
                }
            }
        }
};
