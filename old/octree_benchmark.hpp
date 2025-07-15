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

using namespace ResultChecking;

template <template <typename> class Octree_t, typename Point_t>
class RadiusNeighborsBenchmark {
    private:
        using PointEncoder = PointEncoding::PointEncoder;
        using key_t = PointEncoding::key_t;
        using coords_t = PointEncoding::coords_t;
        std::unique_ptr<Octree_t<Point_t>> oct;
        PointEncoder& enc;
        const std::string comment;
        std::vector<Point_t>& points;
#ifdef HAVE_PCL
        pcl::PointCloud<Point_t>::Ptr pclCloud;
#endif
        std::ofstream &outputFile;
        
        const bool checkResults, useWarmup;
        SearchSet &searchSet;
        ResultSet<Point_t> resultSet;

        // works for both KD-tree and Octree
#ifdef HAVE_PCL
        size_t searchNeighPCL(float radii) {
            size_t averageResultSize = 0;
            std::vector<size_t> &searchIndexes = searchSet.searchPoints[searchSet.currentRepeat];
            #pragma omp parallel for schedule(runtime) reduction(+:averageResultSize)
                for(size_t i = 0; i<searchSet.numSearches; i++) {
                    pcl::PointXYZ searchPoint = (*pclCloud) [searchIndexes[i]];
                    std::vector<int> indexes;
                    std::vector<float> distances;
                    averageResultSize += oct->template radiusSearch(searchPoint, radii, indexes, distances);
                }
            averageResultSize = averageResultSize / searchSet.numSearches;
            searchSet.nextRepeat();
            return averageResultSize;
        }
#endif

        template<Kernel_t kernel>
        size_t searchNeighNanoflann(float radii) {
            size_t averageResultSize = 0;
            std::vector<size_t> &searchIndexes = searchSet.searchPoints[searchSet.currentRepeat];

            #pragma omp parallel for schedule(runtime) reduction(+:averageResultSize)
            for (size_t i = 0; i < searchSet.numSearches; i++) {
                std::vector<nanoflann::ResultItem<size_t, double>> ret_matches;
                const double pt[3] = {points[searchIndexes[i]].getX(), points[searchIndexes[i]].getY(), points[searchIndexes[i]].getZ()};
                const size_t nMatches = oct->template radiusSearch(&pt, radii, ret_matches);
                averageResultSize += nMatches;
            }

            averageResultSize /= searchSet.numSearches;
            searchSet.nextRepeat();
            return averageResultSize;
        }

        template<Kernel_t kernel>
        size_t searchNeighUnibn(float radii) {
            size_t averageResultSize = 0;
            std::vector<size_t> &searchIndexes = searchSet.searchPoints[searchSet.currentRepeat];

            #pragma omp parallel for schedule(runtime) reduction(+:averageResultSize)
            for (size_t i = 0; i < searchSet.numSearches; i++) {
                std::vector<uint32_t> results;
                if constexpr (kernel == Kernel_t::sphere) {
                    oct->template radiusNeighbors<unibn::L2Distance<Point_t>>(points[searchIndexes[i]], radii, results);
                } else if constexpr (kernel == Kernel_t::cube) {
                    oct->template radiusNeighbors<unibn::MaxDistance<Point_t>>(points[searchIndexes[i]], radii, results);
                }
                averageResultSize += results.size();
            }

            averageResultSize /= searchSet.numSearches;
            searchSet.nextRepeat();
            return averageResultSize;
        }

        template<Kernel_t kernel>
        size_t searchNeigh(float radii) {
            size_t averageResultSize = 0;
            std::vector<size_t> &searchIndexes = searchSet.searchPoints[searchSet.currentRepeat];
            #pragma omp parallel for schedule(runtime) reduction(+:averageResultSize)
                for(size_t i = 0; i<searchSet.numSearches; i++) {
                    auto result = oct->template searchNeighbors<kernel>(points[searchIndexes[i]], radii);
                    averageResultSize += result.size();
                    if(checkResults)
                        resultSet.resultsNeigh[i] = std::move(result);
                }
            
            averageResultSize = averageResultSize / searchSet.numSearches;
            searchSet.nextRepeat();
            return averageResultSize;
        }

        template<Kernel_t kernel>
        size_t searchNeighStruct(float radii) {
            size_t averageResultSize = 0;
            std::vector<size_t> &searchIndexes = searchSet.searchPoints[searchSet.currentRepeat];
            #pragma omp parallel for schedule(runtime) reduction(+:averageResultSize)
                for(size_t i = 0; i<searchSet.numSearches; i++) {
                    auto result = oct->template searchNeighborsStruct<kernel>(points[searchIndexes[i]], radii);
                    averageResultSize += result.size();
                    if(checkResults)
                        resultSet.resultsNeighStruct[i] = std::move(result);
                }
            averageResultSize = averageResultSize / searchSet.numSearches;
            searchSet.nextRepeat();
            return averageResultSize;
        }

        template<Kernel_t kernel>
        size_t searchNeighApprox(float radii, double tolerancePercentage, bool upperBound) {
            resultSet.tolerancePercentageUsed = tolerancePercentage;
            size_t averageResultSize = 0;
            std::vector<size_t> &searchIndexes = searchSet.searchPoints[searchSet.currentRepeat];
            #pragma omp parallel for schedule(runtime) reduction(+:averageResultSize)
                for(size_t i = 0; i<searchSet.numSearches; i++) {
                    auto result = oct->template searchNeighborsApprox<kernel>(points[searchIndexes[i]], radii, tolerancePercentage, upperBound);
                    averageResultSize += result.size();
                    if(checkResults) {
                        if(upperBound)
                            resultSet.resultsSearchApproxUpper[i] = std::move(result);
                        else
                            resultSet.resultsSearchApproxLower[i] = std::move(result);
                    }
                }
            
            averageResultSize = averageResultSize / searchSet.numSearches;
            searchSet.nextRepeat();
            return averageResultSize;
        }

        template<Kernel_t kernel>
        size_t searchNeighOld(float radii) {
            size_t averageResultSize = 0;
            std::vector<size_t> &searchIndexes = searchSet.searchPoints[searchSet.currentRepeat];
            #pragma omp parallel for schedule(runtime) reduction(+:averageResultSize)
                for(size_t i = 0; i<searchSet.numSearches; i++) {
                    auto result = oct->template searchNeighborsOld<kernel>(points[searchIndexes[i]], radii);
                    averageResultSize += result.size();
                    if(checkResults)
                        resultSet.resultsNeighOld[i] = std::move(result);
                }
            averageResultSize = averageResultSize / searchSet.numSearches;
            searchSet.nextRepeat();
            return averageResultSize;
        }

        constexpr std::string getOctreeName() {
            if constexpr (std::is_same_v<Octree_t<Point_t>, LinearOctree<Point_t>>) {
                return "LinearOctree";
            } else if constexpr (std::is_same_v<Octree_t<Point_t>, Octree<Point_t>>) {
                return "Octree";
            } else if constexpr (std::is_same_v<Octree_t<Point_t>, unibn::Octree<Point_t>>) {
                return "unibnOctree";
            } 
#ifdef HAVE_PCL
            else if constexpr (std::is_same_v<Octree_t<Point_t>, pcl::KdTreeFLANN<Point_t>>) {
                return "PCLKdTree";
            } else if constexpr(std::is_same_v<Octree_t<Point_t>, pcl::octree::OctreePointCloudSearch<Point_t>>) {
                return "PCLOctree";
            }
#endif
            else {
                return "UnknownOctree";
            }
        }

        constexpr std::string getPointTypeName() {
            if constexpr (std::is_same_v<Point_t, Point>) {
                return "Point";
            }
        #ifdef HAVE_PCL
            else if constexpr (std::is_same_v<Point_t, pcl::PointXYZ>) {
                return "PCLPointXYZ";
            }
        #endif
            else {
                return "UnknownPoint";
            }
        }


        inline void appendToCsv(const std::string& operation, const std::string& kernel, const float radius, const benchmarking::Stats<>& stats, 
                                size_t averageResultSize = 0, int numThreads = omp_get_max_threads(), double tolerancePercentage = 0.0) {
            // Check if the file is empty and append header if it is
            if (outputFile.tellp() == 0) {
                outputFile <<   "date,octree,container,encoder,npoints,operation,kernel,radius,num_searches,sequential_searches,repeats,"
                                "accumulated,mean,median,stdev,used_warmup,warmup_time,avg_result_size,tolerance_percentage,"
                                "openmp_threads,openmp_schedule\n";
            }

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
                << getOctreeName() << ',' 
                << getPointTypeName() << ','
                << enc.getEncoderName() << ','
                << points.size() <<  ','
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
        RadiusNeighborsBenchmark(std::vector<Point_t>& points, std::vector<key_t>& codes, Box box, PointEncoder& enc, SearchSet& searchSet, 
            std::ofstream &file, bool checkResults = mainOptions.checkResults, bool useWarmup = mainOptions.useWarmup) :
            points(points), 
            enc(enc),
            searchSet(searchSet),
            outputFile(file),
            checkResults(checkResults),
            useWarmup(useWarmup),
            resultSet(searchSet){ 

            // Conditional initialization of oct based on the type of Octree_t<Point_t>
            if constexpr (std::is_same_v<Octree_t<Point_t>, LinearOctree<Point_t>>) {
                // Initialize for LinearOctree
                oct = std::make_unique<LinearOctree<Point_t>>(points, codes, box, enc);
            } else if constexpr (std::is_same_v<Octree_t<Point_t>, Octree<Point_t>>) {
                // Initialize for Octree
                oct = std::make_unique<Octree<Point_t>>(points, box);
            } else if constexpr(std::is_same_v<Octree_t<Point_t>, unibn::Octree<Point_t>>) {
                oct = std::make_unique<unibn::Octree<Point_t>>();
                unibn::OctreeParams params;
                params.bucketSize = mainOptions.maxPointsLeaf;
                oct->template initialize(points, params);
            } else if constexpr(std::is_same_v<Octree_t<Point_t>, NanoFlannKDTree<Point_t>>) {
                NanoflannPointCloud<Point_t> npc;
                npc.pts = std::move(points);
                oct = std::make_unique<NanoFlannKDTree<Point_t>>(3, npc, {mainOptions.maxPointsLeaf});
            } else {
                static_assert("Wrong constructor for this octree type");
            }
        }

#ifdef HAVE_PCL
        OctreeBenchmark(std::vector<Point_t>& points, std::vector<key_t>& codes, Box box, PointEncoder& enc, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, 
            SearchSet& searchSet, std::ofstream &file, bool checkResults = mainOptions.checkResults, bool useWarmup = mainOptions.useWarmup) :
            points(points), 
            enc(enc),
            pclCloud(cloud),
            searchSet(searchSet),
            outputFile(file),
            checkResults(checkResults),
            useWarmup(useWarmup),
            resultSet(searchSet) { 
            if constexpr(std::is_same_v<Octree_t<Point_t>, pcl::KdTreeFLANN<Point_t>>) {
                oct = std::make_unique<pcl::KdTreeFLANN<Point_t>>();
                oct->template setInputCloud(pclCloud);
            } 
            else if constexpr(std::is_same_v<Octree_t<Point_t>, pcl::octree::OctreePointCloudSearch<Point_t>>) {
                oct = std::make_unique<pcl::octree::OctreePointCloudSearch<Point_t>>(mainOptions.pclOctResolution);
                oct->template setInputCloud(pclCloud);
                oct->template addPointsFromInputCloud();
            } else {
                static_assert("Wrong constructor for this octree type");
            }
        }
#endif

#ifdef HAVE_PCL
        void benchmarkSearchNeighPCL(size_t repeats, float radius, int numThreads = omp_get_max_threads()) {
            omp_set_num_threads(numThreads);
            const auto kernelStr = "Sphere";
            auto [stats, averageResultSize] = benchmarking::benchmark<size_t>(repeats, [&]() { 
                return searchNeighPCL(radius); 
            }, useWarmup);
            searchSet.reset();
            std::string algoName;
            if constexpr (std::is_same_v<Octree_t<Point_t>, pcl::KdTreeFLANN<Point_t>>) {
                algoName = "PCLKdTree";
            } else if constexpr(std::is_same_v<Octree_t<Point_t>, pcl::octree::OctreePointCloudSearch<Point_t>>) {
                algoName = "PCLOctree";
            }
            appendToCsv(algoName, kernelStr, radius, stats, averageResultSize, numThreads);
        }
#endif

        template<Kernel_t kernel>
        void benchmarkSearchNeighNanoflann(size_t repeats, float radius, int numThreads = omp_get_max_threads()) {
            omp_set_num_threads(numThreads);
            const auto kernelStr = kernelToString(kernel);
            auto [stats, averageResultSize] = benchmarking::benchmark<size_t>(repeats, [&]() { 
                return searchNeighNanoflann<kernel>(radius); 
            }, useWarmup);
            searchSet.reset();
            appendToCsv("neighborsNanoflann", kernelStr, radius, stats, averageResultSize, numThreads);
        }


        template<Kernel_t kernel>
        void benchmarkSearchNeighUnibn(size_t repeats, float radius, int numThreads = omp_get_max_threads()) {
            omp_set_num_threads(numThreads);
            const auto kernelStr = kernelToString(kernel);
            auto [stats, averageResultSize] = benchmarking::benchmark<size_t>(repeats, [&]() { 
                return searchNeighUnibn<kernel>(radius); 
            }, useWarmup);
            searchSet.reset();
            appendToCsv("neighborsUnibn", kernelStr, radius, stats, averageResultSize, numThreads);
        }

        template<Kernel_t kernel>
        void benchmarkSearchNeigh(size_t repeats, float radius, int numThreads = omp_get_max_threads()) {
            omp_set_num_threads(numThreads);
            const auto kernelStr = kernelToString(kernel);
            auto [stats, averageResultSize] = benchmarking::benchmark<size_t>(repeats, [&]() { 
                return searchNeigh<kernel>(radius); 
            }, useWarmup);
            searchSet.reset();
            std::string algoName;
            if constexpr (std::is_same_v<Octree_t<Point_t>, Octree<Point_t>>) {
                algoName = "neighborsPtr";
            } else {
                algoName = "neighborsPrune";
            }
            appendToCsv(algoName, kernelStr, radius, stats, averageResultSize, numThreads);
        }

        template<Kernel_t kernel>
        void benchmarkSearchNeighStruct(size_t repeats, float radius, int numThreads = omp_get_max_threads()) {
            omp_set_num_threads(numThreads);
            const auto kernelStr = kernelToString(kernel);
            auto [stats, averageResultSize] = benchmarking::benchmark<size_t>(repeats, [&]() { 
                return searchNeighStruct<kernel>(radius); 
            }, useWarmup);
            searchSet.reset();
            appendToCsv("neighborsStruct", kernelStr, radius, stats, averageResultSize, numThreads);
        }

        template<Kernel_t kernel>
        void benchmarkSearchNeighApprox(size_t repeats, float radius, double tolerancePercentage, int numThreads = omp_get_max_threads()) {
            omp_set_num_threads(numThreads);
            const auto kernelStr = kernelToString(kernel);
            // first lower bound, then upper bound
            auto [statsLower, averageResultSizeLower] = benchmarking::benchmark<size_t>(repeats, [&]() { 
                return searchNeighApprox<kernel>(radius, tolerancePercentage, false); 
            }, useWarmup);
            searchSet.reset();
            appendToCsv("neighborsApproxLower", kernelStr, radius, statsLower, averageResultSizeLower, numThreads, tolerancePercentage);

            auto [statsUpper, averageResultSizeUpper] = benchmarking::benchmark<size_t>(repeats, [&]() { 
                return searchNeighApprox<kernel>(radius, tolerancePercentage, true); 
            }, useWarmup);
            searchSet.reset();
            appendToCsv("neighborsApproxUpper", kernelStr, radius, statsUpper, averageResultSizeUpper, numThreads, tolerancePercentage);
        }

        template<Kernel_t kernel>
        void benchmarkSearchNeighOld(size_t repeats, float radius, int numThreads = omp_get_max_threads()) {
            omp_set_num_threads(numThreads);
            const auto kernelStr = kernelToString(kernel);
            auto [stats, averageResultSize] = benchmarking::benchmark<size_t>(repeats, [&]() { 
                return searchNeighOld<kernel>(radius); 
            }, useWarmup);
            searchSet.reset();
            appendToCsv("neighbors", kernelStr, radius, stats, averageResultSize, numThreads);
        }
        
        size_t getTotalAmountOfRuns() {
            size_t availableAlgos = 0;
            size_t execPerAlgo = mainOptions.numThreads.size() * mainOptions.benchmarkRadii.size();
            if constexpr (std::is_same_v<Octree_t<Point_t>, Octree<Point_t>>) {
                availableAlgos = mainOptions.searchAlgos.contains(SearchAlgo::NEIGHBORS_PTR) ? 1 : 0;
            } 
            else if constexpr (std::is_same_v<Octree_t<Point_t>, LinearOctree<Point_t>>) {
                availableAlgos = static_cast<int>(mainOptions.searchAlgos.size());
                for (auto nonLoctAlgos : {SearchAlgo::NEIGHBORS_PTR, SearchAlgo::NEIGHBORS_UNIBN, 
                    SearchAlgo::NEIGHBORS_PCLKD, SearchAlgo::NEIGHBORS_PCLOCT}) {
                    if (mainOptions.searchAlgos.contains(nonLoctAlgos)) {
                        availableAlgos--;
                    }
                }
            } 
            else if constexpr (std::is_same_v<Octree_t<Point_t>, unibn::Octree<Point_t>>) {
                availableAlgos = mainOptions.searchAlgos.contains(SearchAlgo::NEIGHBORS_UNIBN) ? 1 : 0;
            } 
#ifdef HAVE_PCL
            else if constexpr (std::is_same_v<Octree_t<Point_t>, pcl::KdTreeFLANN<Point_t>>) {
                availableAlgos = mainOptions.searchAlgos.contains(SearchAlgo::NEIGHBORS_PCLKD) ? 1 : 0;
            }
            else if constexpr (std::is_same_v<Octree_t<Point_t>, pcl::octree::OctreePointCloudSearch<Point_t>>) {
                availableAlgos = mainOptions.searchAlgos.contains(SearchAlgo::NEIGHBORS_PCLOCT) ? 1 : 0;
            }
#endif
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



        void printBenchmarkInfo() {
            const auto& benchmarkRadii = mainOptions.benchmarkRadii;
            const auto& repeats = mainOptions.repeats;

            // Displaying the basic information with formatting
            std::cout << std::fixed << std::setprecision(3);
            std::cout << std::left << "Starting neighbor search benchmark!\n";
            std::cout << std::left << std::setw(LOG_FIELD_WIDTH) << "Octree used:"              << std::setw(LOG_FIELD_WIDTH) << getOctreeName()        << "\n";
            std::cout << std::left << std::setw(LOG_FIELD_WIDTH) << "Point type:"               << std::setw(LOG_FIELD_WIDTH) << getPointTypeName() << "\n";
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
        void runAllBenchmarks() {
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

            printBenchmarkInfo();
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
                        if (algos.contains(SearchAlgo::NEIGHBORS_PRUNE)) {
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
                            printBenchmarkUpdate("neighborsPrune", ++current, benchmarkRadii[r], numThreads[th]);
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
                    // unibn octree
                    } else if constexpr(std::is_same_v<Octree_t<Point_t>, unibn::Octree<Point_t>>) {
                        if(algos.contains(SearchAlgo::NEIGHBORS_UNIBN)) {
                            for (const auto & kernel: kernels) {
                                switch(kernel) {
                                     case Kernel_t::sphere:
                                        benchmarkSearchNeighUnibn<Kernel_t::sphere>(repeats, benchmarkRadii[r], numThreads[th]);
                                        break;
                                    case Kernel_t::cube:
                                        benchmarkSearchNeighUnibn<Kernel_t::cube>(repeats, benchmarkRadii[r], numThreads[th]);
                                        break;
                                }
                            }
                        }
                        printBenchmarkUpdate("neighborsUnibn", ++current, benchmarkRadii[r], numThreads[th]);
                    } 
#ifdef HAVE_PCL
                    else if constexpr(std::is_same_v<Octree_t<Point_t>, pcl::KdTreeFLANN<Point_t>> 
                        || std::is_same_v<Octree_t<Point_t>, pcl::octree::OctreePointCloudSearch<Point_t>>) {
                        for (const auto & kernel: kernels) {
                            switch(kernel) {
                                    case Kernel_t::sphere:
                                    benchmarkSearchNeighPCL(repeats, benchmarkRadii[r], numThreads[th]);
                                    break;
                            }
                        }
                        std::string algoName;
                        if constexpr (std::is_same_v<Octree_t<Point_t>, pcl::KdTreeFLANN<Point_t>>) {
                            algoName = "PCLKdTree";
                        } else if constexpr(std::is_same_v<Octree_t<Point_t>, pcl::octree::OctreePointCloudSearch<Point_t>>) {
                            algoName = "PCLOctree";
                        }
                        printBenchmarkUpdate(algoName, ++current, benchmarkRadii[r], numThreads[th]);
                    }
#endif
                    else if constexpr(std::is_same_v<Octree_t<Point_t>, NanoFlannKDTree<Point_t>>) {
                         if(algos.contains(SearchAlgo::NEIGHBORS_NANOFLANN)) {
                            for (const auto & kernel: kernels) {
                                switch(kernel) {
                                     case Kernel_t::sphere:
                                        benchmarkSearchNeighNanoflann<Kernel_t::sphere>(repeats, benchmarkRadii[r], numThreads[th]);
                                        break;
                                }
                            }
                        }
                        printBenchmarkUpdate("neighborsNanoflann", ++current, benchmarkRadii[r], numThreads[th]);
                    }
                }
            }
            std::cout << std::endl;
        }

        void deleteOctree() {
            oct.reset();
        }

        SearchSet& getSearchSet() const { return searchSet; }
        ResultSet<Point_t> getResultSet() const { return resultSet; }
};
