#include <filesystem>
#include <iomanip>
#include <iostream>
#include <random>
#include <optional>
#include "main_options.hpp"
#include "util.hpp"
#include "TimeWatcher.hpp"
#include "handlers.hpp"
#include "benchmarking/benchmarking.hpp"
#include "benchmarking/neighbor_benchmarks.hpp"
#include "benchmarking/search_set.hpp"
#include "octree.hpp"
#include "linear_octree.hpp"
#include "NeighborKernels/KernelFactory.hpp"
#include "Geometry/point.hpp"
#include "Geometry/Lpoint.hpp"
#include "Geometry/Lpoint.hpp"
#include "Geometry/PointMetadata.hpp"
#include "PointEncoding/point_encoder_factory.hpp"
#include "result_checking.hpp"
#include "omp.h"
#include "encoding_octree_log.hpp"
#include "unibnOctree.hpp"
#include "nanoflann.hpp"
#include "nanoflann_wrappers.hpp"
#ifdef HAVE_PCL
#include <pcl/point_cloud.h>
#include <pcl/octree/octree_search.h>
#include <pcl/kdtree/kdtree_flann.h>
#endif

namespace fs = std::filesystem;
using namespace PointEncoding;

// Set the point type to be used here (Lpoint or Point). If Point is set, we will create separate LiDAR metadata vector)
using Point_t = Point;

/**
 * @brief Benchmark neighSearch and numNeighSearch for a given octree configuration (point type + encoder).
 * Compares LinearOctree and PointerOctree. If passed PointEncoding::NoEncoder, only PointerOctree is used.
 */
void searchBenchmark(std::ofstream &outputFile, EncoderType encoding = EncoderType::NO_ENCODING) {
    // Load points and put their metadata into a separate vector
    auto pointMetaPair = readPoints<Point_t>(mainOptions.inputFile);
    std::vector<Point_t> points = std::move(pointMetaPair.first);
    std::optional<std::vector<PointMetadata>> metadata = std::move(pointMetaPair.second);

    auto& enc = getEncoder(encoding);
    // Sort the point cloud
    auto [codes, box] = enc.sortPoints<Point_t>(points, metadata);
    // Prepare the search set (must be done after sorting since it indexes points)
    SearchSet searchSet = SearchSet(mainOptions.numSearches, points.size());
    // Run the benchmarks
    NeighborsBenchmark<Point_t> octreeBenchmarks(points, codes, box, enc, searchSet, outputFile);   
    octreeBenchmarks.runAllBenchmarks();    
}

void approximateSearchLog(std::ofstream &outputFile, EncoderType encoding) {
    assert(encoding != EncoderType::NO_ENCODING);
    auto pointMetaPair = readPoints<Point_t>(mainOptions.inputFile);
    std::vector<Point_t> points = std::move(pointMetaPair.first);
    std::optional<std::vector<PointMetadata>> metadata = std::move(pointMetaPair.second);
    auto& enc = getEncoder(encoding);
    auto [codes, box] = enc.sortPoints<Point_t>(points, metadata);

    auto lin_oct = LinearOctree<Point_t>(points, codes, box, enc);
    std::array<float, 5> tolerances = {5.0, 10.0, 25.0, 50.0, 100.0};
    float radius = 3.0;
    outputFile << "tolerance,upper,x,y,z\n";
    auto points_exact = lin_oct.searchNeighborsStruct<Kernel_t::sphere>(points[1234], radius);
    for(const Point &p: points_exact) {
        outputFile << "0.0,exact," << p.getX() << "," << p.getY() << "," << p.getZ() << "\n";
    }
    for(float tol: tolerances) {
            auto points_upper = lin_oct.searchNeighborsApprox<Kernel_t::sphere>(points[1234], 3.0, tol, true);
            auto points_lower = lin_oct.searchNeighborsApprox<Kernel_t::sphere>(points[1234], 3.0, tol, false);
            for(const Point &p: points_upper) {
                outputFile << tol << ",upper," << p.getX() << "," << p.getY() << "," << p.getZ() << "\n";
            }
            for(const Point &p: points_lower) {
                outputFile << tol << ",lower," << p.getX() << "," << p.getY() << "," << p.getZ() << "\n";
            }
    }
}

double computeLocality(std::vector<Point_t> &points, size_t windowSize) {
    TimeWatcher tw;
    tw.start();
    if (points.size() < windowSize) return 0.0;
    double locality = 0;
    int seen = 0;
    std::multiset<double> mx;
    std::multiset<double> my;
    std::multiset<double> mz;
    for(int i = 0; i<points.size(); i++) {
        mx.insert(points[i].getX());
        my.insert(points[i].getY());
        mz.insert(points[i].getZ());
        if(i >= windowSize) {
            double volume = (*mx.rbegin() - *mx.begin());
            volume *= (*my.rbegin() - *my.begin());
            volume *= (*mz.rbegin() - *mz.begin());
            
            if(seen == 0) {
                locality = volume;
            } else {
                locality = ((locality / seen) + volume) / (seen+1);
                seen++;
            }

            // erase old elements
            mx.erase(mx.find(points[i-windowSize].getX()));
            my.erase(my.find(points[i-windowSize].getY()));
            mz.erase(mz.find(points[i-windowSize].getZ()));
        }
    }
    tw.stop();
    std::cout << "avg. bounding box size (lower is better) = " << locality << std::endl;
    std::cout << "time to compute: " << tw.getElapsedDecimalSeconds() << std::endl;
    return locality;
}

void outputReorderings(std::ofstream &outputFilePoints, std::ofstream &outputFileOct, EncoderType encoding = EncoderType::NO_ENCODING) {
    auto pointMetaPair = readPoints<Point_t>(mainOptions.inputFile);
    std::vector<Point_t> points = std::move(pointMetaPair.first);
    std::optional<std::vector<PointMetadata>> metadata = std::move(pointMetaPair.second);

    auto& enc = getEncoder(encoding);
    auto [codes, box] = enc.sortPoints<Point_t>(points, metadata);

    computeLocality(points, 100);
    // Output reordered points
    outputFilePoints << std::fixed << std::setprecision(3); 
    for(size_t i = 0; i<points.size(); i++) {
        outputFilePoints <<  points[i].getX() << "," << points[i].getY() << "," << points[i].getZ() << "\n";
    }

    if(encoding != EncoderType::NO_ENCODING) {
        // Build linear octree and output bounds
        auto oct = LinearOctree<Point_t>(points, codes, box, enc);
        oct.logOctreeBounds(outputFileOct, 6);
    }
}


template <template <typename> class Octree_t>
void encodingAndOctreeLog(std::ofstream &outputFile, EncoderType encoding) {
    std::shared_ptr<EncodingOctreeLog> log = std::make_shared<EncodingOctreeLog>();
    log->pointType = "Point";
    auto pointMetaPair = readPoints<Point_t>(mainOptions.inputFile);
    std::vector<Point_t> points = std::move(pointMetaPair.first);
    std::optional<std::vector<PointMetadata>> metadata = std::move(pointMetaPair.second);
    auto& enc = getEncoder(encoding);
    double totalBbox = 0.0, totalEnc = 0.0, totalSort = 0.0;
    std::vector<uint64_t> codes; Box box;
    if(mainOptions.repeats == 0)
        return;
    if(mainOptions.useWarmup) {
        auto [codesWarmup, boxWarmup] = enc.sortPoints<Point_t>(points, metadata, log);
        std::cout << "encoding warmup times - bbox: " << log->boundingBoxTime 
            << " enc:  " << log->encodingTime << " sort: " << log->sortingTime << std::endl;
    }
    for(int i = 0; i<mainOptions.repeats; i++) {
        auto [codesRepeat, boxRepeat] = enc.sortPoints<Point_t>(points, metadata, log);
        totalBbox += log->boundingBoxTime, totalEnc += log->encodingTime, totalSort += log->sortingTime;
        std::cout << "encoding repeat #" << i << " times - bbox: " << log->boundingBoxTime 
            << " enc:  " << log->encodingTime << " sort: " << log->sortingTime << std::endl;
        if(i == mainOptions.repeats - 1) {
            codes = std::move(codesRepeat);
            box = std::move(boxRepeat);
        }
    }
    log->boundingBoxTime = totalBbox / mainOptions.repeats;
    log->encodingTime = totalEnc / mainOptions.repeats;
    log->sortingTime = totalSort / mainOptions.repeats;
    if constexpr (std::is_same_v<Octree_t<Point_t>, LinearOctree<Point_t>>) {
        double totalInternal = 0.0, totalLeaf = 0.0;
        if(mainOptions.useWarmup) {
            LinearOctree<Point_t> oct(points, codes, box, enc, log);
        }
        for(int i = 0; i<mainOptions.repeats; i++) {
            LinearOctree<Point_t> oct(points, codes, box, enc, log);
            totalLeaf += log->octreeLeafTime, totalInternal += log->octreeInternalTime;
        }
        log->octreeLeafTime = totalLeaf / mainOptions.repeats;
        log->octreeInternalTime = totalInternal / mainOptions.repeats;
    } else if constexpr (std::is_same_v<Octree_t<Point_t>, Octree<Point_t>>) {
        // only measure total time for pointer-based Octree, we do it here
        auto stats = benchmarking::benchmark(mainOptions.repeats, 
            [&]() { Octree<Point_t> oct(points, box); }, mainOptions.useWarmup);
        log->octreeTime = stats.mean();
        log->octreeType = "Octree";
        log->max_leaf_points = mainOptions.maxPointsLeaf;
        Octree<Point_t> oct(points, box);
        oct.logOctreeData(log);
    } else if constexpr (std::is_same_v<Octree_t<Point_t>, unibn::Octree<Point_t>>) {
        auto stats = benchmarking::benchmark(mainOptions.repeats, [&]() { 
            auto oct = unibn::Octree<Point_t>();
            unibn::OctreeParams params;
            params.bucketSize = mainOptions.maxPointsLeaf;
            oct.initialize(points, params);    
        }, mainOptions.useWarmup);   
        log->octreeTime = stats.mean();
        log->octreeType = "unibnOctree";
        log->max_leaf_points = mainOptions.maxPointsLeaf;
        auto oct = unibn::Octree<Point_t>();
        unibn::OctreeParams params;
        params.bucketSize = mainOptions.maxPointsLeaf;
        oct.initialize(points, params);  
        oct.logOctreeData(log);
    }

    std::cout << *log << std::endl;
    log->toCSV(outputFile);
}

/// @brief just a debugging method for checking correct knn impl
void testKNN(EncoderType encoding = EncoderType::NO_ENCODING) {
    // Load points and put their metadata into a separate vector
    auto pointMetaPair = readPoints<Point_t>(mainOptions.inputFile);
    std::vector<Point_t> points = std::move(pointMetaPair.first);
    std::optional<std::vector<PointMetadata>> metadata = std::move(pointMetaPair.second);
    auto& enc = getEncoder(encoding);
    // Sort the point cloud
    auto [codes, box] = enc.sortPoints<Point_t>(points, metadata);
    // Build structures
    LinearOctree<Point_t> loct(points, codes, box, enc);
    NanoflannPointCloud<Point_t> npc(points);
    NanoFlannKDTree<Point_t> kdtree(3, npc, {mainOptions.maxPointsLeaf});
    auto pclCloud = NeighborsBenchmark<Point_t>::convertCloudToPCL(points);

    // Build the PCL Octree
    pcl::octree::OctreePointCloudSearch<pcl::PointXYZ> pcloct(mainOptions.pclOctResolution);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudPtr = pclCloud.makeShared();
    pcloct.setInputCloud(cloudPtr);
    pcloct.addPointsFromInputCloud();

    // Build the PCL Kd-tree
    pcl::KdTreeFLANN<pcl::PointXYZ> pclkd = pcl::KdTreeFLANN<pcl::PointXYZ>();
    pclkd.setInputCloud(cloudPtr);
    
    TimeWatcher twLoct; 
    TimeWatcher twNano;
    TimeWatcher twPcloct;
    TimeWatcher twPclKD;
    long nanosOct = 0, nanosNano = 0, nanosPcloct = 0, nanosPclKD = 0;
    bool seq = true; 
    size_t n = 1000;
    SearchSet ss(n, points.size(), seq);
    auto &searchPoints = ss.searchPoints[0];
    std::cout << "Accumulated times (s) for " << n << " kNN searches over " 
        << std::fixed << std::setprecision(2) << points.size()
            << " points cloud" << std::endl;
    omp_set_num_threads(40);
    for(size_t k = 4; k<=1e6; k*=2) {
            for(size_t i = 0; i < searchPoints.size(); i++){
                    // Run loct
                    std::vector<size_t> indexesLoct(k);
                    std::vector<double> distancesLoct(k);
                    twLoct.start();
                    loct.knnV2(points[searchPoints[i]], k, indexesLoct, distancesLoct);
                    twLoct.stop();
                    nanosOct += twLoct.getElapsedNanos();
                    // Run nanoflann
                    std::vector<size_t> indexesNanoflann(k);
                    std::vector<double> distancesNanoflann(k);
                    const double pt[3] = {points[searchPoints[i]].getX(), points[searchPoints[i]].getY(), points[searchPoints[i]].getZ()};
                    twNano.start();
                    kdtree.knnSearch(pt, k, &indexesNanoflann[0], &distancesNanoflann[0]);
                    twNano.stop();
                    nanosNano += twNano.getElapsedNanos();

                    // Run PCLOCT
                    pcl::PointXYZ searchPoint = pclCloud[searchPoints[i]];
                    std::vector<int> indexesPcloct(k);
                    std::vector<float> distancesPcloct(k);
                    twPcloct.start();
                    pcloct.nearestKSearch(searchPoint, k, indexesPcloct, distancesPcloct);
                    twPcloct.stop();
                    nanosPcloct += twPcloct.getElapsedNanos();

                    // Run PCLKD
                    std::vector<int> indexesPclKD(k);
                    std::vector<float> distancesPclKD(k);
                    twPclKD.start();
                    pclkd.nearestKSearch(searchPoint, k, indexesPclKD, distancesPclKD);
                    twPclKD.stop();
                    nanosPclKD += twPclKD.getElapsedNanos();

                    std::unordered_set<size_t> indexSetLoct;
                    for (auto index : indexesLoct) {
                        indexSetLoct.insert(index);
                    }
                    std::unordered_set<size_t> indexSetNanoflann;
                    for (auto index : indexesNanoflann) {
                        indexSetNanoflann.insert(index);
                    }
                    std::unordered_set<size_t> indexSetPcloct;
                    for (auto index : indexesPcloct) {
                        indexSetPcloct.insert(index);
                    }
                    std::unordered_set<size_t> indexSetPclKD;
                    for (auto index : indexesPclKD) {
                        indexSetPclKD.insert(index);
                    }
                    if(indexSetLoct != indexSetNanoflann) {
                        std::cout << "KNN results are different for nanoflann!" << std::endl;

                        // Optional: print differences
                        std::cout << "In Loct but not in nanoflann:" << std::endl;
                        for (size_t id : indexSetLoct) {
                            if (indexSetNanoflann.find(id) == indexSetNanoflann.end()) {
                                std::cout << id << std::endl;
                            }
                        }

                        std::cout << "In nanoflann but not in Loct:" << std::endl;
                        for (size_t id : indexSetNanoflann) {
                            if (indexSetLoct.find(id) == indexSetLoct.end()) {
                                std::cout << id << std::endl;
                            }
                        }
                    }
                    if(indexSetLoct != indexSetPcloct) {
                        std::cout << "KNN results are different for pcloctree!" << std::endl;

                        // Optional: print differences
                        std::cout << "In Loct but not in pcloct:" << std::endl;
                        for (size_t id : indexSetLoct) {
                            if (indexSetPcloct.find(id) == indexSetPcloct.end()) {
                                std::cout << id << std::endl;
                            }
                        }

                        std::cout << "In pcloct but not in Loct:" << std::endl;
                        for (size_t id : indexSetPcloct) {
                            if (indexSetLoct.find(id) == indexSetLoct.end()) {
                                std::cout << id << std::endl;
                            }
                        }
                    }
                    if(indexSetLoct != indexSetPclKD) {
                        std::cout << "KNN results are different for pclkdtree!" << std::endl;

                        // Optional: print differences
                        std::cout << "In Loct but not in pclkdtree:" << std::endl;
                        for (size_t id : indexSetLoct) {
                            if (indexSetPclKD.find(id) == indexSetPclKD.end()) {
                                std::cout << id << std::endl;
                            }
                        }

                        std::cout << "In pclkdtree but not in Loct:" << std::endl;
                        for (size_t id : indexSetPclKD) {
                            if (indexSetLoct.find(id) == indexSetLoct.end()) {
                                std::cout << id << std::endl;
                            }
                        }
                    }
                }
        std::cout << "k = " << k << std::endl << std::fixed << std::setprecision(5);
        std::cout << "  linear octree: "    << (double) nanosOct / 1e9  << std::endl;
        std::cout << "  nanoflann kdtree: " << (double) nanosNano / 1e9 << std::endl;
        std::cout << "  pcl octree: " << (double) nanosPcloct / 1e9 << std::endl;
        std::cout << "  pcl kdtree: " << (double) nanosPclKD / 1e9 << std::endl;
    }
}

int main(int argc, char *argv[]) {
    // Set default OpenMP schedule: dynamic and auto chunk size
    omp_set_schedule(omp_sched_dynamic, 0);
    processArgs(argc, argv);
    std::cout << std::fixed << std::setprecision(3); 


    std::cout << "Tamaño octante unibn: " << sizeof(unibn::Octree<Point_t>::Octant) << std::endl;
    std::cout << "Tamaño octante pointer: " << sizeof(Octree<Point_t>) << std::endl;

    // Handle input file
    fs::path inputFile = mainOptions.inputFile;
    std::string fileName = inputFile.stem();
    if (!mainOptions.outputDirName.empty()) {
        mainOptions.outputDirName = mainOptions.outputDirName / fileName;
    }

    // Create output directory
    createDirectory(mainOptions.outputDirName);

    using namespace PointEncoding;
    if(!mainOptions.debug) {
        // Open the benchmark output file
        std::string csvFilename = mainOptions.inputFileName + "-" + getCurrentDate() + ".csv";
        std::filesystem::path csvPath = mainOptions.outputDirName / csvFilename;
        std::ofstream outputFile = std::ofstream(csvPath, std::ios::app);
        if (!outputFile.is_open()) {
            throw std::ios_base::failure(std::string("Failed to open benchmark output file: ") + csvPath.string());
        }
        // Run search benchmarks
        if(mainOptions.encodings.contains(EncoderType::NO_ENCODING))
            searchBenchmark(outputFile, EncoderType::NO_ENCODING);
        if(mainOptions.encodings.contains(EncoderType::MORTON_ENCODER_3D))
            searchBenchmark(outputFile, EncoderType::MORTON_ENCODER_3D);
        if(mainOptions.encodings.contains(EncoderType::HILBERT_ENCODER_3D))
            searchBenchmark(outputFile, EncoderType::HILBERT_ENCODER_3D);
    } else {
        testKNN(HILBERT_ENCODER_3D);
        // Output encoded point clouds to the files (for plots and such)
        // std::filesystem::path unencodedPath = mainOptions.outputDirName / "output_unencoded.csv";
        // std::filesystem::path mortonPath = mainOptions.outputDirName / "output_morton.csv";
        // std::filesystem::path hilbertPath = mainOptions.outputDirName / "output_hilbert.csv";
        // std::filesystem::path unencodedPathOct = mainOptions.outputDirName / "output_unencoded_oct.csv";
        // std::filesystem::path mortonPathOct = mainOptions.outputDirName / "output_morton_oct.csv";
        // std::filesystem::path hilbertPathOct = mainOptions.outputDirName / "output_hilbert_oct.csv";
        // std::ofstream unencodedFile(unencodedPath);
        // std::ofstream mortonFile(mortonPath);
        // std::ofstream hilbertFile(hilbertPath);
        // std::ofstream unencodedFileOct(unencodedPathOct);
        // std::ofstream mortonFileOct(mortonPathOct);
        // std::ofstream hilbertFileOct(hilbertPathOct);
        
        // if (!unencodedFile.is_open() || !mortonFile.is_open() || !hilbertFile.is_open() || 
        //     !unencodedFileOct.is_open() || !mortonFileOct.is_open() || !hilbertFileOct.is_open()) {
        //     throw std::ios_base::failure("Failed to open output files");
        // }
        
        // std::cout << "Output files created successfully." << std::endl;
        // outputReorderings(unencodedFile, unencodedFileOct);  
        // outputReorderings(mortonFile, mortonFileOct, EncoderType::MORTON_ENCODER_3D);  
        // outputReorderings(hilbertFile, hilbertFileOct, EncoderType::HILBERT_ENCODER_3D);  
        
        // Encoding and build times benchmark
        // std::filesystem::path encAndOctreeLogsPath = mainOptions.outputDirName / "enc_octree_times.csv";
        // std::ofstream encAndOctreeLogsFile(encAndOctreeLogsPath);
        // EncodingOctreeLog::writeCSVHeader(encAndOctreeLogsFile);
        // if(mainOptions.encodings.contains(EncoderType::NO_ENCODING)) {
        //     encodingAndOctreeLog<Octree>(encAndOctreeLogsFile, EncoderType::NO_ENCODING);
        //     encodingAndOctreeLog<unibn::Octree>(encAndOctreeLogsFile, EncoderType::NO_ENCODING);
        // }
        // if(mainOptions.encodings.contains(EncoderType::MORTON_ENCODER_3D)) {
        //     encodingAndOctreeLog<LinearOctree>(encAndOctreeLogsFile, EncoderType::MORTON_ENCODER_3D);
        //     encodingAndOctreeLog<Octree>(encAndOctreeLogsFile, EncoderType::MORTON_ENCODER_3D);
        //     encodingAndOctreeLog<unibn::Octree>(encAndOctreeLogsFile, EncoderType::MORTON_ENCODER_3D);
        // }

        // if(mainOptions.encodings.contains(EncoderType::HILBERT_ENCODER_3D)) {
        //     encodingAndOctreeLog<LinearOctree>(encAndOctreeLogsFile, EncoderType::HILBERT_ENCODER_3D);
        //     encodingAndOctreeLog<Octree>(encAndOctreeLogsFile, EncoderType::HILBERT_ENCODER_3D);
        //     encodingAndOctreeLog<unibn::Octree>(encAndOctreeLogsFile, EncoderType::HILBERT_ENCODER_3D);
        // }
    }

    return EXIT_SUCCESS;
}
