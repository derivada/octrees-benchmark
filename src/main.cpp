#include <filesystem>
#include <iomanip>
#include <iostream>
#include <random>
#include <optional>
#include "util.hpp"
#include "type_names.hpp"
#include "TimeWatcher.hpp"
#include "handlers.hpp"
#include "main_options.hpp"
#include "benchmarking.hpp"
#include "octree_benchmark.hpp"
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

namespace fs = std::filesystem;
using namespace PointEncoding;

// Set the point type to be used here (Lpoint or Point). If Point is set, we will create separate LiDAR metadata vector)
using Point_t = Point;

/**
 * @brief Benchmark neighSearch and numNeighSearch for a given octree configuration (point type + encoder).
 * Compares LinearOctree and PointerOctree. If passed PointEncoding::NoEncoder, only PointerOctree is used.
 */
void searchBenchmark(std::ofstream &outputFile, EncoderType encoding = EncoderType::NO_ENCODING) {
    auto pointMetaPair = readPoints<Point_t>(mainOptions.inputFile);
    std::vector<Point_t> points = std::move(pointMetaPair.first);
    std::optional<std::vector<PointMetadata>> metadata = std::move(pointMetaPair.second);
    auto& enc = getEncoder(encoding);
    // Sort the point cloud
    auto [codes, box] = enc.sortPoints<Point_t>(points, metadata);
    // Create the searchSet (WARMING: this should be done after sorting since it indexes points!)
    SearchSet<Point_t> searchSet = SearchSet<Point_t>(mainOptions.numSearches, points);

    OctreeBenchmark<Octree, Point_t> obPointer(points, codes, box, enc, searchSet, outputFile);
    obPointer.searchBench();
    obPointer.deleteOctree();
    if(encoding != EncoderType::NO_ENCODING) {
        OctreeBenchmark<LinearOctree, Point_t> obLinear(points,  codes, box, enc, searchSet, outputFile);
        obLinear.searchBench();
        obLinear.deleteOctree();
        if(mainOptions.checkResults)
            ResultChecking::checkResultsLinearVsPointer(obLinear.getResultSet(), obPointer.getResultSet());
    }
}

/**
 * @brief Runs the parallel execution benchmark.
 * 
 * Only uses LinearOctree, so don't pass PointEncoding::NoEncoder!
 */
template <template <typename> class Octree_t>
void parallelScalabilityBenchmark(std::ofstream &outputFile, EncoderType encoding = EncoderType::NO_ENCODING) {
    auto pointMetaPair = readPoints<Point_t>(mainOptions.inputFile);
    std::vector<Point_t> points = std::move(pointMetaPair.first);
    std::optional<std::vector<PointMetadata>> metadata = std::move(pointMetaPair.second);
    // Sort the point cloud
    auto& enc = getEncoder(encoding);
    auto [codes, box] = enc.sortPoints<Point_t>(points, metadata);

    // Create the searchSet (WARMING: this should be done after sorting since it indexes points!)
    SearchSet<Point_t> searchSet = SearchSet<Point_t>(mainOptions.numSearches, points);
    OctreeBenchmark<Octree_t, Point_t> ob(points, codes, box, enc, searchSet, outputFile);
    ob.parallelScalabilityBenchmark();
}

std::vector<Point_t> generateGridCloud(size_t n) {
    std::vector<Point_t> points;
    points.reserve(n * n * n);
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < n; j++)
            for (size_t k = 0; k < n; k++) {
                points.push_back(Point_t(i * n * n + j * n + k, i, j, k));
            }

    return points;
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

template <template <typename> class Octree_t>
void encodingAndOctreeLog(std::ofstream &outputFile, EncoderType encoding) {
    std::shared_ptr<EncodingOctreeLog> log = std::make_shared<EncodingOctreeLog>();
    log->pointType = getPointName<Point_t>();
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
    }
    std::cout << *log << std::endl;
    log->toCSV(outputFile);
}

void outputReorderings(std::ofstream &outputFilePoints, std::ofstream &outputFileOct, EncoderType encoding = EncoderType::NO_ENCODING) {
    auto points = generateGridCloud(4);
    std::optional<std::vector<PointMetadata>> metadata = std::nullopt;
    // auto pointMetaPair = readPoints<Point_t>(mainOptions.inputFile);
    // std::vector<Point_t> points = std::move(pointMetaPair.first);
    // std::optional<std::vector<PointMetadata>> metadata = std::move(pointMetaPair.second);

    auto& enc = getEncoder(encoding);
    auto [codes, box] = enc.sortPoints<Point_t>(points, metadata);

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

void unibnSearchTest( EncoderType encoding = EncoderType::NO_ENCODING) {
    auto pointMetaPair = readPoints<Point_t>(mainOptions.inputFile);
    std::vector<Point_t> points = std::move(pointMetaPair.first);
    std::optional<std::vector<PointMetadata>> metadata = std::move(pointMetaPair.second);
    auto& enc = getEncoder(encoding);
    // Sort the point cloud
    auto [codes, box] = enc.sortPoints<Point_t>(points, metadata);

    unibn::Octree<Point_t> uniOct;
    unibn::OctreeParams params;
    SearchSet<Point_t> searchSet = SearchSet<Point_t>(mainOptions.numSearches, points);
    params.bucketSize = mainOptions.maxPointsLeaf;
    uniOct.initialize(points, params);
    const Point_t& p = points[0];
    // sphere kernel
    std::vector<float> distances;
    size_t averageResultSize = 0;
    std::vector<size_t> &searchIndexes = searchSet.searchPoints[searchSet.currentRepeat];
    TimeWatcher tw;
    tw.start();
    omp_set_num_threads(35);
    #pragma omp parallel for schedule(runtime) reduction(+:averageResultSize)
        for(size_t i = 0; i<searchSet.numSearches; i++) {
            std::vector<uint32_t> results;
            uniOct.radiusNeighbors<unibn::L2Distance<Point_t>>(points[searchIndexes[i]], 20.0f, results);
            averageResultSize += results.size();
        }
    
    averageResultSize = averageResultSize / searchSet.numSearches;
    tw.stop();
    std::cout << "unibn time: " << tw.getElapsedDecimalSeconds() << std::endl;
    std::cout << "unibn mu " << averageResultSize << std::endl;
    averageResultSize = 0;
    LinearOctree<Point_t> loct(points, codes, box, enc);
    tw.start();
    #pragma omp parallel for schedule(runtime) reduction(+:averageResultSize)
        for(size_t i = 0; i<searchSet.numSearches; i++) {
            auto results = loct.searchNeighborsStruct<Kernel_t::sphere>(points[searchIndexes[i]], 20.0f);
            averageResultSize += results.size();
        }
    
    averageResultSize = averageResultSize / searchSet.numSearches;
    tw.stop();
    std::cout << "loct time: " << tw.getElapsedDecimalSeconds() << std::endl;
    std::cout << "loct mu " << averageResultSize << std::endl;
}

int main(int argc, char *argv[]) {
    // Set default OpenMP schedule: dynamic and auto chunk size
    omp_set_schedule(omp_sched_dynamic, 0);
    processArgs(argc, argv);
    std::cout << std::fixed << std::setprecision(3); 
    
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
        if(mainOptions.encodings.contains(EncoderType::NO_ENCODING))
            searchBenchmark(outputFile, EncoderType::NO_ENCODING);
        if(mainOptions.encodings.contains(EncoderType::MORTON_ENCODER_3D))
            searchBenchmark(outputFile, EncoderType::MORTON_ENCODER_3D);
        if(mainOptions.encodings.contains(EncoderType::HILBERT_ENCODER_3D))
            searchBenchmark(outputFile, EncoderType::HILBERT_ENCODER_3D);
    } else {
        unibnSearchTest(EncoderType::MORTON_ENCODER_3D);
        // Output encoded point clouds to the files
        // std::filesystem::path unencodedPath = mainOptions.outputDirName / "output_unencoded.csv";
        // std::filesystem::path mortonPath = mainOptions.outputDirName / "output_morton.csv";
        // std::filesystem::path hilbertPath = mainOptions.outputDirName / "output_hilbert.csv";
        // std::filesystem::path unencodedPathOct = mainOptions.outputDirName / "output_unencoded_oct.csv";
        // std::filesystem::path mortonPathOct = mainOptions.outputDirName / "output_morton_oct.csv";
        // std::filesystem::path hilbertPathOct = mainOptions.outputDirName / "output_hilbert_oct.csv";
        // std::ofstream unencodedFile(unencodedPath, std::ios::app);
        // std::ofstream mortonFile(mortonPath, std::ios::app);
        // std::ofstream hilbertFile(hilbertPath, std::ios::app);
        // std::ofstream unencodedFileOct(unencodedPathOct, std::ios::app);
        // std::ofstream mortonFileOct(mortonPathOct, std::ios::app);
        // std::ofstream hilbertFileOct(hilbertPathOct, std::ios::app);
        
        // if (!unencodedFile.is_open() || !mortonFile.is_open() || !hilbertFile.is_open() || 
        //     !unencodedFileOct.is_open() || !mortonFileOct.is_open() || !hilbertFileOct.is_open()) {
        //     throw std::ios_base::failure("Failed to open output files");
        // }
        
        // std::cout << "Output files created successfully." << std::endl;
        // outputReorderings(unencodedFile, unencodedFileOct);  
        // outputReorderings(mortonFile, mortonFileOct, EncoderType::MORTON_ENCODER_3D);  
        // outputReorderings(hilbertFile, hilbertFileOct, EncoderType::HILBERT_ENCODER_3D);  
        // std::filesystem::path encAndOctreeLogsPath = mainOptions.outputDirName / "enc_octree_times.csv";
        // std::ofstream encAndOctreeLogsFile(encAndOctreeLogsPath);
        // EncodingOctreeLog::writeCSVHeader(encAndOctreeLogsFile);
        // if(mainOptions.encodings.contains(EncoderType::NO_ENCODING)) {
        //     encodingAndOctreeLog<Octree>(encAndOctreeLogsFile, EncoderType::NO_ENCODING);
        // }
        // if(mainOptions.encodings.contains(EncoderType::MORTON_ENCODER_3D)) {
        //     encodingAndOctreeLog<LinearOctree>(encAndOctreeLogsFile, EncoderType::MORTON_ENCODER_3D);
        //     encodingAndOctreeLog<Octree>(encAndOctreeLogsFile, EncoderType::MORTON_ENCODER_3D);
        // }

        // if(mainOptions.encodings.contains(EncoderType::HILBERT_ENCODER_3D)) {
        //     encodingAndOctreeLog<LinearOctree>(encAndOctreeLogsFile, EncoderType::HILBERT_ENCODER_3D);
        //     encodingAndOctreeLog<Octree>(encAndOctreeLogsFile, EncoderType::HILBERT_ENCODER_3D);
        // }
    }

    return EXIT_SUCCESS;
}
