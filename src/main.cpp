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
#include "Geometry/Lpoint64.hpp"
#include "Geometry/PointMetadata.hpp"
#include "PointEncoding/point_encoder_factory.hpp"
#include "result_checking.hpp"
#include "omp.h"
#include "encoding_octree_log.hpp"

namespace fs = std::filesystem;
using namespace PointEncoding;

/**
 * @brief Benchmark neighSearch and numNeighSearch for a given octree configuration (point type + encoder).
 * Compares LinearOctree and PointerOctree. If passed PointEncoding::NoEncoder, only PointerOctree is used.
 */
template <typename Point_t>
void searchBenchmark(std::ofstream &outputFile, EncoderType encoding = EncoderType::NO_ENCODING) {
    auto pointMetaPair = readPointsWithMetadata<Point_t>(mainOptions.inputFile);
    std::vector<Point_t> points = std::move(pointMetaPair.first);
    std::optional<std::vector<PointMetadata>> metadata = std::move(pointMetaPair.second);
    auto& enc = getEncoder(encoding);
    // Sort the point cloud
    auto [codes, box] = enc.sortPoints<Point_t>(points, metadata);
    // Create the searchSet (WARMING: this should be done after sorting since it indexes points!)
    const SearchSet<Point_t> searchSet = SearchSet<Point_t>(mainOptions.numSearches, points);

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
 * @brief Benchmark oldNeighSearch vs neighSearch and oldNumNeighSearch vs numNeighSearch for a given octree configuration (point type + encoder),
 * in order to see the performance improvement of the new implementation.
 * 
 * Only uses LinearOctree, since that's where the implementation is being improved, so don't pass PointEncoding::NoEncoder!
 */
template <typename Point_t>
void algoCompBenchmark(std::ofstream &outputFile, EncoderType encoding) {
    assert(encoding != EncoderType::NO_ENCODING);
    auto pointMetaPair = readPointsWithMetadata<Point_t>(mainOptions.inputFile);
    std::vector<Point_t> points = std::move(pointMetaPair.first);
    std::optional<std::vector<PointMetadata>> metadata = std::move(pointMetaPair.second);

    // Sort the point cloud
    auto& enc = getEncoder(encoding);
    auto [codes, box] = enc.sortPoints<Point_t>(points, metadata);
    
    // Create the searchSet (WARMING: this should be done after sorting since it indexes points!)
    const SearchSet<Point_t> searchSet = SearchSet<Point_t>(mainOptions.numSearches, points);

    OctreeBenchmark<LinearOctree, Point_t> ob(points, codes, box, enc, searchSet, outputFile);
    ob.searchImplComparisonBench();

    if(mainOptions.checkResults) {
        ResultChecking::checkResultsAlgoComp(ob.getResultSet());
    }
}

/**
 * @brief Runs the approximate searches benchmark with the configuration given in mainOptions
 * 
 * Only uses LinearOctree, so don't pass PointEncoding::NoEncoder!
 */
template <typename Point_t>
void approxSearchBenchmark(std::ofstream &outputFile, EncoderType encoding) {
    assert(encoding != EncoderType::NO_ENCODING);
    auto pointMetaPair = readPointsWithMetadata<Point_t>(mainOptions.inputFile);
    std::vector<Point_t> points = std::move(pointMetaPair.first);
    std::optional<std::vector<PointMetadata>> metadata = std::move(pointMetaPair.second);
    // Sort the point cloud
    auto& enc = getEncoder(encoding);
    auto [codes, box] = enc.sortPoints<Point_t>(points, metadata);
    
    // Create the searchSet (WARMING: this should be done after sorting since it indexes points!)
    const SearchSet<Point_t> searchSet = SearchSet<Point_t>(mainOptions.numSearches, points);

    OctreeBenchmark<LinearOctree, Point_t> ob(points, codes, box, enc, searchSet, outputFile);
    ob.approxSearchBench();

    if(mainOptions.checkResults) {
        ResultChecking::checkResultsApproxSearches(ob.getResultSet());
    }
}

/**
 * @brief Runs the parallel execution benchmark.
 * 
 * Only uses LinearOctree, so don't pass PointEncoding::NoEncoder!
 */
template <template <typename> class Octree_t, typename Point_t>
void parallelScalabilityBenchmark(std::ofstream &outputFile, EncoderType encoding = EncoderType::NO_ENCODING) {
    auto pointMetaPair = readPointsWithMetadata<Point_t>(mainOptions.inputFile);
    std::vector<Point_t> points = std::move(pointMetaPair.first);
    std::optional<std::vector<PointMetadata>> metadata = std::move(pointMetaPair.second);
    // Sort the point cloud
    auto& enc = getEncoder(encoding);
    auto [codes, box] = enc.sortPoints<Point_t>(points, metadata);

    // Create the searchSet (WARMING: this should be done after sorting since it indexes points!)
    const SearchSet<Point_t> searchSet = SearchSet<Point_t>(mainOptions.numSearches, points);
    OctreeBenchmark<Octree_t, Point_t> ob(points, codes, box, enc, searchSet, outputFile);
    ob.parallelScalabilityBenchmark();
}

template <typename Point_t>
std::vector<Point_t> generateGridCloud(size_t n) {
    std::vector<Point_t> points;
    points.reserve(n * n * n);
    for (size_t i = 0; i < n; i++)  
        for (size_t j = 0; j < n; j++)  
            for (size_t k = 0; k < n; k++)  
                points.push_back(Point_t(i * n * n + j * n + k, i, j, k));
    return points;
}

void approximateSearchLog(std::ofstream &outputFile, EncoderType encoding) {
    assert(encoding != EncoderType::NO_ENCODING);
    auto pointMetaPair = readPointsWithMetadata<Lpoint64>(mainOptions.inputFile);
    std::vector<Lpoint64> points = std::move(pointMetaPair.first);
    std::optional<std::vector<PointMetadata>> metadata = std::move(pointMetaPair.second);
    auto& enc = getEncoder(encoding);
    auto [codes, box] = enc.sortPoints<Lpoint64>(points, metadata);

    auto lin_oct = LinearOctree<Lpoint64>(points, codes, box, enc);
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

template <template <typename> class Octree_t, typename Point_t>
void encodingAndOctreeLog(std::ofstream &outputFile, EncoderType encoding) {
    std::shared_ptr<EncodingOctreeLog> log = std::make_shared<EncodingOctreeLog>();
    log->pointType = getPointName<Point_t>();
    auto pointMetaPair = readPointsWithMetadata<Lpoint64>(mainOptions.inputFile);
    std::vector<Point_t> points = std::move(pointMetaPair.first);
    std::optional<std::vector<PointMetadata>> metadata = std::move(pointMetaPair.second);
    auto& enc = getEncoder(encoding);
    auto [codes, box] = enc.sortPoints<Point_t>(points, metadata, log);
    if constexpr (std::is_same_v<Octree_t<Point_t>, LinearOctree<Point_t>>) {
        LinearOctree<Point_t> oct(points, codes, box, enc, log);
    } else if constexpr (std::is_same_v<Octree_t<Point_t>, Octree<Point_t>>) {
        // only measure total time for pointer-based Octree, we do it here
        TimeWatcher tw;
        tw.start();
        Octree<Point_t> oct(points, box);
        tw.stop();
        log->octreeTime = tw.getElapsedDecimalSeconds();
        log->octreeType = "Octree";
    }
    std::cout << *log << std::endl;
    log->toCSV(outputFile);
}

void outputReorderings(std::ofstream &outputFilePoints, std::ofstream &outputFileOct, EncoderType encoding = EncoderType::NO_ENCODING) {
    auto pointMetaPair = readPointsWithMetadata<Lpoint64>(mainOptions.inputFile);
    std::vector<Lpoint64> points = std::move(pointMetaPair.first);
    std::optional<std::vector<PointMetadata>> metadata = std::move(pointMetaPair.second);

    auto& enc = getEncoder(encoding);
    auto [codes, box] = enc.sortPoints<Lpoint64>(points, metadata);

    // Output reordered points
    outputFilePoints << std::fixed << std::setprecision(3); 
    for(size_t i = 0; i<points.size(); i++) {
        outputFilePoints <<  points[i].getX() << "," << points[i].getY() << "," << points[i].getZ() << "\n";
    }

    if(encoding != EncoderType::NO_ENCODING) {
        // Build linear octree and output bounds
        auto oct = LinearOctree<Lpoint64>(points, codes, box, enc);
        oct.logOctreeBounds(outputFileOct, 6);
    }
}

int main(int argc, char *argv[]) {
    // Set default OpenMP schedule: dynamic and auto chunk size
    omp_set_schedule(omp_sched_dynamic, 0);
    processArgs(argc, argv);
    std::cout << std::fixed << std::setprecision(3); 
    fs::path inputFile = mainOptions.inputFile;
    std::string fileName = inputFile.stem();

    if (!mainOptions.outputDirName.empty()) {
        mainOptions.outputDirName = mainOptions.outputDirName / fileName;
    }

    // Handling Directories
    createDirectory(mainOptions.outputDirName);

    // Open the benchmark output file
    std::string csvFilename = mainOptions.inputFileName + "-" + getCurrentDate() + ".csv";
    std::filesystem::path csvPath = mainOptions.outputDirName / csvFilename;
    std::ofstream outputFile(csvPath, std::ios::app);
    if (!outputFile.is_open()) {
        throw std::ios_base::failure(std::string("Failed to open benchmark output file: ") + csvPath.string());
    }

    switch(mainOptions.benchmarkMode) {
        case BenchmarkMode::SEARCH:
            searchBenchmark<Point>(outputFile);
            searchBenchmark<Point>(outputFile, EncoderType::MORTON_ENCODER_3D);
            searchBenchmark<Point>(outputFile, EncoderType::HILBERT_ENCODER_3D);
        break;
        case BenchmarkMode::COMPARE:
            algoCompBenchmark<Point>(outputFile, EncoderType::HILBERT_ENCODER_3D);
        break;
        case BenchmarkMode::POINT_TYPE:
            searchBenchmark<Point>(outputFile, EncoderType::HILBERT_ENCODER_3D);
            searchBenchmark<Lpoint64>(outputFile, EncoderType::HILBERT_ENCODER_3D);
            searchBenchmark<Lpoint>(outputFile, EncoderType::HILBERT_ENCODER_3D);
        break;
        case BenchmarkMode::APPROX:
            approxSearchBenchmark<Point>(outputFile, EncoderType::HILBERT_ENCODER_3D);
        break;
        case BenchmarkMode::PARALLEL:
            parallelScalabilityBenchmark<Octree, Point>(outputFile);
            parallelScalabilityBenchmark<Octree, Point>(outputFile, EncoderType::HILBERT_ENCODER_3D);
            parallelScalabilityBenchmark<LinearOctree, Point>(outputFile, EncoderType::HILBERT_ENCODER_3D);
        break;
        case BenchmarkMode::LOG_OCTREE:
            // std::filesystem::path unencodedPath = mainOptions.outputDirName / "output_unencoded.csv";
            // std::filesystem::path mortonPath = mainOptions.outputDirName / "output_morton.csv";
            // std::filesystem::path hilbertPath = mainOptions.outputDirName / "output_hilbert.csv";
            // std::filesystem::path unencodedPathOct = mainOptions.outputDirName / "output_unencoded_oct.csv";
            // std::filesystem::path mortonPathOct = mainOptions.outputDirName / "output_morton_oct.csv";
            // std::filesystem::path hilbertPathOct = mainOptions.outputDirName / "output_hilbert_oct.csv";
            // // Open files
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
            std::filesystem::path encAndOctreeLogsPath = mainOptions.outputDirName / "enc_octree_times.csv";
            std::ofstream encAndOctreeLogsFile(encAndOctreeLogsPath);
            EncodingOctreeLog::writeCSVHeader(encAndOctreeLogsFile);
            encodingAndOctreeLog<LinearOctree, Lpoint64>(encAndOctreeLogsFile, EncoderType::MORTON_ENCODER_3D);
            encodingAndOctreeLog<LinearOctree, Lpoint64>(encAndOctreeLogsFile, EncoderType::HILBERT_ENCODER_3D);
            encodingAndOctreeLog<Octree, Lpoint64>(encAndOctreeLogsFile, EncoderType::NO_ENCODING);
            encodingAndOctreeLog<Octree, Lpoint64>(encAndOctreeLogsFile, EncoderType::MORTON_ENCODER_3D);
            encodingAndOctreeLog<Octree, Lpoint64>(encAndOctreeLogsFile, EncoderType::HILBERT_ENCODER_3D);
    }
    return EXIT_SUCCESS;
}
