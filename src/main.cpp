#include <filesystem>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <optional>
#include <papi.h>
#include <random>
#include <type_traits>

#include "main_options.hpp"
#include "util.hpp"

#include "readers/handlers.hpp"

#include "benchmarking/benchmarking.hpp"
#include "benchmarking/enc_build_benchmarks.hpp"
#include "benchmarking/locality_benchmarks.hpp"
#include "benchmarking/memory_benchmarks.hpp"
#include "benchmarking/neighbor_benchmarks.hpp"
#include "benchmarking/search_set.hpp"
#include "benchmarking/time_watcher.hpp"

#include "encoding/point_encoder_factory.hpp"

#include "kernels/kernel_factory.hpp"

#include "geometry/point.hpp"
#include "geometry/point_containers.hpp"
#include "geometry/point_metadata.hpp"

#include "structures/linear_octree.hpp"
#include "structures/nanoflann.hpp"
#include "structures/nanoflann_wrappers.hpp"
#include "structures/octree.hpp"
#include "structures/unibn_octree.hpp"

#ifdef HAVE_PCL
#include <pcl/point_cloud.h>
#include <pcl/octree/octree_search.h>
#include <pcl/kdtree/kdtree_flann.h>
#include "structures/pcl_wrappers.hpp"
#endif


namespace fs = std::filesystem;
using namespace PointEncoding;

/**
 * @brief Benchmark neighSearch and numNeighSearch for a given octree configuration (point type + encoder).
 * Compares LinearOctree and PointerOctree. If passed PointEncoding::NoEncoder, only PointerOctree is used.
 */
template <PointContainer Container>
void searchBenchmark(std::ofstream &outputFile, EncoderType encoding = EncoderType::NO_ENCODING) {
    // Load points and put their metadata into a separate vector
    auto pointMetaPair = readPoints<Container>(mainOptions.inputFile);
    auto points = std::move(pointMetaPair.first);
    std::optional<std::vector<PointMetadata>> metadata = std::move(pointMetaPair.second);

    auto& enc = getEncoder(encoding);
    // Sort the point cloud
    auto [codes, box] = enc.sortPoints(points, metadata);
    // Prepare the search set (must be done after sorting since it indexes points)
    SearchSet searchSet = SearchSet(mainOptions.numSearches, points.size());
    // Run the benchmarks
    NeighborsBenchmark octreeBenchmarks(points, codes, box, enc, searchSet, outputFile);   
    octreeBenchmarks.runAllBenchmarks();    
}

template <PointContainer Container>
void localityBenchmark(std::ofstream &outputFile, EncoderType encoding = EncoderType::NO_ENCODING) {
    // Load points and put their metadata into a separate vector
    auto pointMetaPair = readPoints<Container>(mainOptions.inputFile);
    auto points = std::move(pointMetaPair.first);
    std::optional<std::vector<PointMetadata>> metadata = std::move(pointMetaPair.second);
    auto& enc = getEncoder(encoding);
    // Sort the point cloud
    auto [codes, box] = enc.sortPoints(points, metadata);
    
    // Prepare the search set (must be done after sorting since it indexes points)
    // Run the benchmarks
    LocalityBenchmark localityBenchmark(points, codes, box, enc, outputFile);   
    localityBenchmark.histogramLocality(50);    
}


/**
 * @brief Benchmark encoding times for each different encoder (Morton, Hilbert) and build times of the structures under each
 * of this encodings (or without them if possible)
 */
template <PointContainer Container>
void buildEncodingBenchmark(std::ofstream &encodingFile, std::ofstream &buildFile) {
    // Load points and put their metadata into a separate vector
    auto pointMetaPair = readPoints<Container>(mainOptions.inputFile);
    auto points = std::move(pointMetaPair.first);
    std::optional<std::vector<PointMetadata>> metadata = std::move(pointMetaPair.second);
    auto& enc = getEncoder(EncoderType::MORTON_ENCODER_3D);
    auto [codes, box] = enc.sortPoints(points, metadata);
    EncodingBuildBenchmarks encBuildBenchmarks(points, metadata, encodingFile, buildFile);
    encBuildBenchmarks.runEncodingBuildBenchmarks();
}

/*
 * Benchmark similar to encBuild, but we only encode once, don't care about encoding (we use Morton)
 * and build the structure once to get a stable memory profile (e.g. with heaptrack). The theoretical
 * structure size is also printed
*/
template <PointContainer Container>
void memoryBenchmark() {
    // Load points and put their metadata into a separate vector
    auto pointMetaPair = readPoints<Container>(mainOptions.inputFile);
    auto points = std::move(pointMetaPair.first);
    std::optional<std::vector<PointMetadata>> metadata = std::move(pointMetaPair.second);
    auto& enc = getEncoder(EncoderType::MORTON_ENCODER_3D);
    auto [codes, box] = enc.sortPoints(points, metadata);
    
    // Delete the unnecesary metadata so the heap profile has less stuff
    metadata.reset(); 

    MemoryBenchmarks tm(points, codes, box, enc);
    tm.run();
}

template <PointContainer Container>
void outputReorderings(std::ofstream &outputFilePoints, std::ofstream &outputFileOct, EncoderType encoding = EncoderType::NO_ENCODING) {
    auto pointMetaPair = readPoints<Container>(mainOptions.inputFile);
    auto points = std::move(pointMetaPair.first);
    std::optional<std::vector<PointMetadata>> metadata = std::move(pointMetaPair.second);

    auto& enc = getEncoder(encoding);
    auto [codes, box] = enc.sortPoints(points, metadata);

    // Output reordered points
    outputFilePoints << std::fixed << std::setprecision(3); 
    for(size_t i = 0; i<points.size(); i++) {
        outputFilePoints <<  points[i].getX() << "," << points[i].getY() << "," << points[i].getZ() << "\n";
    }

    if(encoding != EncoderType::NO_ENCODING) {
        // Build linear octree and output bounds
        auto oct = LinearOctree(points, codes, box, enc);
        oct.logOctreeBounds(outputFileOct, 6);
    }
}

/// @brief just a debugging method for checking correct knn impl
template <PointContainer Container>
void testKNN(EncoderType encoding = EncoderType::NO_ENCODING) {
    // Load points and put their metadata into a separate vector
    auto pointMetaPair = readPoints<Container>(mainOptions.inputFile);
    auto points = std::move(pointMetaPair.first);
    std::optional<std::vector<PointMetadata>> metadata = std::move(pointMetaPair.second);
    auto& enc = getEncoder(encoding);
    // Sort the point cloud
    auto [codes, box] = enc.sortPoints(points, metadata);
    // Build structures
    LinearOctree loct(points, codes, box, enc);
    NanoflannPointCloud npc(points);
    NanoFlannKDTree kdtree(3, npc, {mainOptions.maxPointsLeaf});
    auto pclCloud = convertCloudToPCL(points);
    
    TimeWatcher twLoct; 
    TimeWatcher twNano;
    TimeWatcher twPcloct;
    TimeWatcher twPclKD;
    long nanosOct = 0, nanosNano = 0, nanosPcloct = 0, nanosPclKD = 0;
    bool seq = true; 
    size_t n = 10000;
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
                    loct.knn(points[searchPoints[i]], k, indexesLoct, distancesLoct);
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
                }
        std::cout << "k = " << k << std::endl << std::fixed << std::setprecision(5);
        std::cout << "  linear octree: "    << (double) nanosOct / 1e9  << std::endl;
        std::cout << "  nanoflann kdtree: " << (double) nanosNano / 1e9 << std::endl;
        std::cout << "  pcl octree: " << (double) nanosPcloct / 1e9 << std::endl;
        std::cout << "  pcl kdtree: " << (double) nanosPclKD / 1e9 << std::endl;
    }
}


template <PointContainer Container>
void testContainersMemLayout(EncoderType encoding = EncoderType::NO_ENCODING) {
    auto pointMetaPair = readPoints<Container>(mainOptions.inputFile);
    auto points = std::move(pointMetaPair.first);
    std::optional<std::vector<PointMetadata>> metadata = std::move(pointMetaPair.second);

    // check 32-byte cache-line alignment
    auto isAligned = [](const void* ptr) -> bool {
        return reinterpret_cast<uintptr_t>(ptr) % 32 == 0;
    };

    auto& enc = getEncoder(encoding);
    auto [codes, box] = enc.sortPoints(points, metadata);

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\nContainer Type: ";
    
    if constexpr(std::is_same_v<Container, PointsAoS>) {
        std::cout << "PointsAoS\n";
        std::cout << "  Total points: " << points.size() << '\n';
        std::cout << "  Size of Point: " << sizeof(Point) << " bytes\n";
        std::cout << "  Total memory used: " << sizeof(Point) * points.size() << " bytes\n";

        if (!points.size()) return;

        const Point* basePtr = &points[0];
        std::cout << "  First point memory address:  " << static_cast<const void*>(basePtr)
                << (isAligned(basePtr) ? " (aligned to 32 bytes)\n" : " (NOT aligned to 32 bytes)\n");

        std::cout << "  Second point memory address: " << static_cast<const void*>(basePtr + 1) << '\n';
        std::cout << "  Layout: Contiguous AoS (Array of Structs)\n";
    } else {
        std::cout << "PointsSoA\n";
        std::cout << "  Total points: " << points.size() << '\n';
    
        const auto* soa = dynamic_cast<const PointsSoA*>(&points);
        if (!soa) {
            std::cerr << "  [Error] Could not cast to PointsSoA\n";
            return;
        }
    
        const size_t N = soa->size();
    
        auto* xs = soa->dataX();
        auto* ys = soa->dataY();
        auto* zs = soa->dataZ();
        auto* ids = soa->dataIds();
    
        auto printMemRange = [&](const char* label, const void* base, size_t count, size_t elementSize) {
            const void* end = static_cast<const char*>((const void*)base) + count * elementSize;
            std::cout << "  [" << label << "] address range:   "
                      << base << " - " << end
                      << " (" << count * elementSize << " bytes) "
                      << (isAligned(base) ? "(aligned to 32 bytes)" : "(NOT aligned)") << '\n';
        };
    
        printMemRange("XS ", xs, N, sizeof(double));
        printMemRange("YS ", ys, N, sizeof(double));
        printMemRange("ZS ", zs, N, sizeof(double));
        printMemRange("IDS", ids, N, sizeof(size_t));
    
        std::cout << "  Layout: SoA (Structure of Arrays), SIMD-friendly\n";
    }

    std::cout << '\n';
}


int main(int argc, char *argv[]) {
    // Set default OpenMP schedule: dynamic and auto chunk size
    omp_set_schedule(omp_sched_dynamic, 0);
    processArgs(argc, argv);
    std::cout << std::fixed << std::setprecision(3); 
    if(mainOptions.cacheProfiling) {
        if (PAPI_library_init(PAPI_VER_CURRENT) != PAPI_VER_CURRENT) {
            std::cerr << "PAPI init error, can't measure cache failures" << std::endl;
            return 1;
        }
    }
    
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
        if(mainOptions.memoryStructure.has_value()) {
            if(mainOptions.containerType == ContainerType::AoS) {
                memoryBenchmark<PointsAoS>();
            } else {
                memoryBenchmark<PointsSoA>();
            }
            // EARLY EXIT: memory benchmark is the only one done if requested, 
            // to allow for easy heap profiling
            return EXIT_SUCCESS; 
        }
        if(mainOptions.buildEncBenchmarks) {
            std::string csvFilenameEnc = mainOptions.inputFileName + "-" + getCurrentDate() + "-encoding.csv";
            std::string csvFilenameBuild = mainOptions.inputFileName + "-" + getCurrentDate() + "-build.csv";
            std::filesystem::path csvPathEnc = mainOptions.outputDirName / csvFilenameEnc;
            std::filesystem::path csvPathBuild = mainOptions.outputDirName / csvFilenameBuild;
            std::ofstream outputFileEnc = std::ofstream(csvPathEnc, std::ios::app);
            std::ofstream outputFileBuild = std::ofstream(csvPathBuild, std::ios::app);
            if(mainOptions.containerType == ContainerType::AoS) {
                buildEncodingBenchmark<PointsAoS>(outputFileEnc, outputFileBuild);
            } else {
                buildEncodingBenchmark<PointsSoA>(outputFileEnc, outputFileBuild);
            }
        }
        
        if(mainOptions.localityBenchmarks) {
            for(EncoderType enc: mainOptions.encodings){
                std::string csvFilenameLocality = mainOptions.inputFileName + "-" + std::string(encoderTypeToString(enc)) + "-locality.csv";
                std::filesystem::path csvPathLocality = mainOptions.outputDirName / csvFilenameLocality;
                std::ofstream outputFileLocality = std::ofstream(csvPathLocality);
                std::cout << "Running locality bench for " << encoderTypeToString(enc) << std::endl;
                if(mainOptions.containerType == ContainerType::AoS) {
                        localityBenchmark<PointsAoS>(outputFileLocality, enc);
                } else {
                    for(EncoderType enc: mainOptions.encodings)
                        localityBenchmark<PointsSoA>(outputFileLocality, enc);
                }
            }
        }

        if(!mainOptions.buildEncBenchmarks && !mainOptions.localityBenchmarks) {
            // Open the benchmark output file
            std::string csvFilename = mainOptions.inputFileName + "-" + getCurrentDate() + ".csv";
            std::filesystem::path csvPath = mainOptions.outputDirName / csvFilename;
            std::ofstream outputFile = std::ofstream(csvPath, std::ios::app);
            if (!outputFile.is_open()) {
                throw std::ios_base::failure(std::string("Failed to open benchmark output file: ") + csvPath.string());
            }
            // Run search benchmarks
            if(mainOptions.containerType == ContainerType::AoS) {
                for(EncoderType enc: mainOptions.encodings)
                    searchBenchmark<PointsAoS>(outputFile, enc);  
            } else {
                for(EncoderType enc: mainOptions.encodings)
                    searchBenchmark<PointsSoA>(outputFile, enc);  
            }
        }
    } else {
        if(mainOptions.containerType == ContainerType::AoS) {
            testContainersMemLayout<PointsAoS>(HILBERT_ENCODER_3D);  
        } else {
            testContainersMemLayout<PointsSoA>(HILBERT_ENCODER_3D);  
        }
        // if(mainOptions.containerType == ContainerType::AoS) {
        //     testKNN<PointsAoS>(HILBERT_ENCODER_3D);  
        // } else {
        //     testKNN<PointsSoA>(HILBERT_ENCODER_3D);  
        // }
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
    }

    return EXIT_SUCCESS;
}
