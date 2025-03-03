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
#include "PointEncoding/morton_encoder.hpp"
#include "PointEncoding/hilbert_encoder.hpp"
#include "result_checking.hpp"

namespace fs = std::filesystem;

template <template <typename, typename> class Octree_t, PointType Point_t, typename Encoder_t>
ResultSet<Point_t> runSearchBenchmark(std::ofstream &outputFile, std::vector<Point_t>& points,
  std::shared_ptr<const SearchSet> searchSet, std::optional<std::vector<PointMetadata>> &metadata = std::nullopt,
  std::string comment = "", bool useParallel = true) {
  OctreeBenchmark<Octree_t, Point_t, Encoder_t> ob(points, metadata, mainOptions.numSearches, searchSet, outputFile, 
     comment, mainOptions.checkResults, mainOptions.useWarmup, useParallel);
  ob.searchBench(mainOptions.benchmarkRadii, mainOptions.repeats, mainOptions.numSearches);
  return ob.getResultSet();
}

/**
 * Benchmark neighSearch and numNeighSearch for a given octree configuration (point type + encoder).
 * Compares LinearOctree and PointerOctree. If passed PointEncoding::NoEncoder, only PointerOctree is used.
 */
template <PointType Point_t, typename Encoder_t>
void searchBenchmark(std::ofstream &outputFile) {
  auto pointMetaPair = readPointsWithMetadata<Point_t>(mainOptions.inputFile);
  std::vector<Point_t> points = std::move(pointMetaPair.first);
  std::optional<std::vector<PointMetadata>> metadata = std::move(pointMetaPair.second);
  std::shared_ptr<const SearchSet> searchSet = std::make_shared<const SearchSet>(mainOptions.numSearches, points);

  if constexpr (std::is_same_v<Encoder_t, PointEncoding::NoEncoder>) {
    // Only do pointer octree, since we are not encoding the points
    OctreeBenchmark<Octree, Point_t, Encoder_t> obPointer(points, metadata, mainOptions.numSearches, searchSet, outputFile);
    obPointer.searchBench(mainOptions.benchmarkRadii, mainOptions.repeats, mainOptions.numSearches);
  } else {
    // Do both linear (which encodes and sorts the points) and pointer octree after it
    OctreeBenchmark<LinearOctree, Point_t, Encoder_t> obLinear(points, metadata, mainOptions.numSearches, searchSet, outputFile);
    obLinear.searchBench(mainOptions.benchmarkRadii, mainOptions.repeats, mainOptions.numSearches);
    obLinear.deleteOctree();
    OctreeBenchmark<Octree, Point_t, Encoder_t> obPointer(points, metadata, mainOptions.numSearches, searchSet, outputFile);
    obPointer.searchBench(mainOptions.benchmarkRadii, mainOptions.repeats, mainOptions.numSearches);
    obPointer.deleteOctree();
    if(mainOptions.checkResults) {
        ResultChecking::checkResultsLinearVsPointer(obLinear.getResultSet(), obPointer.getResultSet());
    }
  }
}

/**
 * Benchmark oldNeighSearch vs neighSearch and oldNumNeighSearch vs numNeighSearch for a given octree configuration (point type + encoder),
 * in order to see the performance improvement of the new implementation.
 * 
 * Only uses LinearOctree, since that's where the implementation is being improved.
 */
template <PointType Point_t, typename Encoder_t>
void algoCompBenchmark(std::ofstream &outputFile) {
  auto pointMetaPair = readPointsWithMetadata<Point_t>(mainOptions.inputFile);
  std::vector<Point_t> points = std::move(pointMetaPair.first);
  std::optional<std::vector<PointMetadata>> metadata = std::move(pointMetaPair.second);
  std::shared_ptr<const SearchSet> searchSet = std::make_shared<const SearchSet>(mainOptions.numSearches, points);

  OctreeBenchmark<LinearOctree, Point_t, Encoder_t> ob(points, metadata, mainOptions.numSearches, searchSet, outputFile, "");
  ob.searchImplComparisonBench(mainOptions.benchmarkRadii, mainOptions.repeats, mainOptions.numSearches);

  if(mainOptions.checkResults) {
    ResultChecking::checkResultsAlgoComp(ob.getResultSet());
  }
}

template <PointType Point_t, typename Encoder_t>
void approxSearchBenchmark(std::ofstream &outputFile) {
  auto pointMetaPair = readPointsWithMetadata<Point_t>(mainOptions.inputFile);
  std::vector<Point_t> points = std::move(pointMetaPair.first);
  std::optional<std::vector<PointMetadata>> metadata = std::move(pointMetaPair.second);  
  std::shared_ptr<const SearchSet> searchSet = std::make_shared<const SearchSet>(mainOptions.numSearches, points);

  OctreeBenchmark<LinearOctree, Point_t, Encoder_t> ob(points, metadata, mainOptions.numSearches, searchSet, outputFile);
  ob.approxSearchBench(mainOptions.benchmarkRadii, mainOptions.repeats, mainOptions.numSearches, mainOptions.approximateTolerances);

  if(mainOptions.checkResults) {
    ResultChecking::checkResultsApproxSearches(ob.getResultSet());
  }
}

/**
 * Runs the search benchmark for both sequential slices and random points
 * The start point for the sequential slice is chosed at random from [0, points.size() - numSearches]
 */
template <PointType Point_t, typename Encoder_t, Kernel_t kernel>
void sequentialVsShuffleBenchmark(std::ofstream &outputFile) {
  auto pointMetaPair = readPointsWithMetadata<Point_t>(mainOptions.inputFile);
  std::vector<Point_t> points = std::move(pointMetaPair.first);
  std::optional<std::vector<PointMetadata>> metadata = std::move(pointMetaPair.second);
  /**
   * We do shuffle searchSet first since points are chosen at random and we don't care if they are already ordered
   * by the encoder. We then do sequential searchSet with the points already sorted in encoder-order so they
   * have the spatial locality, which is what we want to test.
   */
  std::shared_ptr<SearchSet> searchSetShuffle = std::make_shared<SearchSet>(mainOptions.numSearches, points);
  runSearchBenchmark<LinearOctree, Point_t, Encoder_t>(outputFile, points, searchSetShuffle, metadata, "shuffled");
  searchSetShuffle->searchPoints.clear();
  searchSetShuffle->searchKNNLimits.clear();
  std::shared_ptr<SearchSet> searchSetSeq = std::make_shared<SearchSet>(mainOptions.numSearches, points, true);
  runSearchBenchmark<LinearOctree, Point_t, Encoder_t>(outputFile, points, searchSetSeq, metadata, "sequential");
}

template <PointType Point_t>
std::vector<Point_t> generateGridCloud(size_t n) {
  std::vector<Point_t> points;
  points.reserve(n * n * n);
  for (size_t i = 0; i < n; i++)  
    for (size_t j = 0; j < n; j++)  
      for (size_t k = 0; k < n; k++)  
        points.push_back(Point_t(i * n * n + j * n + k, i, j, k));
  return points;
}

/**
 * Constructs a linear octree and logs it to two separate files
 *  octree-structure.txt contains the structure of the octree
 *  encoded-points.csv contains the encodings of each point and its x,y,z coordinates
 * 
 * Keep in mind that those files (specially the second, since it will be on the same order of magnitude as the .las file) 
 * will be huge for big clouds, so only use this for small clouds (e.g. <5M points)
 */
template <PointType Point_t, typename Encoder_t>
void linearOctreeLog(std::ofstream &outputFile, bool useGridCloud = false, size_t gridCloudSize = 16) {
  TimeWatcher tw;
  tw.start();
  std::vector<Point_t> points;
  std::string cloudName;
  if(useGridCloud) {
    points = generateGridCloud<Point_t>(gridCloudSize);
    cloudName = "regular-grid-" + std::to_string(gridCloudSize);
  } else {
    points = readPointCloud<Point_t>(mainOptions.inputFile);
    cloudName = mainOptions.inputFileName;
  }
  tw.stop();
  pointCloudReadLog(points, tw, mainOptions.inputFile);
  std::string encoderTypename = PointEncoding::getShortEncoderName<Encoder_t>();
  
  // Open the files for outputting the octree structure and the encoded points
  // Dont put the timestamp on this file to not occupy a lot of memory if we forget to remove them after user
  std::string octreeOutputFilename = cloudName + "-" + encoderTypename + "-octree-structure.txt";
  std::string encodedPointsFilename = cloudName + "-" + encoderTypename + "-encoded-points.csv";
  std::filesystem::path octreeOutputPath = mainOptions.outputDirName / octreeOutputFilename;
  std::filesystem::path encodedPointsPath = mainOptions.outputDirName / encodedPointsFilename;
  std::ofstream octreeOutputFile(octreeOutputPath, std::ios::out);
  std::ofstream encodedPointsFile(encodedPointsPath, std::ios::out);
  if (!octreeOutputFile.is_open() || !encodedPointsFile.is_open()) {
    throw std::ios_base::failure(std::string("Failed to open octree log output files"));
  }
  std::optional<std::vector<PointMetadata>> metadata = std::nullopt;
  LinearOctree<Point_t, Encoder_t> linearOctree(points, metadata, true);
  std::cout << "OCTREE LOGGING MODE -- Outputting linear octree structure built from point cloud " << cloudName << " to " 
    << octreeOutputFilename << " and encoded points to " << encodedPointsFilename << std::endl;
  linearOctree.logOctree(octreeOutputFile, encodedPointsFile);
  octreeOutputFile.close();
  encodedPointsFile.close();
}

int main(int argc, char *argv[]) {
  setDefaults();
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
      searchBenchmark<Lpoint64, PointEncoding::HilbertEncoder3D>(outputFile);
      searchBenchmark<Lpoint64, PointEncoding::MortonEncoder3D>(outputFile);
      searchBenchmark<Lpoint64, PointEncoding::NoEncoder>(outputFile);
    break;
    case BenchmarkMode::COMPARE:
      algoCompBenchmark<Lpoint64, PointEncoding::HilbertEncoder3D>(outputFile);
    break;
    case BenchmarkMode::POINT_TYPE:
      searchBenchmark<Point, PointEncoding::HilbertEncoder3D>(outputFile);
      searchBenchmark<Lpoint64, PointEncoding::HilbertEncoder3D>(outputFile);
      searchBenchmark<Lpoint, PointEncoding::HilbertEncoder3D>(outputFile);
    break;
    case BenchmarkMode::APPROX:
      approxSearchBenchmark<Lpoint64, PointEncoding::HilbertEncoder3D>(outputFile);
    break;
    case BenchmarkMode::SEQUENTIAL:
      sequentialVsShuffleBenchmark<Lpoint64, PointEncoding::HilbertEncoder3D, Kernel_t::sphere>(outputFile);
    break;
    case BenchmarkMode::LOG_OCTREE:
      linearOctreeLog<Lpoint64, PointEncoding::MortonEncoder3D>(outputFile);
      linearOctreeLog<Lpoint64, PointEncoding::MortonEncoder3D>(outputFile, true, 16);
      linearOctreeLog<Lpoint64, PointEncoding::HilbertEncoder3D>(outputFile);
      linearOctreeLog<Lpoint64, PointEncoding::HilbertEncoder3D>(outputFile, true, 16);
  }

  return EXIT_SUCCESS;
}
