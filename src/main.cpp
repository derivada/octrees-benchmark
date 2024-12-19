#include "util.hpp"
#include "type_names.hpp"
#include "TimeWatcher.hpp"
#include "handlers.hpp"
#include "main_options.hpp"
#include "octree.hpp"
#include "octree_linear.hpp"
#include <filesystem> // Only C++17 and beyond
#include <iomanip>
#include <iostream>
#include "benchmarking.hpp"
#include <random>
#include "NeighborKernels/KernelFactory.hpp"
#include "octree_benchmark.hpp"
#include "Geometry/Lpoint.hpp"
#include "Geometry/Lpoint64.hpp"
#include "PointEncoding/morton_encoder.hpp"
#include "PointEncoding/hilbert_encoder.hpp"
#include <new>

namespace fs = std::filesystem;

// Global benchmark parameters
const std::vector<float> BENCHMARK_RADII = {0.5, 1.0, 2.5, 5.0};
constexpr size_t REPEATS = 5;
constexpr size_t NUM_SEARCHES = 1000;
constexpr bool CHECK_RESULTS = true;

template <template <typename, typename> class Octree_t, PointType Point_t, typename Encoder_t>
std::shared_ptr<ResultSet<Point_t>> buildAndRunBenchmark(std::ofstream &outputFile, std::vector<Point_t>& points,
  std::shared_ptr<const SearchSet> searchSet, std::string comment = "") {
  OctreeBenchmark<Octree_t, Point_t, Encoder_t> ob(points, NUM_SEARCHES, searchSet, outputFile, CHECK_RESULTS, comment);
  OctreeBenchmark<Octree_t, Point_t, Encoder_t>::runFullBenchmark(ob, BENCHMARK_RADII, REPEATS, NUM_SEARCHES);
  return ob.getResultSet();
}

template <template <typename, typename> class Octree_t, PointType Point_t, typename Encoder_t>
std::shared_ptr<ResultSet<Point_t>> buildAndRunAlgoComparisonBenchmark(std::ofstream &outputFile, std::vector<Point_t>& points,
  std::shared_ptr<const SearchSet> searchSet, std::string comment = "") {
  OctreeBenchmark<Octree_t, Point_t, Encoder_t> ob(points, NUM_SEARCHES, searchSet, outputFile, CHECK_RESULTS, comment);
  OctreeBenchmark<Octree_t, Point_t, Encoder_t>::runAlgoComparisonBenchmark(ob, BENCHMARK_RADII, REPEATS, NUM_SEARCHES);
  return ob.getResultSet();
}

template <PointType Point_t, typename Encoder_t>
void octreeComparisonBenchmark(std::ofstream &outputFile) {
  // TODO: maybe a better idea is to choose radii based on point cloud density
  TimeWatcher tw;
  tw.start();
  auto points = readPointCloud<Point_t>(mainOptions.inputFile);
  tw.stop();

  std::cout << "Number of read points: " << points.size() << "\n";
  std::cout << "Time to read points: " << tw.getElapsedDecimalSeconds()
            << " seconds\n";
  checkVectorMemory(points);

  // Generate a shared search set for each benchmark execution
  std::shared_ptr<const SearchSet> searchSet = std::make_shared<const SearchSet>(NUM_SEARCHES, points);

  buildAndRunBenchmark<Octree, Point_t, PointEncoding::NoEncoder>(outputFile, points, searchSet);
  auto resultsLinearMorton = buildAndRunBenchmark<LinearOctree, Point_t, PointEncoding::MortonEncoder3D>(outputFile, points, searchSet);
  auto resultsPointerMorton = buildAndRunBenchmark<Octree, Point_t, PointEncoding::MortonEncoder3D>(outputFile, points, searchSet);
  auto resultsLinearHilbert = buildAndRunBenchmark<LinearOctree, Point_t, PointEncoding::HilbertEncoder3D>(outputFile, points, searchSet);
  auto resultsPointerHilbert = buildAndRunBenchmark<Octree, Point_t, PointEncoding::HilbertEncoder3D>(outputFile, points, searchSet);

  // Check the results if needed
  if(CHECK_RESULTS) {
    resultsLinearMorton->checkResults(resultsPointerMorton);
    resultsLinearHilbert->checkResults(resultsPointerHilbert);
  }
}

// To test different implementations of the same methods (i.e. numNeighbors vs numNeighborsOld)
template <PointType Point_t, typename Encoder_t>
void algorithmComparisonBenchmark(std::ofstream &outputFile) {
  // TODO: maybe a better idea is to choose radii based on point cloud density
  TimeWatcher tw;
  tw.start();
  auto points = readPointCloud<Point_t>(mainOptions.inputFile);
  tw.stop();

  std::cout << "Number of read points: " << points.size() << "\n";
  std::cout << "Time to read points: " << tw.getElapsedDecimalSeconds()
            << " seconds\n";
  checkVectorMemory(points);

  // Generate a shared search set for each benchmark execution
  std::shared_ptr<const SearchSet> searchSet = std::make_shared<const SearchSet>(NUM_SEARCHES, points);

  auto results = buildAndRunAlgoComparisonBenchmark<LinearOctree, Point_t, Encoder_t>(outputFile, points, searchSet);

  // Check the results if needed
  if(CHECK_RESULTS) {
    results->checkResultsAlgo();
  }
}

int main(int argc, char *argv[]) {
  setDefaults();
  processArgs(argc, argv);
  
  fs::path inputFile = mainOptions.inputFile;
  std::string fileName = inputFile.stem();

  if (!mainOptions.outputDirName.empty()) {
    mainOptions.outputDirName = mainOptions.outputDirName / fileName;
  }

  // Handling Directories
  createDirectory(mainOptions.outputDirName);

  // Print three decimals
  std::cout << std::fixed;
  std::cout << std::setprecision(3);

  // Open the benchmark output file
  std::string csvFilename = mainOptions.inputFileName + "-" + getCurrentDate() + ".csv";
  std::filesystem::path csvPath = mainOptions.outputDirName / csvFilename;
  std::ofstream outputFile(csvPath, std::ios::app);
  if (!outputFile.is_open()) {
      throw std::ios_base::failure(std::string("Failed to open benchmark output file: ") + csvPath.string());
  }

  octreeComparisonBenchmark<Lpoint64, PointEncoding::HilbertEncoder3D>(outputFile);
  octreeComparisonBenchmark<Lpoint64, PointEncoding::MortonEncoder3D>(outputFile);
  
  return EXIT_SUCCESS;
}
