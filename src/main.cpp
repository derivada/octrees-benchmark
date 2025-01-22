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
const std::vector<float> BENCHMARK_RADII = {2.5, 5.0, 7.5, 10.0};
constexpr size_t REPEATS = 5;
constexpr size_t NUM_SEARCHES = 10000;
constexpr bool CHECK_RESULTS = false;

template <template <typename, typename> class Octree_t, PointType Point_t, typename Encoder_t>
std::shared_ptr<ResultSet<Point_t>> runSearchBenchmark(std::ofstream &outputFile, std::vector<Point_t>& points,
  std::shared_ptr<const SearchSet> searchSet, std::string comment = "") {
  OctreeBenchmark<Octree_t, Point_t, Encoder_t> ob(points, NUM_SEARCHES, searchSet, outputFile, CHECK_RESULTS, comment);
  ob.searchBench(BENCHMARK_RADII, REPEATS, NUM_SEARCHES);
  return ob.getResultSet();
}

template <template <typename, typename> class Octree_t, PointType Point_t, typename Encoder_t>
std::shared_ptr<ResultSet<Point_t>> runSearchImplComparisonBenchmark(std::ofstream &outputFile, std::vector<Point_t>& points,
  std::shared_ptr<const SearchSet> searchSet, std::string comment = "") {
  OctreeBenchmark<Octree_t, Point_t, Encoder_t> ob(points, NUM_SEARCHES, searchSet, outputFile, CHECK_RESULTS, comment);
  ob.searchImplComparisonBench(BENCHMARK_RADII, REPEATS, NUM_SEARCHES);
  return ob.getResultSet();
}

template <template <typename, typename> class Octree_t, PointType Point_t, typename Encoder_t, Kernel_t kernel>
std::shared_ptr<ResultSet<Point_t>> runSingleKernelRadiiBenchmark(std::ofstream &outputFile, std::vector<Point_t>& points,
  std::shared_ptr<const SearchSet> searchSet, const float radius, std::string comment = "") {
  OctreeBenchmark<Octree_t, Point_t, Encoder_t> ob(points, NUM_SEARCHES, searchSet, outputFile, CHECK_RESULTS, comment);
  ob.template benchmarkSearchNeighSeq<kernel>(REPEATS, radius);
  return ob.getResultSet();
}

template <PointType Point_t, typename Encoder_t>
void searchBenchmark(std::ofstream &outputFile) {
  TimeWatcher tw;
  tw.start();
  auto points = readPointCloud<Point_t>(mainOptions.inputFile);
  tw.stop();

  std::cout << "Number of read points: " << points.size() << "\n";
  std::cout << "Time to read points: " << tw.getElapsedDecimalSeconds()
            << " seconds\n";
  checkVectorMemory(points);
  std::shared_ptr<const SearchSet> searchSet = std::make_shared<const SearchSet>(NUM_SEARCHES, points);

  if constexpr (std::is_same_v<Encoder_t, PointEncoding::NoEncoder>) {
    // Only do pointer octree, since we are not encoding the points
    runSearchBenchmark<Octree, Point_t, PointEncoding::NoEncoder>(outputFile, points, searchSet);
  } else {
    // Do both linear (which encodes and sorts the points) and pointer octree after it
    auto resultsLinear = runSearchBenchmark<LinearOctree, Point_t, Encoder_t>(outputFile, points, searchSet);
    auto resultsPointer = runSearchBenchmark<Octree, Point_t, Encoder_t>(outputFile, points, searchSet);
    if(CHECK_RESULTS) {
      resultsLinear->checkResults(resultsPointer);
    }
  }
}

template <PointType Point_t, typename Encoder_t>
void searchImplComparisonBenchmark(std::ofstream &outputFile) {
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

  auto results = runSearchImplComparisonBenchmark<LinearOctree, Point_t, Encoder_t>(outputFile, points, searchSet);

  // Check the results if needed
  if(CHECK_RESULTS) {
    results->checkResultsAlgo();
  }
}

template <PointType Point_t, typename Encoder_t, Kernel_t kernel>
void sequentialVsShuffleBenchmark(std::ofstream &outputFile, const float radius) {
  TimeWatcher tw;
  tw.start();
  auto points = readPointCloud<Point_t>(mainOptions.inputFile);
  tw.stop();

  std::cout << "Number of read points: " << points.size() << "\n";
  std::cout << "Time to read points: " << tw.getElapsedDecimalSeconds()
            << " seconds\n";
  checkVectorMemory(points);

  // Generate a shared search set for each benchmark execution
  std::shared_ptr<SearchSet> searchSetSeq = std::make_shared<SearchSet>(points, false);

  // In this benchmark we only do one radii and one kernel, because otherwise it would be too much
  // and we are only interested in the difference between sequential and shuffled points
  runSingleKernelRadiiBenchmark<LinearOctree, Point_t, Encoder_t, kernel>(outputFile, points, searchSetSeq, radius, "Sequential");
  // free memory from this first sequential execution since we don't need it anymore
  searchSetSeq->searchPoints.clear();
  searchSetSeq->searchKNNLimits.clear();

  std::shared_ptr<const SearchSet> searchSetShuffle = std::make_shared<const SearchSet>(points, true);
  runSingleKernelRadiiBenchmark<LinearOctree, Point_t, Encoder_t, kernel>(outputFile, points, searchSetShuffle, radius, "Shuffled");
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
  /*
    // Compare linear and pointer octree, both encoded with Hilbert and Morton SFC order
    searchBenchmark<Lpoint64, PointEncoding::HilbertEncoder3D>(outputFile);
    searchBenchmark<Lpoint64, PointEncoding::MortonEncoder3D>(outputFile);

    // Baseline with no encoder (only pointer-based octree)
    searchBenchmark<Lpoint64, PointEncoding::NoEncoder>(outputFile);
  */
  sequentialVsShuffleBenchmark<Lpoint64, PointEncoding::HilbertEncoder3D, Kernel_t::sphere>(outputFile, 5.0);
  return EXIT_SUCCESS;
}
