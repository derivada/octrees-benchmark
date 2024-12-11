#include "util.hpp"
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
#include <new>

namespace fs = std::filesystem;

// Global benchmark parameters
const std::vector<float> BENCHMARK_RADII = {0.5, 1.0, 2.5, 5.0};
constexpr size_t REPEATS = 5;
constexpr size_t NUM_SEARCHES = 10000;
constexpr bool CHECK_RESULTS = false;

template <typename T>
void checkVectorMemory(std::vector<T> vec) {
    std::cout << "Size in memory: " << (sizeof(std::vector<T>) + (sizeof(T) * vec.size())) / (1024.0 * 1024.0) << "MB" << std::endl;

    void* data = vec.data();
    // Check if the data is aligned to cache liens
    constexpr std::size_t CACHE_LINE_SIZE = std::hardware_destructive_interference_size;
    if (reinterpret_cast<std::uintptr_t>(data) % CACHE_LINE_SIZE == 0) {
        std::cout << "The vector's data is aligned to a cache line!" << std::endl;
    } else {
        std::cout << "The vector's data is NOT aligned to a cache line." << std::endl;
    }
}

template <template <typename> class Octree_t, typename Point_t, typename Encoder_t>
requires OctreeType<Octree_t<Point_t>>
std::shared_ptr<ResultSet<Point_t>> buildAndRunBenchmark(std::ofstream &outputFile, std::vector<Point_t>& points,
  std::shared_ptr<const SearchSet> searchSet, std::string comment = "") {
  OctreeBenchmark<Octree_t<Point_t>, Point_t, Encoder_t> ob(points, NUM_SEARCHES, searchSet, outputFile, CHECK_RESULTS, comment);
  OctreeBenchmark<Octree_t<Point_t>, Point_t, Encoder_t>::runFullBenchmark(ob, BENCHMARK_RADII, REPEATS, NUM_SEARCHES);
  return ob.getResultSet();
}

template <template <typename> class Octree_t, typename Point_t, typename Encoder_t>
requires OctreeType<Octree_t<Point_t>>
std::shared_ptr<ResultSet<Point_t>> buildAndRunAlgoComparisonBenchmark(std::ofstream &outputFile, std::vector<Point_t>& points,
  std::shared_ptr<const SearchSet> searchSet, std::string comment = "") {
  OctreeBenchmark<Octree_t<Point_t>, Point_t, Encoder_t> ob(points, NUM_SEARCHES, searchSet, outputFile, CHECK_RESULTS, comment);
  OctreeBenchmark<Octree_t<Point_t>, Point_t, Encoder_t>::runAlgoComparisonBenchmark(ob, BENCHMARK_RADII, REPEATS, NUM_SEARCHES);
  return ob.getResultSet();
}

template <PointType Point_t, typename Encoder_t = PointEncoding::MortonEncoder64>
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

  auto resultsPointer = buildAndRunBenchmark<Octree>(outputFile, points, searchSet, "unsorted");
  auto resultsLinear = buildAndRunBenchmark<LinearOctree>(outputFile, points, searchSet);
  auto resultsPointerSorted = buildAndRunBenchmark<Octree>(outputFile, points, searchSet, "sorted");

  // Check the results if needed
  if(CHECK_RESULTS) {
    resultsLinear->checkResults(resultsPointerSorted);
  }
}


// To test different implementations of the same methods (i.e. numNeighbors vs numNeighborsOld)
template <PointType Point_t, typename Encoder_t = PointEncoding::MortonEncoder64>
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
  std::cout << "Size of Point: " << sizeof(Point) << " bytes\n";
  std::cout << "Size of Lpoint: " << sizeof(Lpoint) << " bytes\n";
  std::cout << "Size of Lpoint64: " << sizeof(Lpoint64) << " bytes\n";

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

  TimeWatcher tw;

  // Open the benchmark output file
  std::string csvFilename = mainOptions.inputFileName + "-" + getCurrentDate() + ".csv";
  std::filesystem::path csvPath = mainOptions.outputDirName / csvFilename;
  std::ofstream outputFile(csvPath, std::ios::app);
  if (!outputFile.is_open()) {
      throw std::ios_base::failure(std::string("Failed to open benchmark output file: ") + csvPath.string());
  }

  // Run the comparison benchmarks
  // octreeComparisonBenchmark<Lpoint>(outputFile);
  // octreeComparisonBenchmark<Lpoint64>(outputFile);

  algorithmComparisonBenchmark<Lpoint64>(outputFile);

  // Regular testing grid
  // std::vector<Point> points;
  // constexpr size_t GRID_SIZE = 6;
  // for(int i = 0; i<GRID_SIZE*2; i++) {
  // for(int j = 0; j<GRID_SIZE; j++) {
  // for(int k = 0; k<GRID_SIZE*3; k++) {
  //   Point p = Point(i*GRID_SIZE*GRID_SIZE + j*GRID_SIZE + k, i, j, k);
  //   points.push_back(p);
  // }}}
  // auto points = readPointCloud<Lpoint64>(mainOptions.inputFile);
  // auto loct = LinearOctree(points);
  // loct.printMetadata();
  return EXIT_SUCCESS;
}
