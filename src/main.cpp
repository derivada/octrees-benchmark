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
#include "Lpoint.hpp"
#include "Lpoint64.hpp"
#include <new>

namespace fs = std::filesystem;

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

template <PointType Point_t>
void octreeComparisonBenchmark(std::ofstream &outputFile, bool check = false) {
  const std::vector<float> benchmarkRadii = {0.5, 1.0, 2.5, 3.5, 5.0};
  const size_t repeats = 5;
  const size_t numSearches = 1000;
  TimeWatcher tw;
  tw.start();
  auto points = readPointCloud<Point_t>(mainOptions.inputFile);
  tw.stop();

  std::cout << "Number of read points: " << points.size() << "\n";
  std::cout << "Time to read points: " << tw.getElapsedDecimalSeconds()
            << " seconds\n";
  checkVectorMemory(points);
  std::shared_ptr<const SearchSet> searchSet = std::make_shared<const SearchSet>(numSearches, points);
  OctreeBenchmark<LinearOctree<Point_t>, Point_t> obLinear(points, numSearches, searchSet, outputFile, check);
  OctreeBenchmark<LinearOctree<Point_t>, Point_t>::runFullBenchmark(obLinear, benchmarkRadii, repeats, numSearches);

  OctreeBenchmark<Octree<Point_t>, Point_t> obPointer(points, numSearches, searchSet, outputFile, check);
  OctreeBenchmark<Octree<Point_t>, Point_t>::runFullBenchmark(obPointer, benchmarkRadii, repeats, numSearches);

  if(check)
    OctreeBenchmark<Octree<Point_t>, Point_t>::checkResults(obPointer, obLinear);
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

  // Benchmark parameters
  // TODO: maybe a better idea is to choose radii based on point cloud density
  const std::vector<float> benchmarkRadii = {0.5, 1.0, 2.5, 3.5, 5.0};
  const size_t repeats = 5;
  const size_t numSearches = 1000;
  std::string csvFilename = mainOptions.inputFileName + "-" + getCurrentDate() + ".csv";
  std::filesystem::path csvPath = mainOptions.outputDirName / csvFilename;
  std::ofstream outputFile(csvPath, std::ios::app);
  if (!outputFile.is_open()) {
      throw std::ios_base::failure(std::string("Failed to open benchmark output file: ") + csvPath.string());
  }
  octreeComparisonBenchmark<Lpoint>(outputFile);
  octreeComparisonBenchmark<Lpoint64>(outputFile);
  return EXIT_SUCCESS;
}