#include "TimeWatcher.hpp"
#include "handlers.hpp"
#include "main_options.hpp"
#include "octree.hpp"
#include "octree_linear.hpp"
#include "octree_linear_old.hpp"
#include <filesystem> // Only C++17 and beyond
#include <iomanip>
#include <iostream>
#include "benchmarking.hpp"
#include <octree_benchmark.hpp>
#include <random>
#include "NeighborKernels/KernelFactory.hpp"
#include "octree_benchmark_generic.hpp"
#include "Lpoint.hpp"
#include "Lpoint2.hpp"
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

int main(int argc, char *argv[]) {
  setDefaults();
  processArgs(argc, argv);
  std::cout << "Size of Point: " << sizeof(Point) << " bytes\n";
  std::cout << "Size of Lpoint: " << sizeof(Lpoint) << " bytes\n";
  std::cout << "Size of Lpoint2: " << sizeof(Lpoint2) << " bytes\n";

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

  tw.start();
  auto points = readPointCloud<Lpoint>(mainOptions.inputFile);
  tw.stop();

  std::cout << "Number of read points: " << points.size() << "\n";
  std::cout << "Time to read points: " << tw.getElapsedDecimalSeconds()
            << " seconds\n";
  checkVectorMemory(points);

  tw.start();
  auto points2 = readPointCloud<Lpoint2>(mainOptions.inputFile);
  tw.stop();

  std::cout << "Number of read points: " << points.size() << "\n";
  std::cout << "Time to read points: " << tw.getElapsedDecimalSeconds()
            << " seconds\n";
  checkVectorMemory(points2);

  // Benchmark parameters
  // TODO: maybe a better idea is to choose radii based on point cloud density
  const std::vector<float> benchmarkRadii = {0.5, 1.0, 2.5, 3.5, 5.0};
  const size_t repeats = 5;
  const size_t numSearches = 100;

  // Generate the search set
  std::shared_ptr<const SearchSet> searchSet = std::make_shared<const SearchSet>(numSearches, points);

  // Get the final .csv file
  std::string csvFilename = mainOptions.inputFileName + "-" + getCurrentDate() + ".csv";
  std::filesystem::path csvPath = mainOptions.outputDirName / csvFilename;
  std::ofstream outputFile(csvPath, std::ios::app);
  if (!outputFile.is_open()) {
      throw std::ios_base::failure(std::string("Failed to open benchmark output file: ") + csvPath.string());
  }
  
  OctreeBenchmarkGeneric<LinearOctree> obLinear(points, numSearches, searchSet, outputFile);
  OctreeBenchmarkGeneric<LinearOctree>::runFullBenchmark(obLinear, benchmarkRadii, repeats, numSearches);

  OctreeBenchmarkGeneric<Octree> obPointer(points, numSearches, searchSet, outputFile);
  OctreeBenchmarkGeneric<Octree>::runFullBenchmark(obPointer, benchmarkRadii, repeats, numSearches);
  // OctreeBenchmarkGeneric<Octree>::checkResults(obPointer, obLinear);
  return EXIT_SUCCESS;
}