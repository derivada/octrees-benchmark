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

// Global benchmark parameters
const std::vector<float> benchmarkRadii = {0.5, 1.0, 2.5, 5.0, 10.0};
constexpr size_t repeats = 5;
constexpr size_t numSearches = 1000;

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

// template <PointType Point_t>
// double getDensity(std::vector<Point_t> &points) {
//   std::cout << points.size() << std::endl;
//   LinearOctree<Point_t> oct(points);
//   return oct.getDensity();
// }

template <PointType Point_t>
void octreeComparisonBenchmark(std::ofstream &outputFile, bool check = false) {
  // TODO: maybe a better idea is to choose radii based on point cloud density
  TimeWatcher tw;
  tw.start();
  auto points = readPointCloud<Point_t>(mainOptions.inputFile);
  tw.stop();

  std::cout << "Number of read points: " << points.size() << "\n";
  std::cout << "Time to read points: " << tw.getElapsedDecimalSeconds()
            << " seconds\n";
  checkVectorMemory(points);

  std::shared_ptr<const SearchSet> searchSet = std::make_shared<const SearchSet>(numSearches, points);

  OctreeBenchmark<Octree<Point_t>, Point_t> obPointer(points, numSearches, searchSet, outputFile, check, "unsorted");
  OctreeBenchmark<Octree<Point_t>, Point_t>::runFullBenchmark(obPointer, benchmarkRadii, repeats, numSearches);

  OctreeBenchmark<LinearOctree<Point_t>, Point_t> obLinear(points, numSearches, searchSet, outputFile, check);
  OctreeBenchmark<LinearOctree<Point_t>, Point_t>::runFullBenchmark(obLinear, benchmarkRadii, repeats, numSearches);

  OctreeBenchmark<Octree<Point_t>, Point_t> obPointerSorted(points, numSearches, searchSet, outputFile, check, "sorted");
  OctreeBenchmark<Octree<Point_t>, Point_t>::runFullBenchmark(obPointerSorted, benchmarkRadii, repeats, numSearches);

  if(check) {
    OctreeBenchmark<Octree<Point_t>, Point_t>::checkResults(obPointerSorted, obLinear);
  }
}

template <OctreeType Octree_t, PointType Point_t>
void buildAndRunSimpleBenchmark(std::ofstream &outputFile, std::vector<Point_t> &points, std::shared_ptr<const SearchSet> searchSet, std::string comment = "") {
  OctreeBenchmark<Octree_t, Point_t> ob(points, numSearches, searchSet, outputFile, false, comment);
  for(int i = 0; i<benchmarkRadii.size(); i++){
    float radius = benchmarkRadii[i];
    ob.template benchmarkSearchNeigh<Kernel_t::sphere>(repeats, radius);
    ob.template benchmarkNumNeigh<Kernel_t::sphere>(repeats, radius);
    std::cout << getCurrentDate() << " (" << i+1 << "/" << benchmarkRadii.size() << ") Benchmark with radius " << benchmarkRadii[i] << " completed" << std::endl;
  }
}

template <PointType Point_t>
void octreeSimpleBenchmark(std::ofstream &outputFile) {
  // For bigger datasets
  TimeWatcher tw;
  tw.start();
  auto points = readPointCloud<Point_t>(mainOptions.inputFile);
  tw.stop();

  std::cout << "Number of read points: " << points.size() << "\n";
  std::cout << "Time to read points: " << tw.getElapsedDecimalSeconds()
            << " seconds\n";
  checkVectorMemory(points);

  std::shared_ptr<const SearchSet> searchSet = std::make_shared<const SearchSet>(numSearches, points);
  std::cout << "Running benchmarks on octree " << getOctreeName<Octree<Point_t>, Point_t>() << std::endl;
  buildAndRunSimpleBenchmark<Octree<Point_t>, Point_t>(outputFile, points, searchSet, "unsorted");
  std::cout << "Running benchmarks on octree " << getOctreeName<LinearOctree<Point_t>, Point_t>() << std::endl;
  buildAndRunSimpleBenchmark<LinearOctree<Point_t>, Point_t>(outputFile, points, searchSet);
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

  std::string csvFilename = mainOptions.inputFileName + "-" + getCurrentDate() + ".csv";
  std::filesystem::path csvPath = mainOptions.outputDirName / csvFilename;
  std::ofstream outputFile(csvPath, std::ios::app);
  if (!outputFile.is_open()) {
      throw std::ios_base::failure(std::string("Failed to open benchmark output file: ") + csvPath.string());
  }

  octreeComparisonBenchmark<Lpoint>(outputFile, false);
  octreeComparisonBenchmark<Lpoint64>(outputFile, false);

  return EXIT_SUCCESS;
}
