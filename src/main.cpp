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

namespace fs = std::filesystem;

int main(int argc, char *argv[]) {
  setDefaults();
  processArgs(argc, argv);
  std::cout << "Size of Point: " << sizeof(Point) << " bytes\n";
  std::cout << "Size of Lpoint: " << sizeof(Lpoint) << " bytes\n";

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
  std::vector<Lpoint> points = readPointCloud(mainOptions.inputFile);
  tw.stop();

  std::cout << "Number of read points: " << points.size() << "\n";
  std::cout << "Size of the points array: " << points.size() * sizeof(Lpoint) / (1024.0 * 1024.0) << " MB \n";
  std::cout << "Time to read points: " << tw.getElapsedDecimalSeconds()
            << " seconds\n";
  
  // Benchmark parameters
  // TODO: maybe a better idea is to choose radii based on point cloud density
  const std::vector<float> benchmarkRadii = {0.5, 1.0, 2.5, 3.5, 5.0};
  const size_t repeats = 5;
  const size_t numSearches = 50;

  // Generate the search set
  std::shared_ptr<SearchSet> searchSet = std::make_shared<SearchSet>(numSearches, points);

  // Get the final .csv file
  std::string csvFilename = mainOptions.inputFileName + "-" + getCurrentDate() + ".csv";
  std::filesystem::path csvPath = mainOptions.outputDirName / csvFilename;
  std::ofstream outputFile(csvPath, std::ios::app);
  if (!outputFile.is_open()) {
      throw std::ios_base::failure(std::string("Failed to open benchmark output file: ") + csvPath.string());
  }

  OctreeBenchmarkGeneric<Octree> obPointer(points, numSearches, searchSet, outputFile);
  OctreeBenchmarkGeneric<Octree>::runFullBenchmark(obPointer, benchmarkRadii, repeats, numSearches);

  OctreeBenchmarkGeneric<LinearOctree> obLinear(points, numSearches, searchSet, outputFile);
  OctreeBenchmarkGeneric<LinearOctree>::runFullBenchmark(obLinear, benchmarkRadii, repeats, numSearches);
  obLinear.resultsNeigh[4].clear();
  std::cout << obPointer.resultsNeigh[4].size() << " " << obLinear.resultsNeigh[4].size() << std::endl;
  OctreeBenchmarkGeneric<Octree>::checkResults(obPointer, obLinear);
  return EXIT_SUCCESS;
}