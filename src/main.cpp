#include "TimeWatcher.hpp"
#include "handlers.hpp"
#include "main_options.hpp"
#include "octree.hpp"
#include "octree_v2.hpp"
#include "octree_linear.hpp"
#include "octree_pointer.hpp"
#include <filesystem> // Only C++17 and beyond
#include <iomanip>
#include <iostream>
#include "benchmarking.hpp"
#include <octree_benchmark.hpp>
#include <random>
#include "octree_linear_sort.hpp"
namespace fs = std::filesystem;

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

  TimeWatcher tw;

  tw.start();
  std::vector<Lpoint> points = readPointCloud(mainOptions.inputFile);

  // {0, 1, ..., 15}^3 testing grid
  // std::vector<Lpoint> points;
  // points.reserve(16*16*16);
  // for(int i = 0; i<16;i++) {
  //     for(int j = 0; j<16;j++) {
  //       for(int k = 0; k<16; k++) {
  //           points.push_back(Lpoint(Point(i*1.0, j*1.0, k*1.0)));
  //       }
  //   }  
  // }

  tw.stop();

  std::cout << "Number of read points: " << points.size() << "\n";
  std::cout << "Time to read points: " << tw.getElapsedDecimalSeconds()
            << " seconds\n";

  // Global Octree Creation
  std::cout << "Building global octree..." << std::endl;
  tw.start();
  Octree gOctree(points);
  tw.stop();
  std::cout << "Time to build global octree: " << tw.getElapsedDecimalSeconds()
            << " seconds\n";
  std::ofstream gOctreeStream(mainOptions.outputDirName / "global_octree.txt");
  gOctree.writeOctree(gOctreeStream, 0);

  std::vector<Lpoint> lsOctreePoints(points);
  
  OctreeBenchmark ob(points, lsOctreePoints);
  ob.benchmarkSearchNeighSphere(10, true);

  return EXIT_SUCCESS;
}
