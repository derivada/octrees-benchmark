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


  // Global Pointer Octree Creation
  std::cout << "Building global (pointer) octree..." << std::endl;
  tw.start();
  PointerOctree pOctree(points);
  tw.stop();
  std::cout << "Time to build global (pointer) octree: " << tw.getElapsedDecimalSeconds()
            << " seconds\n";
  std::ofstream pOctreeStream(mainOptions.outputDirName / "pointer_octree.txt");
  pOctree.writeOctree(pOctreeStream, 0);

  // Global Linear Octree Creation
  std::cout << "Building global (linear) octree..." << std::endl;
  tw.start();
  LinearOctree lOctree(points);
  tw.stop();
  std::cout << "Time to build global (linear) octree: " << tw.getElapsedDecimalSeconds()
            << " seconds\n";
  std::ofstream lOctreeStream(mainOptions.outputDirName / "linear_octree.txt");
  lOctree.writeOctree(lOctreeStream, 0);

  return EXIT_SUCCESS;
}
