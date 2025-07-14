#ifndef CPP_MAIN_OPTIONS_HPP
#define CPP_MAIN_OPTIONS_HPP

#include <filesystem>
#include <getopt.h>
#include <iostream>
#include <vector>
#include "omp.h"
#include <set>

namespace fs = std::filesystem;

enum SearchStructure { 
	PTR_OCTREE, LINEAR_OCTREE, UNIBN_OCTREE, 
	PCL_OCTREE, PCL_KDTREE, NANOFLANN_KDTREE 
};

constexpr std::pair<SearchStructure, std::string_view> structureMap[] = {
    { SearchStructure::PTR_OCTREE,       "Octree" },
    { SearchStructure::LINEAR_OCTREE,    "LinearOctree" },
    { SearchStructure::UNIBN_OCTREE,     "UnibnOctree" },
    { SearchStructure::PCL_OCTREE,       "PCLOctree" },
    { SearchStructure::PCL_KDTREE,       "PCLKDTree" },
    { SearchStructure::NANOFLANN_KDTREE, "NanoflannKDTree" }
};

enum SearchAlgo { 
	NEIGHBORS_PTR, NEIGHBORS, NEIGHBORS_PRUNE,
	NEIGHBORS_STRUCT, NEIGHBORS_APPROX, NEIGHBORS_UNIBN, 
	NEIGHBORS_PCLKD, NEIGHBORS_PCLOCT, NEIGHBORS_NANOFLANN, 
	KNN_V2, KNN_NANOFLANN, KNN_PCLKD, KNN_PCLOCT
};

constexpr std::pair<SearchAlgo, std::string_view> searchAlgoMap[] = {
	{ SearchAlgo::NEIGHBORS_PTR, 		"neighborsPtr" },
	{ SearchAlgo::NEIGHBORS, 			"neighbors" },
	{ SearchAlgo::NEIGHBORS_PRUNE, 		"neighborsPrune" },
	{ SearchAlgo::NEIGHBORS_STRUCT, 	"neighborsStruct" },
	{ SearchAlgo::NEIGHBORS_APPROX, 	"neighborsApprox" },
	{ SearchAlgo::NEIGHBORS_UNIBN, 		"neighborsUnibn" },
	{ SearchAlgo::NEIGHBORS_PCLKD, 		"neighborsPCLKD" },
	{ SearchAlgo::NEIGHBORS_PCLOCT, 	"neighborsPCLOct" },
	{ SearchAlgo::NEIGHBORS_NANOFLANN, 	"neighborsNanoflann" },
	{ SearchAlgo::KNN_V2, 				"KNNV2" },
	{ SearchAlgo::KNN_NANOFLANN, 		"KNNNanoflann" },
	{ SearchAlgo::KNN_PCLKD,			"KNNPCLKD" },
	{ SearchAlgo::KNN_PCLOCT,			"KNNPCLOCT" }

};

constexpr std::pair<SearchAlgo, SearchStructure> algoToStructureMap[] = {
    { SearchAlgo::NEIGHBORS_PTR,      	SearchStructure::PTR_OCTREE },
    { SearchAlgo::NEIGHBORS,          	SearchStructure::LINEAR_OCTREE },
    { SearchAlgo::NEIGHBORS_PRUNE,    	SearchStructure::LINEAR_OCTREE },
    { SearchAlgo::NEIGHBORS_STRUCT,   	SearchStructure::LINEAR_OCTREE },
    { SearchAlgo::NEIGHBORS_APPROX,   	SearchStructure::LINEAR_OCTREE },
    { SearchAlgo::NEIGHBORS_UNIBN,    	SearchStructure::UNIBN_OCTREE },
    { SearchAlgo::NEIGHBORS_PCLKD,    	SearchStructure::PCL_KDTREE },
    { SearchAlgo::NEIGHBORS_PCLOCT,   	SearchStructure::PCL_OCTREE },
    { SearchAlgo::NEIGHBORS_NANOFLANN,	SearchStructure::NANOFLANN_KDTREE },
    { SearchAlgo::KNN_V2,             	SearchStructure::LINEAR_OCTREE },
	{ SearchAlgo::KNN_NANOFLANN,		SearchStructure::NANOFLANN_KDTREE },
	{ SearchAlgo::KNN_PCLKD,			SearchStructure::PCL_KDTREE },
	{ SearchAlgo::KNN_PCLOCT,			SearchStructure::PCL_OCTREE }
};

enum EncoderType { MORTON_ENCODER_3D, HILBERT_ENCODER_3D, NO_ENCODING };

constexpr std::pair<EncoderType, std::string_view> encoderTypeMap[] = {
    { EncoderType::NO_ENCODING,       	"none" },
    { EncoderType::MORTON_ENCODER_3D,   "mort" },
    { EncoderType::HILBERT_ENCODER_3D,  "hilb" }
};

enum class Kernel_t
{
	circle,
	sphere,
	square,
	cube
};

constexpr std::pair<Kernel_t, std::string_view> kernelMap[] = {
	{Kernel_t::sphere, "sphere"},
	{Kernel_t::circle, "circle"},
	{Kernel_t::cube, "cube"},
	{Kernel_t::square, "square"}
};

constexpr SearchStructure algoToStructure(SearchAlgo algo) {
    for (const auto& [key, val] : algoToStructureMap) {
        if (key == algo) return val;
    }
    return SearchStructure::PTR_OCTREE; // Default fallback
}

constexpr std::string_view searchStructureToString(SearchStructure structure) {
    for (const auto& [key, val] : structureMap) {
        if (key == structure) return val;
    }
    return "Unknown";
}

constexpr std::string_view searchAlgoToString(SearchAlgo algo) {
    for (const auto& [key, val] : searchAlgoMap) {
        if (key == algo) return val;
    }
    return "Unknown";
}

constexpr std::string_view encoderTypeToString(EncoderType enc) {
    for (const auto& [key, val] : encoderTypeMap) {
        if (key == enc) return val;
    }
    return "Unknown";
}

constexpr std::string_view kernelToString(Kernel_t kernel) {
    for (const auto& [key, val] : kernelMap) {
        if (key == kernel) return val;
    }
    return "Unknown";
}

class main_options
{
public:
	// Files & paths
	fs::path inputFile{};
	fs::path outputDirName{"out"};
	std::string inputFileName{};

	// Benchmark parameters
	std::vector<float> benchmarkRadii{2.5, 5.0, 7.5, 10.0};
	std::vector<size_t> benchmarkKValues{10, 50, 250, 1000};

	size_t repeats{2};
	size_t numSearches{10000};
	
	std::set<Kernel_t> kernels{Kernel_t::sphere, Kernel_t::circle, Kernel_t::cube, Kernel_t::square};
	std::set<SearchAlgo> searchAlgos{SearchAlgo::NEIGHBORS_PTR, SearchAlgo::NEIGHBORS, SearchAlgo::NEIGHBORS_PRUNE, SearchAlgo::NEIGHBORS_STRUCT};
	std::set<SearchStructure> searchStructures{SearchStructure::PTR_OCTREE, SearchStructure::LINEAR_OCTREE};
	std::set<EncoderType> encodings{EncoderType::NO_ENCODING, EncoderType::MORTON_ENCODER_3D, EncoderType::HILBERT_ENCODER_3D};

	bool debug{false};
	bool buildEncBenchmarks{false};
	bool cacheProfiling{false};
	bool checkResults{false};
	bool useWarmup{true};
	std::vector<double> approximateTolerances{50.0};
	std::vector<int> numThreads{omp_get_max_threads()};
	bool sequentialSearches{false};
	bool searchAll{false};
	size_t maxPointsLeaf = 128;
	double pclOctResolution = 0.1;

};

extern main_options mainOptions;

enum LongOptions : int
{
	HELP,
	INPUT,
	OUTPUT,
	RADII,
	K_VALUES,
	REPEATS,
	SEARCHES,
	KERNELS,
	SEARCH_ALGOS,
	ENCODINGS,
	
	DEBUG,
	BUILD_ENC,
	CACHE_PROFILING,
	CHECK,
	NO_WARMUP,
	APPROXIMATE_TOLERANCES,
	NUM_THREADS,
	SEQUENTIAL_SEARCH_SET,
	MAX_POINTS_LEAF,
	PCL_OCT_RESOLUTION
};

// Define short options
const char* const short_opts = "h:i:o:r:v:s:t:b:k:a:e:cb:";

// Define long options
const option long_opts[] = {
	{ "help", no_argument, nullptr, LongOptions::HELP },
	{ "input", required_argument, nullptr, LongOptions::INPUT },
	{ "output", required_argument, nullptr, LongOptions::OUTPUT },
	{ "radii", required_argument, nullptr, LongOptions::RADII },
	{ "kvalues", required_argument, nullptr, LongOptions::K_VALUES },
	{ "repeats", required_argument, nullptr, LongOptions::REPEATS },
	{ "searches", required_argument, nullptr, LongOptions::SEARCHES },
	{ "kernels", required_argument, nullptr, LongOptions::KERNELS},
	{ "search-algos", required_argument, nullptr, LongOptions::SEARCH_ALGOS },
	{ "encodings", required_argument, nullptr, LongOptions::ENCODINGS },

	{ "debug", no_argument, nullptr, LongOptions::DEBUG },
	{ "build-enc", no_argument, nullptr, LongOptions::BUILD_ENC },
	{ "cache-profiling", no_argument, nullptr, LongOptions::CACHE_PROFILING },
	{ "check", no_argument, nullptr, LongOptions::CHECK },
	{ "no-warmup", no_argument, nullptr, LongOptions::NO_WARMUP },
	{ "approx-tol", required_argument, nullptr, LongOptions::APPROXIMATE_TOLERANCES },
	{ "num-threads", required_argument, nullptr, LongOptions::NUM_THREADS },
	{ "sequential", no_argument, nullptr, LongOptions::SEQUENTIAL_SEARCH_SET },
	{ "max-leaf", required_argument, nullptr, LongOptions::MAX_POINTS_LEAF },
	{ "pcl-oct-resolution", required_argument, nullptr, LongOptions::PCL_OCT_RESOLUTION },
	{ nullptr, 0, nullptr, 0 }
};

void printHelp();
void setDefaults();
std::set<Kernel_t> parseKernelOptions(const std::string& kernelStr);
std::set<SearchAlgo> parseSearchAlgoOptions(const std::string& kernelStr);
std::set<EncoderType> parseEncodingOptions(const std::string& kernelStr);
std::string getKernelListString();
std::string getSearchAlgoListString();
std::string getEncoderListString();
void processArgs(int argc, char** argv);


#endif // CPP_MAIN_OPTIONS_HPP
