#ifndef CPP_MAIN_OPTIONS_HPP
#define CPP_MAIN_OPTIONS_HPP

#include <filesystem>
#include <getopt.h>
#include <iostream>
#include <vector>
#include "omp.h"
#include <set>
#include "NeighborKernels/KernelFactory.hpp"

namespace fs = std::filesystem;

enum BenchmarkMode { SEARCH, APPROX, PARALLEL, LOG_OCTREE };
enum SearchAlgo { NEIGHBORS_PTR, NEIGHBORS, NEIGHBORS_V2, NEIGHBORS_STRUCT, NEIGHBORS_APPROX };
enum EncoderType { MORTON_ENCODER_3D, HILBERT_ENCODER_3D, NO_ENCODING };

constexpr const char* searchAlgoToString(SearchAlgo algo) {
    switch (algo) {
		case SearchAlgo::NEIGHBORS_PTR: return "neighborsPtr";
        case SearchAlgo::NEIGHBORS: return "neighbors";
        case SearchAlgo::NEIGHBORS_V2: return "neighborsV2";
        case SearchAlgo::NEIGHBORS_STRUCT: return "neighborsStruct";
		case SearchAlgo::NEIGHBORS_APPROX: return "neighbors";
        default: return "Unknown";
    }
}

constexpr const char* encoderTypeToString(EncoderType enc) {
    switch (enc) {
		case EncoderType::NO_ENCODING: return "none";
        case EncoderType::MORTON_ENCODER_3D: return "mort";
        case EncoderType::HILBERT_ENCODER_3D: return "hilb";
        default: return "Unknown";
    }
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
	size_t repeats{2};
	size_t numSearches{10000};
	BenchmarkMode benchmarkMode{SEARCH};
	std::set<Kernel_t> kernels{Kernel_t::sphere, Kernel_t::circle, Kernel_t::cube, Kernel_t::square};
	std::set<SearchAlgo> searchAlgos{ SearchAlgo::NEIGHBORS_PTR, SearchAlgo::NEIGHBORS_V2 };
	std::set<EncoderType> encodings { EncoderType::NO_ENCODING, EncoderType::MORTON_ENCODER_3D, EncoderType::HILBERT_ENCODER_3D };

	bool checkResults{false};
	bool useWarmup{true};
	std::vector<double> approximateTolerances{50.0};
	std::vector<int> numThreads{omp_get_max_threads()};
	bool sequentialSearches{false};
	bool searchAll{false};
	size_t maxPointsLeaf = 128;

};

extern main_options mainOptions;

enum LongOptions : int
{
	HELP = 0,
	RADII,
	REPEATS,
	SEARCHES,
	BENCHMARK,
	KERNELS,
	SEARCH_ALGOS,
	ENCODINGS,

	CHECK,
	NO_WARMUP,
	APPROXIMATE_TOLERANCES,
	NUM_THREADS,
	SEQUENTIAL_SEARCH_SET,
	MAX_POINTS_LEAF
};

// Define short options
const char* const short_opts = "h:i:o:r:s:t:b:k:a:e:cb:";

// Define long options
const option long_opts[] = {
	{ "help", no_argument, nullptr, LongOptions::HELP },
	{ "radii", required_argument, nullptr, LongOptions::RADII },
	{ "repeats", required_argument, nullptr, LongOptions::REPEATS },
	{ "searches", required_argument, nullptr, LongOptions::SEARCHES },
	{ "benchmark", required_argument, nullptr, LongOptions::BENCHMARK },
	{ "kernels", required_argument, nullptr, LongOptions::KERNELS},
	{ "search-algos", required_argument, nullptr, LongOptions::SEARCH_ALGOS },
	{ "encodings", required_argument, nullptr, LongOptions::ENCODINGS },

	{ "check", no_argument, nullptr, LongOptions::CHECK },
	{ "no-warmup", no_argument, nullptr, LongOptions::NO_WARMUP },
	{ "approx-tol", required_argument, nullptr, LongOptions::APPROXIMATE_TOLERANCES },
	{ "num-threads", required_argument, nullptr, LongOptions::NUM_THREADS},
	{ "sequential", no_argument, nullptr, LongOptions::SEQUENTIAL_SEARCH_SET},
	{ "max-leaf", required_argument, nullptr, LongOptions::MAX_POINTS_LEAF},
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
