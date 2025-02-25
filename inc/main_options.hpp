#ifndef CPP_MAIN_OPTIONS_HPP
#define CPP_MAIN_OPTIONS_HPP

#include <filesystem>
#include <getopt.h>
#include <iostream>
#include <vector>

namespace fs = std::filesystem;

enum BenchmarkMode { SEARCH, COMPARE, SEQUENTIAL, POINT_TYPE, APPROX, LOG_OCTREE };

class main_options
{
public:
	// Files & paths
	fs::path inputFile{};
	fs::path outputDirName{};
	std::string inputFileName{};

	// Benchmark parameters
	std::vector<float> benchmarkRadii{2.5, 5.0, 7.5, 10.0};
	size_t repeats{2};
	size_t numSearches{10000};
	bool checkResults{false};
	bool useWarmup{true};
	BenchmarkMode benchmarkMode{SEARCH};
	double approximateTolerance = 50.0;
};

extern main_options mainOptions;

enum LongOptions : int
{
	HELP = 0,
	RADII,
	REPEATS,
	SEARCHES,
	CHECK,
	BENCHMARK,
	NO_WARMUP,
	APPROXIMATE_TOLERANCE
};

// Define short options
const char* const short_opts = "h:i:o:r:t:s:cb:";

// Define long options
const option long_opts[] = {
	{ "help", no_argument, nullptr, LongOptions::HELP },
	{ "radii", required_argument, nullptr, LongOptions::RADII },
	{ "repeats", required_argument, nullptr, LongOptions::REPEATS },
	{ "searches", required_argument, nullptr, LongOptions::SEARCHES },
	{ "check", no_argument, nullptr, LongOptions::CHECK },
	{ "benchmark", required_argument, nullptr, LongOptions::BENCHMARK },
	{ "no-warmup", no_argument, nullptr, LongOptions::NO_WARMUP },
	{ "approx-tol", required_argument, nullptr, LongOptions::APPROXIMATE_TOLERANCE },
	{ nullptr, 0, nullptr, 0 }
};

void printHelp();
void processArgs(int argc, char** argv);
void setDefaults();

#endif // CPP_MAIN_OPTIONS_HPP
