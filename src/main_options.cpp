#include "main_options.hpp"
#include <sstream>
#include <cstdlib>

main_options mainOptions{};

void printHelp()
{
	std::cout
		<< "-h, --help: Show this message\n"
		   "-i: Path to input file\n"
		   "-o: Path to output file\n"
		   "-r, --radii: Benchmark radii (comma-separated, e.g., '2.5,5.0,7.5')\n"
		   "-s, --searches: Number of searches\n"
		   "-t, --repeats: Number of repeats\n"
		   "-c, --check: Enable result checking\n"
		   "-b, --benchmark: Benchmark to run: 'srch' for search (default), 'comp' for comparison, 'seq' for sequential vs shuffled points, 'pt' for point type comparison\n"
		   "    --no-warmup: Disable warmup phase\n";
	exit(1);
}

void setDefaults()
{
	if (mainOptions.outputDirName.empty()) { mainOptions.outputDirName = "out"; }
	mainOptions.benchmarkMode = BenchmarkMode::SEARCH;
}

std::vector<float> parseRadii(const std::string& radiiStr)
{
	std::vector<float> radii;
	std::stringstream ss(radiiStr);
	std::string token;

	while (std::getline(ss, token, ',')) {
		radii.push_back(std::stof(token));
	}

	return radii;
}

void processArgs(int argc, char** argv)
{
	while (true)
	{
		const auto opt = getopt_long(argc, argv, short_opts, long_opts, nullptr);

		if (opt == -1) { break; } // No more options to process

		switch (opt)
		{
			case 'h':
			case LongOptions::HELP:
				printHelp();
				break;

			case 'i':
				mainOptions.inputFile = fs::path(std::string(optarg));
				mainOptions.inputFileName = mainOptions.inputFile.stem().string();
				break;

			case 'o':
				mainOptions.outputDirName = fs::path(std::string(optarg));
				break;

			case 'r':
			case LongOptions::RADII:
				mainOptions.benchmarkRadii = parseRadii(std::string(optarg));
				break;

			case 't':
			case LongOptions::REPEATS:
				mainOptions.repeats = std::stoul(std::string(optarg));
				break;

			case 's':
			case LongOptions::SEARCHES:
				mainOptions.numSearches = std::stoul(std::string(optarg));
				break;

			case 'c':
			case LongOptions::CHECK:
				mainOptions.checkResults = true;
				break;

			case 'b':
			case LongOptions::BENCHMARK:
				if (std::string(optarg) == "srch") {
					mainOptions.benchmarkMode = SEARCH;
				} else if (std::string(optarg) == "comp") {
					mainOptions.benchmarkMode = COMPARE;
				} else if (std::string(optarg) == "seq") {
					mainOptions.benchmarkMode = SEQUENTIAL;
				} else if(std::string(optarg) == "pt") {
					mainOptions.benchmarkMode = POINT_TYPE;
				} else {
					std::cerr << "Invalid benchmark mode: " << optarg << "\n";
					printHelp();
				}
				break;

			case LongOptions::NO_WARMUP:
				mainOptions.useWarmup = false;
				break;

			case '?': // Unrecognized option
			default:
				printHelp();
				break;
		}
	}
}
