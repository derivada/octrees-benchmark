#include "main_options.hpp"
#include <sstream>
#include <cstdlib>
#include <set>
#include "NeighborKernels/KernelFactory.hpp"
#include <unordered_map>
#include "PointEncoding/point_encoder_factory.hpp"
main_options mainOptions{};

void printHelp()
{
	std::cout
		<< "Main options:\n"
			"-h, --help: Show this message\n"
			"-i: Path to input file\n"
			"-o: Path to output file\n"
			"-r, --radii: Benchmark radii (comma-separated, e.g., '2.5,5.0,7.5')\n"
			"-s, --searches: Number of searches\n"
			"-t, --repeats: Number of repeats\n"
			"-k, --kernels: Specify which kernels to use (comma-separated or all). possible values are sphere, cube, square and circle\n"
			"-a, --search-algo: Specify which search algorithms to run on the linear octree (comma-separated or 'all'), default='neighborsPtr,neighborsV2', possible values:\n\t"
				"'neighborsPtr' basic search algorihtm on pointer-based octree,\n\t"
				"'neighbors' basic search algorithm on linear octree\n\t"
				"'neighborsV2' optimized search algorithm on linear octree, uses octant inside kernel check and bulk insert\n\t"
				"'neighborsStruct' optimized search algorithm on linear octree, uses struct of range of indexes for result\n\t"
				"'neighborsApprox' optimized search algorithm on linear octree, uses approximate searches\n"
			"-e, --encodings: Specify which encodings (Reordering SFCs) to use (comma-separated or 'all'), default=all, possible values:\n\t"
				"'none' run pointer-based octree algos selected (i.e. neighborsPtr) without encoding\n\t"
				"'mort' run both octrees with their selected algos with Morton SFC Reordering\n\t"
				"'hilb' run both octrees with their selected algos with Hilbert SFC Reordering\n\n"
			
			"Other options:\n"
			"--debug: Enable debug mode\n"
			"--check: Enable result checking\n"
			"--no-warmup: Disable warmup phase\n"
			"--approx-tol: For specifying tolerance percentage in approximate searches (e.g. 80.0 = 80% tolerance on kernel size), format is list of doubles in format e.g. '10.0,50.0,100.0'\n"
			"--num-threads: List of number of threads to use in the parallelism scalability benchmark (e.g. 1,2,4,8,16,32)\n"
			"--sequential: Make the search set sequential instead of random\n"
			"--max-leaf: Maximum numbers of points in an octree leaf (default = 128)\n";
	exit(1);
}

template <typename T>
std::vector<T> readVectorArg(const std::string& vStr)
{
	std::vector<T> v;
	std::stringstream ss(vStr);
	std::string token;

	while (std::getline(ss, token, ',')) {
		v.push_back(std::stof(token));
	}

	return v;
}

std::set<Kernel_t> parseKernelOptions(const std::string& kernelStr) {
    static const std::unordered_map<std::string, Kernel_t> kernelMap = {
        {"sphere", Kernel_t::sphere},
        {"circle", Kernel_t::circle},
        {"cube", Kernel_t::cube},
        {"square", Kernel_t::square}
    };

    std::set<Kernel_t> selectedKernels;

    if (kernelStr == "all") {
        for (const auto& [key, value] : kernelMap) {
            selectedKernels.insert(value);
        }
    } else {
        std::stringstream ss(kernelStr);
        std::string token;
        while (std::getline(ss, token, ',')) {
            auto it = kernelMap.find(token);
            if (it != kernelMap.end()) {
                selectedKernels.insert(it->second);
            } else {
                std::cerr << "Warning: Unknown kernel '" << token << "' ignored.\n";
            }
        }
    }

    return selectedKernels;
}

std::set<SearchAlgo> parseSearchAlgoOptions(const std::string& algoStr) {
    static const std::unordered_map<std::string, SearchAlgo> algoMap = {
		{"neighborsPtr", SearchAlgo::NEIGHBORS_PTR},
        {"neighbors", SearchAlgo::NEIGHBORS},
        {"neighborsV2", SearchAlgo::NEIGHBORS_V2},
        {"neighborsStruct", SearchAlgo::NEIGHBORS_STRUCT},
		{"neighborsApprox", SearchAlgo::NEIGHBORS_APPROX}
    };

    std::set<SearchAlgo> selectedSearchAlgos;

    if (algoStr == "all") {
        for (const auto& [key, value] : algoMap) {
            selectedSearchAlgos.insert(value);
        }
    } else {
        std::stringstream ss(algoStr);
        std::string token;
        while (std::getline(ss, token, ',')) {
            auto it = algoMap.find(token);
            if (it != algoMap.end()) {
                selectedSearchAlgos.insert(it->second);
            } else {
                std::cerr << "Warning: Unknown search algorithm '" << token << "' ignored.\n";
            }
        }
    }
    return selectedSearchAlgos;
}

std::set<EncoderType> parseEncodingOptions(const std::string& kernelStr) {
    static const std::unordered_map<std::string, EncoderType> encoderMap = {
        {"none", EncoderType::NO_ENCODING},
        {"mort", EncoderType::MORTON_ENCODER_3D},
        {"hilb", EncoderType::HILBERT_ENCODER_3D}
    };

    std::set<EncoderType> selectedEncoders;

    if (kernelStr == "all") {
        for (const auto& [key, value] : encoderMap) {
            selectedEncoders.insert(value);
        }
    } else {
        std::stringstream ss(kernelStr);
        std::string token;
        while (std::getline(ss, token, ',')) {
            auto it = encoderMap.find(token);
            if (it != encoderMap.end()) {
                selectedEncoders.insert(it->second);
            } else {
                std::cerr << "Warning: Unknown kernel '" << token << "' ignored.\n";
            }
        }
    }

    return selectedEncoders;
}

std::string getKernelListString() {
    std::ostringstream oss;
    auto it = mainOptions.kernels.begin();
    for (; it != mainOptions.kernels.end(); ++it) {
        oss << kernelToString(*it);
        if (std::next(it) != mainOptions.kernels.end()) {
            oss << ", ";
        }
    }
    return oss.str();
}

std::string getSearchAlgoListString() {
    std::ostringstream oss;
    auto it = mainOptions.searchAlgos.begin();
    for (; it != mainOptions.searchAlgos.end(); ++it) {
        oss << searchAlgoToString(*it);
        if (std::next(it) != mainOptions.searchAlgos.end()) {
            oss << ", ";
        }
    }
    return oss.str();
}

std::string getEncoderListString() {
    std::ostringstream oss;
    auto it = mainOptions.encodings.begin();
    for (; it != mainOptions.encodings.end(); ++it) {
        oss << encoderTypeToString(*it);
        if (std::next(it) != mainOptions.encodings.end()) {
            oss << ", ";
        }
    }
    return oss.str();
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
				mainOptions.benchmarkRadii = readVectorArg<float>(std::string(optarg));
				break;
			case 't':
			case LongOptions::REPEATS:
				mainOptions.repeats = std::stoul(std::string(optarg));
				break;
			case 's':
			case LongOptions::SEARCHES:
				if (std::string(optarg) == "all") {
					mainOptions.searchAll = true;
					mainOptions.numSearches = 0;
				} else {
					mainOptions.numSearches = std::stoul(std::string(optarg));
				}
				break;
			case 'k':
			case LongOptions::KERNELS:
				mainOptions.kernels = parseKernelOptions(std::string(optarg));
				break;
			case 'a':
			case LongOptions::SEARCH_ALGOS:
				mainOptions.searchAlgos = parseSearchAlgoOptions(std::string(optarg));
				break;
			case 'e':
			case LongOptions::ENCODINGS:
				mainOptions.encodings = parseEncodingOptions(std::string(optarg));
				break;
			
			case LongOptions::DEBUG:
				mainOptions.debug = true;
				break;
			case LongOptions::CHECK:
				mainOptions.checkResults = true;
				break;
			case LongOptions::NO_WARMUP:
				mainOptions.useWarmup = false;
				break;
			case LongOptions::APPROXIMATE_TOLERANCES:
				mainOptions.approximateTolerances = readVectorArg<double>(std::string(optarg));
				break;
			case LongOptions::NUM_THREADS:
				mainOptions.numThreads = readVectorArg<int>(std::string(optarg));
				break;
			case LongOptions::SEQUENTIAL_SEARCH_SET:
				mainOptions.sequentialSearches = true;
				break;

			case LongOptions::MAX_POINTS_LEAF:
				mainOptions.maxPointsLeaf = std::stoul(std::string(optarg));
				break;
			
			default:
				printHelp();
				break;
		}
	}
}