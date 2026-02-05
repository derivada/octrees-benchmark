#include <cstdlib>
#include <optional>
#include <set>
#include <sstream>
#include <unordered_map>

#include "encoding/point_encoder_factory.hpp"
#include "kernels/kernel_factory.hpp"
#include "main_options.hpp"

main_options mainOptions{};

void printHelp() {
	std::cout
		<< "Main options:\n"
		<< "-h, --help: Show help message\n"
		<< "-i, --input: Path to input file\n"
		<< "-c, --container-type: Container type to use, possible values. Default: AoS. Possible values: SoA, AoS.\n"
		<< "-o, --output: Path to output file\n"
		<< "-r, --radii: Benchmark radii (comma-separated, e.g., '2.5,5.0,7.5')\n"
		<< "-v, --kvalues: kNN benchmark k's (comma-separated, e.g., '10,50,250,1000')\n"
		<< "-s, --searches: Number of searches (random centers unless --sequential is set), type 'all' to search over the whole cloud (with sequential indexing)\n"
		<< "-t, --repeats: Number of repeats to do for each benchmark\n"
		<< "-k, --kernels: Specify which kernels to use (comma-separated or 'all'). Possible values: sphere, cube, square, circle\n"
		<< "-a, --search-algo: Specify which search algorithms to run (comma-separated or 'all'). Default: neighborsPtr,neighbors,neighborsPrune,neighborsStruct. Possible values:\n"
		<< "    'neighborsPtr'       - basic search on pointer-based octree\n"
		<< "    'neighbors'          - basic search on linear octree\n"
		<< "    'neighborsPrune'     - optimized linear octree search with octant pruning\n"
		<< "    'neighborsStruct'    - optimized linear search using index ranges\n"
		<< "    'neighborsApprox'    - approximate search with upper/lower bounds, requires --approx-tol\n"
		<< "    'neighborsUnibn'     - unibnOctree search\n"
		<< "    'neighborsPCLKD'     - PCL KD-tree search (if available)\n"
		<< "    'neighborsPCLOct'    - PCL Octree search (if available)\n"
		<< "    'neighborsPico'      - PicoTree search\n"
		<< "	'KNNV2' 		     - linear octree KNN searches\n"
		<< "	'KNNNanoflann'		 - nanoflann KNN searches\n"
		<< "	'KNNPCLKD'			 - PCL KD-tree KNN search (if available)\n"
		<< "	'KNNPCLOCT'			 - PCL Octree KNN search (if available)\n"
		<< "	'KNNPico'			 - PicoTree KNN search\n"
		<< "-e, --encodings: Select SFC encodings to reorder the cloud before the searches (comma-separated or 'all'). Default: all. Possible values:\n"
		<< "    'none'  - no encoding; disables Linear Octree building for those runs\n"
		<< "    'mort'  - Morton SFC Reordering\n"
		<< "    'hilb'  - Hilbert SFC Reordering\n\n"

		<< "Other options:\n"
		<< "--debug: Enable debug mode (measures octree build and encoding times)\n"
		<< "--build-enc: Run benchmarks for the encoding and build of selected structures (the ones with a representative on -a / --search-algo)\n"
		<< "--memory: Run a simple benchmark for measuring the memory consumed by an structure, so heap profiling can be easy. Possible values: ptrOct,linOct,unibnOct,nanoKD,pclOct,pclKD,picoTree\n"
		<< "--locality: Run benchmarks for the analyzing the locality of the point cloud after given reorderings\n"
		<< "--cache-profiling: Enable cache profiling during search algo. executions using PAPI\n"
		<< "--check: Enable result checking (legacy option; use avg_result_size to verify correctness)\n"
		<< "--no-warmup: Disable warmup phase\n"
		<< "--approx-tol: Tolerance values for approximate search (comma-separated e.g., '10.0,50.0,100.0')\n"
		<< "--num-threads: List of thread counts for scalability test (comma-separated e.g., '1,2,4,8,16,32')\n"
		<< "               If not specified, OpenMP defaults to maximum threads and no scalability test is run\n"
		<< "--sequential: Make the search set sequential instead of random (usually faster). Automatically set when -s all is used\n"
		<< "--max-leaf: Max number of points per octree leaf (default = 128). Does not apply to PCL Octree\n"
		<< "--pcl-oct-resolution: Min octant size for subdivision in PCL Octree\n";
	exit(1);
}

SearchStructure structureFromString(std::string_view str) {
    for (const auto& [key, val] : structureMap) {
        if (val == str) return key;
    }
    throw std::invalid_argument("Unknown search algorithm: " + std::string(str));
}

SearchAlgo searchAlgoFromString(std::string_view str) {
    for (const auto& [key, val] : searchAlgoMap) {
        if (val == str) return key;
    }
    throw std::invalid_argument("Unknown search algorithm: " + std::string(str));
}

EncoderType encoderTypeFromString(std::string_view str) {
    for (const auto& [key, val] : encoderTypeMap) {
        if (val == str) return key;
    }
    throw std::invalid_argument("Unknown encoder type: " + std::string(str));
}

Kernel_t kernelFromString(std::string_view str) {
    for (const auto& [key, val] : kernelMap) {
        if (val == str) return key;
    }
    throw std::invalid_argument("Unknown kernel type: " + std::string(str));
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


std::set<SearchAlgo> parseSearchAlgoOptions(const std::string& algoStr) {
    std::set<SearchAlgo> selectedSearchAlgos;

    if (algoStr == "all") {
		for (const auto& [algo, _] : searchAlgoMap) {
			selectedSearchAlgos.insert(algo);
		}
    } else {
        std::stringstream ss(algoStr);
        std::string token;
        while (std::getline(ss, token, ',')) {
            try {
                selectedSearchAlgos.insert(searchAlgoFromString(token));
            } catch (const std::invalid_argument& e) {
                std::cerr << e.what() << "\n";
            }
        }
    }

#ifndef HAVE_PCL
    if (selectedSearchAlgos.count(NEIGHBORS_PCLKD) ||
        selectedSearchAlgos.count(NEIGHBORS_PCLOCT)) {
        std::cerr << "Error: PCL-based search algorithms selected, but HAVE_PCL is not defined.\n";
        std::exit(EXIT_FAILURE);
    }
#endif

    return selectedSearchAlgos;
}

std::set<EncoderType> parseEncodingOptions(const std::string& encoderStr) {
    std::set<EncoderType> selectedEncoders;

    if (encoderStr == "all") {
		for (const auto& [enc, _] : encoderTypeMap) {
			selectedEncoders.insert(enc);
		}
    } else {
        std::stringstream ss(encoderStr);
        std::string token;
        while (std::getline(ss, token, ',')) {
            try {
                selectedEncoders.insert(encoderTypeFromString(token));
            } catch (const std::invalid_argument& e) {
                std::cerr << e.what() << "\n";
            }
        }
    }

    return selectedEncoders;
}

std::set<Kernel_t> parseKernelOptions(const std::string& kernelStr) {
    std::set<Kernel_t> selectedKernels;

    if (kernelStr == "all") {
		for (const auto& [kernel, _] : kernelMap) {
			selectedKernels.insert(kernel);
		}
    } else {
        std::stringstream ss(kernelStr);
        std::string token;
        while (std::getline(ss, token, ',')) {
            try {
                selectedKernels.insert(kernelFromString(token));
            } catch (const std::invalid_argument& e) {
                std::cerr << e.what() << "\n";
            }
        }
    }

    return selectedKernels;
}

std::set<SearchStructure> getSearchStructures(std::set<SearchAlgo> &algos) {
	std::set<SearchStructure> structures;

	for (SearchAlgo algo : algos) {
		structures.insert(algoToStructure(static_cast<SearchAlgo>(algo)));
	}

	return structures;
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
			case LongOptions::INPUT:
				mainOptions.inputFile = fs::path(std::string(optarg));
				mainOptions.inputFileName = mainOptions.inputFile.stem().string();
				break;
			case 'c':
			case LongOptions::CONTAINER_TYPE: {
				std::string containerTypeStr = optarg;
				// lowercase it
				std::transform(containerTypeStr.begin(), containerTypeStr.end(), containerTypeStr.begin(),
	               [](unsigned char c) { return std::tolower(c); });
				
				if (containerTypeStr == "aos") {
					mainOptions.containerType = ContainerType::AoS;
				} else if (containerTypeStr == "soa") {
					mainOptions.containerType = ContainerType::SoA;
				} else {
					std::cerr << "Invalid container type: " << containerTypeStr << ". Use 'AoS' or 'SoA'.\n";
					std::exit(EXIT_FAILURE);
				}				
				break;
			}
			case 'o':
			case LongOptions::OUTPUT:
				mainOptions.outputDirName = fs::path(std::string(optarg));
				break;
			case 'r':
			case LongOptions::RADII:
				mainOptions.benchmarkRadii = readVectorArg<float>(std::string(optarg));
				break;
			case 'v':
			case LongOptions::K_VALUES:
				mainOptions.benchmarkKValues = readVectorArg<size_t>(std::string(optarg));
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
				mainOptions.searchStructures = getSearchStructures(mainOptions.searchAlgos);
				break;
			case 'e':
			case LongOptions::ENCODINGS:
				mainOptions.encodings = parseEncodingOptions(std::string(optarg));
				break;
			case LongOptions::DEBUG:
				mainOptions.debug = true;
				break;
			case LongOptions::BUILD_ENC:
				mainOptions.buildEncBenchmarks = true;
				break;
			case LongOptions::MEMORY:
				mainOptions.memoryStructure.emplace(structureFromString(std::string(optarg)));
			case LongOptions::LOCALITY:
				mainOptions.localityBenchmarks = true;
				break;
			case LongOptions::CACHE_PROFILING:
				mainOptions.cacheProfiling = true;
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
			case LongOptions::PCL_OCT_RESOLUTION:
				mainOptions.pclOctResolution = std::stod(std::string(optarg));
				break;
			default:
				printHelp();
				break;
		}
	}
}