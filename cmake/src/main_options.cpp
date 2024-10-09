//
// Created by miguelyermo on 11/3/20.
//

#include "main_options.hpp"

#include <cmath>
#include <filesystem>

namespace fs = std::filesystem;

main_options mainOptions{};

void printHelp()
{
	std::cout
	    << "-h: Show this message\n"
	       "-i: Path to input file\n"
	       "-o: Path to output file\n";
	exit(1);
}

void setDefaults()
{
	if (mainOptions.outputDirName.empty()) { mainOptions.outputDirName = "out"; }
}


void processArgs(int argc, char** argv)
{
	while (true)
	{
		const auto opt = getopt_long(argc, argv, short_opts, long_opts, nullptr);

		if (-1 == opt) { break; }

		switch (opt)
		{
				// Short Options
			case 'h': {
				printHelp();
				break;
			}
			case 'i': {
				mainOptions.inputFile = fs::path(std::string(optarg));
				std::cout << "Read file set to: " << mainOptions.inputFile << "\n";
				break;
			}
			case 'o': {
				mainOptions.outputDirName = fs::path(std::string(optarg));
				std::cout << "Output path set to: " << mainOptions.outputDirName << "\n";
				break;
			}
			case '?': // Unrecognized option
			default:
				printHelp();
				break;
		}
	}
}