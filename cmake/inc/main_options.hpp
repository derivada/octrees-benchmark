#ifndef CPP_MAIN_OPTIONS_HPP
#define CPP_MAIN_OPTIONS_HPP

//
// Created by miguelyermo on 11/3/20.
//

/*
* FILENAME :  main_options.h  
* PROJECT  :  rule-based-classifier-cpp
*
* DESCRIPTION :
*
*
*
*
*
* AUTHOR :    Miguel Yermo        START DATE : 18:50 11/3/20
*
*/

#include <filesystem>
#include <getopt.h>
#include <iostream>

namespace fs = std::filesystem;

class main_options
{
	public:
	// Files & paths
	fs::path      inputFile{};
	fs::path      outputDirName{};
};

extern main_options mainOptions;

enum LongOptions : int
{
	HELP = 0, // Help message
};

// Define short options
const char* const short_opts = "h:i:o:";

// Define long options
const option long_opts[] = {
	{ "help", no_argument, nullptr, LongOptions::HELP },
};

void printHelp();
void processArgs(int argc, char** argv);
void setDefaults();

#endif //CPP_MAIN_OPTIONS_HPP
