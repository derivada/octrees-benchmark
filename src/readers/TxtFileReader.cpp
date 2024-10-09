//
// Created by miguelyermo on 6/8/21.
//

#include "TxtFileReader.hpp"
#include "main_options.hpp"
#include <fstream>
#include <iterator>

std::vector<std::string> splitLine(std::string& line)
{
	std::istringstream                 buf(line);
	std::istream_iterator<std::string> beg(buf), end;
	std::vector<std::string>           tokens(beg, end);

	return tokens;
}

void TxtFileReader::setNumberOfColumns(std::ifstream& file)
{
	std::string line, item;

	std::getline(file, line);
	file.seekg(0); // Return to first line to be read later.
	std::stringstream ss(line);

	numCols = 0;
	while (ss >> item)
		numCols++;
}

std::vector<Lpoint> TxtFileReader::read()
{
	std::ifstream file(path.string());
	std::string   line{};

	setNumberOfColumns(file);

	unsigned int        idx = 0;
	std::vector<Lpoint> points;

	// TODO: Pensar como modularizarlo...
	// TODO: Factory as a function of the number of columns to read different inputs!
	switch (numCols)
	{
		case 3:
			while (std::getline(file, line, '\n'))
			{
				auto tokens = splitLine(line);
				points.emplace_back(idx++,                 // id
				                    std::stod(tokens[0]),  // x
				                    std::stod(tokens[1]),  // y
				                    std::stod(tokens[2])); // z
			}
			break;

		case 7:
			while (std::getline(file, line, '\n'))
			{
				auto tokens = splitLine(line);
				points.emplace_back(idx++,                 // id
				                    std::stod(tokens[0]),  // x
				                    std::stod(tokens[1]),  // y
				                    std::stod(tokens[2]),  // z
				                    std::stod(tokens[3]),  // I
				                    std::stod(tokens[4]),  // rn
				                    std::stod(tokens[5]),  // nor
				                    std::stoi(tokens[6])); // classification
			}
			break;

		// Raw point cloud without RGB
		case 9:
			while (std::getline(file, line, '\n'))
			{
				auto tokens = splitLine(line);
				points.emplace_back(idx++,                 // id
				                    std::stod(tokens[0]),  // x
				                    std::stod(tokens[1]),  // y
				                    std::stod(tokens[2]),  // z
				                    std::stod(tokens[3]),  // I
				                    std::stoi(tokens[4]),  // rn
				                    std::stoi(tokens[5]),  // nor
				                    std::stoi(tokens[6]),  // dir
				                    std::stoi(tokens[7]),  // edge
				                    std::stoi(tokens[8])); // classification
			}
			break;

		case 12:
			while (std::getline(file, line, '\n'))
			{
				auto tokens = splitLine(line);
				points.emplace_back(idx++,                  // id
				                    std::stod(tokens[0]),   // x
				                    std::stod(tokens[1]),   // y
				                    std::stod(tokens[2]),   // z
				                    std::stod(tokens[3]),   // I
				                    std::stoi(tokens[4]),   // rn
				                    std::stoi(tokens[5]),   // nor
				                    std::stoi(tokens[6]),   // dir
				                    std::stoi(tokens[7]),   // edge
				                    std::stoi(tokens[8]),   // classification
				                    std::stoi(tokens[9]),   // r
				                    std::stoi(tokens[10]),  // g
				                    std::stoi(tokens[11])); // b
			}
			break;

		default:
			std::cout << "Unrecognized format\n";
			exit(1);
	}
	file.close();
	std::cout << "Read points: " << idx << "\n";
	return points;
};