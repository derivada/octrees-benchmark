//
// Created by miguelyermo on 6/8/21.
//

#pragma once

#include "FileReader.hpp"
#include <iterator>
#include <fstream>
#include <functional>
/**
 * @brief Specialization of FileRead to read .txt/.xyz files
 */
template <PointType Point_t>
class TxtFileReader : public FileReader<Point_t>
{
	public:
	uint8_t numCols{};

	// ***  CONSTRUCTION / DESTRUCTION  *** //
	// ************************************ //
	TxtFileReader(const fs::path& path) : FileReader<Point_t>(path){};
	~TxtFileReader(){};

	std::vector<std::string> splitLine(std::string& line)
	{
		std::istringstream                 buf(line);
		std::istream_iterator<std::string> beg(buf), end;
		std::vector<std::string>           tokens(beg, end);

		return tokens;
	}

	/**
	 * @brief Sets the number of columns of the file to be read
	 * @return Number of columns of the file
	 */
	void setNumberOfColumns(std::ifstream& file)
	{
		std::string line, item;

		std::getline(file, line);
		file.seekg(0); // Return to first line to be read later.
		std::stringstream ss(line);

		numCols = 0;
		while (ss >> item)
			numCols++;
	}


	/**
	 * @brief Reads the points contained in the .txt/.xyz file
	 * @return Vector of point_t
	 */
	
	std::vector<Point_t> read()
	{
		std::ifstream file(this->path.string());
		std::string   line{};

		setNumberOfColumns(file);

		unsigned int        idx = 0;
		std::vector<Point_t> points;

		// TODO: Pensar como modularizarlo...
		// TODO: Factory as a function of the number of columns to read different inputs!
		auto terminationCondition = [&file, &line] { 
			if (std::getline(file, line)) {
				return true;
			} else {
				return false;
			}
		 };

		auto pointInserter = [&](size_t& idx, std::vector<Point_t>& points) { 
			auto tokens = splitLine(line);
			switch (numCols) {
				case 3:
					points.emplace_back(idx,  	// id
						std::stod(tokens[0]),  	// x
						std::stod(tokens[1]),  	// y
						std::stod(tokens[2])); 	// z
				break;
				case 7:
					points.emplace_back(idx,
						std::stod(tokens[0]),  	// x
						std::stod(tokens[1]),  	// y
						std::stod(tokens[2]),  	// z
						std::stod(tokens[3]),  	// I
						std::stod(tokens[4]),  	// rn
						std::stod(tokens[5]),  	// nor
						std::stoi(tokens[6])   	// classification
					);
				break;
				case 9:
					points.emplace_back(idx,
						std::stod(tokens[0]),  	// x
						std::stod(tokens[1]),  	// y
						std::stod(tokens[2]),  	// z
						std::stod(tokens[3]),  	// I
						std::stoi(tokens[4]),  	// rn
						std::stoi(tokens[5]),  	// nor
						std::stoi(tokens[6]),  	// dir
						std::stoi(tokens[7]),  	// edge
						std::stoi(tokens[8])   	// classification
					);
				break;
				case 12:
					points.emplace_back(idx,
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
						std::stoi(tokens[11])   // b
					);
				break;
				default:
					std::cout << "Unrecognized format\n";
					exit(1);
			};
		};

		this->file_reading_loop(points, terminationCondition, pointInserter, -1, true);

		file.close();
		std::cout << "Read points: " << idx << "\n";
		return points;
	};
};

