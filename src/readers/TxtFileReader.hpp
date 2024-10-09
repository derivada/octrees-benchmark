//
// Created by miguelyermo on 6/8/21.
//

#pragma once

#include "FileReader.hpp"
/**
 * @brief Specialization of FileRead to read .txt/.xyz files
 */
class TxtFileReader : public FileReader
{
	public:
	uint8_t numCols{};

	// ***  CONSTRUCTION / DESTRUCTION  *** //
	// ************************************ //
	TxtFileReader(const fs::path& path) : FileReader(path){};
	~TxtFileReader(){};

	/**
	 * @brief Reads the points contained in the .txt/.xyz file
	 * @return Vector of Lpoint
	 */
	std::vector<Lpoint> read();

	/**
	 * @brief Sets the number of columns of the file to be read
	 * @return Number of columns of the file
	 */
	void setNumberOfColumns(std::ifstream& file);
};

std::vector<std::string> splitLine(std::string& line);