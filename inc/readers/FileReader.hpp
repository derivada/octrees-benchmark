//
// Created by miguelyermo on 6/8/21.
//

/*
* FILENAME :  FileReader.h  
* PROJECT  :  rule-based-classifier-cpp
* DESCRIPTION :
*  
*
*
*
*
* AUTHOR :    Miguel Yermo        START DATE : 14:28 6/8/21
*
*/

#pragma once

#include "util.hpp"
#include <filesystem>
#include <iostream>
#include <vector>

namespace fs = std::filesystem;

/**
 * @author Miguel Yermo
 * @brief Abstract class defining common behavor for all file readers
 */
template <PointType Point_t>
class FileReader
{
	protected:
	/**
	 * @brief Path to file to be written
	 */
	fs::path path{};

	public:
	// ***  CONSTRUCTION / DESTRUCTION  *** //
	// ************************************ //

	/**
	 * @brief Instantiate a FileReader which reads a file from a given path
	 * @param path
	 */
	FileReader(const fs::path& path) : path(path){};
	virtual ~FileReader(){}; // Every specialization of this class must manage its own destruction
	virtual std::vector<Point_t> read() = 0;
};