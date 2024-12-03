//
// Created by miguelyermo on 1/3/20.
//

/*
* FILENAME :  handlers.h  
* PROJECT  :  rule-based-classifier-cpp
* DESCRIPTION :
*  
*
*
*
*
* AUTHOR :    Miguel Yermo        START DATE : 03:07 1/3/20
*
*/

#ifndef CPP_HANDLERS_H
#define CPP_HANDLERS_H

#include "readers/FileReaderFactory.hpp"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <lasreader.hpp>
#include <random>

namespace fs = std::filesystem;

template <typename Point_t>
void handleNumberOfPoints(std::vector<Point_t>& points);
unsigned int getNumberOfCols(const fs::path& filePath);

void createDirectory(const fs::path& dirName)
/**
 * This function creates a directory if it does not exist.
 * @param dirname
 * @return
 */
{
	if (!fs::is_directory(dirName)) { fs::create_directories(dirName); }
}

template <typename Point_t>
void writePoints(fs::path& filename, std::vector<Point_t>& points)
{
	std::ofstream f(filename);
	f << std::fixed << std::setprecision(2);

	for (Point_t& p : points)
	{
		f << p << "\n";
	}

	f.close();
}

template <typename Point_t>
std::vector<Point_t> readPointCloud(const fs::path& fileName)
{
	// Get Input File extension
	auto fExt = fileName.extension();

	FileReader_t readerType = chooseReaderType(fExt);

	// asdf
	if (readerType == err_t)
	{
		std::cout << "Uncompatible file format\n";
		exit(-1);
	}

	std::shared_ptr<FileReader<Point_t>> fileReader = FileReaderFactory::makeReader<Point_t>(readerType, fileName);

	std::vector<Point_t> points = fileReader->read();
	// Decimation. Implemented here because, tbh, I don't want to implement it for each reader type.
	std::cout << "Point cloud size: " << points.size() << std::endl;

	return points;
}

#endif //CPP_HANDLERS_H
