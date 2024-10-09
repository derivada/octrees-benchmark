//
// Created by miguelyermo on 1/3/20.
//

#include "handlers.hpp"
#include "FileReaderFactory.hpp"
#include "TxtFileReader.hpp"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <lasreader.hpp>
#include <random>

void createDirectory(const fs::path& dirName)
/**
 * This function creates a directory if it does not exist.
 * @param dirname
 * @return
 */
{
	if (!fs::is_directory(dirName)) { fs::create_directories(dirName); }
}

void writePoints(fs::path& filename, std::vector<Lpoint>& points)
{
	std::ofstream f(filename);
	f << std::fixed << std::setprecision(2);

	for (Lpoint& p : points)
	{
		f << p << "\n";
	}

	f.close();
}

std::vector<Lpoint> readPointCloud(const fs::path& filename)
{
	// Get Input File extension
	auto fExt = filename.extension();

	FileReader_t readerType = chooseReaderType(fExt);

	// asdf
	if (readerType == err_t)
	{
		std::cout << "Uncompatible file format\n";
		exit(-1);
	}

	std::shared_ptr<FileReader> fileReader = FileReaderFactory::makeReader(readerType, filename);

	std::vector<Lpoint> points = fileReader->read();
	// Decimation. Implemented here because, tbh, I don't want to implement it for each reader type.
	std::cout << "Point cloud size: " << points.size() << "\n";

	return points;
}