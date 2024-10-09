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

#include "main_options.hpp"
#include "point.hpp"
#include "Lpoint.hpp"
#include <filesystem> // File extensions
#include <string>
#include <vector>

namespace fs = std::filesystem;

void handleNumberOfPoints(std::vector<Lpoint>& points);

unsigned int getNumberOfCols(const fs::path& filePath);

void createDirectory(const fs::path& dirname);

std::vector<Lpoint> readPointCloud(const fs::path& fileName);


#endif //CPP_HANDLERS_H
