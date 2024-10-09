//
// Created by miguelyermo on 6/8/21.
//

#include "FileReaderFactory.hpp"

FileReader_t chooseReaderType(const std::string& fExt)
{
	if (fExt == ".las" || fExt == ".laz") return las_t;
	if (fExt == ".txt" || fExt == ".xyz") return txt_t;

	return err_t;
}