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
#include "Geometry/PointMetadata.hpp"

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
	

	template<typename TerminationCondition, typename PointInserter>
	void file_reading_loop(
		TerminationCondition&& terminationCondition, 
		PointInserter&& pointInserter,
		size_t total_points = -1,
		bool show_progress = false
	) {
		size_t percent_threeshold = total_points != -1 ? total_points / 100 : 0;
		size_t idx = 0, current_count = 0, current_percent = 0;

		while (terminationCondition()) {
			// Perform the loop action (point creation)
			pointInserter(idx);
			idx++, current_count++;
			if(show_progress) {
				if(total_points != -1 && current_count > percent_threeshold) {
					current_count = 0;
					current_percent++;
					progressBar(current_percent, total_points);
				} else if(total_points == -1 && idx % 100000 == 0) { // every 100000 points when we don't have a given total size for instance
					progressNumber(idx);
				}
			}
		}
		if (show_progress)
			if(total_points != -1)
				progressBar(100, idx+1);
			else
				progressNumber(idx+1);
	}

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
	virtual std::pair<std::vector<Point_t>, std::vector<PointMetadata>> readMeta() = 0;
};