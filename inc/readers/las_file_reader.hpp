//
// Created by miguelyermo on 6/8/21.
//

#pragma once

#include <lasreader.hpp>
#include <vector>
#include <utility>
#include "geometry/point.hpp"
#include "geometry/point_metadata.hpp"
#include "file_reader.hpp"
#include "util.hpp"

/**
 * @author Miguel Yermo
 * @brief Specialization of FileRead to read .las/.laz files
 */
template <PointContainer Container>
class LasFileReader : public FileReader<Container>
{
	public:
	// ***  CONSTRUCTION / DESTRUCTION  *** //
	// ************************************ //
	LasFileReader(const fs::path& path) : FileReader<Container>(path){};
	~LasFileReader(){};

	/**
	 * @brief Reads the points contained in the .las/.laz file
	 * @return Container of points
	 */
	Container read()
	{
		Container points;

		// LAS File reading
		LASreadOpener lasreadopener;
		lasreadopener.set_file_name(this->path.c_str());
		LASreader* lasreader = lasreadopener.open();

		// TODO: Use extended_point_records para LAS v1.4
		// https://gitlab.citius.usc.es/oscar.garcia/las-shapefile-classifier/-/blob/master/lasreader_wrapper.cc

		// Scale factors for each coordinate
		double xScale = lasreader->header.x_scale_factor;
		double yScale = lasreader->header.y_scale_factor;
		double zScale = lasreader->header.z_scale_factor;

		double xOffset = lasreader->header.x_offset;
		double yOffset = lasreader->header.y_offset;
		double zOffset = lasreader->header.z_offset;

  		// Get total number of points for progress bar
		size_t total_points = lasreader->header.number_of_point_records;

		// Reset reader to beginning
		lasreadopener.set_file_name(this->path.c_str());
		lasreader = lasreadopener.open();

		// Termination condition
		auto terminationCondition = [&lasreader]() { 
			return lasreader->read_point(); 
		};

		// Point creation
		auto pointInserter = [&](size_t& idx) {
			points.push_back(Point(
				idx, 
				static_cast<double>(lasreader->point.get_X() * xScale + xOffset),
				static_cast<double>(lasreader->point.get_Y() * yScale + yOffset),
				static_cast<double>(lasreader->point.get_Z() * zScale + zOffset))
				// static_cast<double>(lasreader->point.get_intensity()),
				// static_cast<unsigned short>(lasreader->point.get_return_number()),
				// static_cast<unsigned short>(lasreader->point.get_number_of_returns()),
				// static_cast<unsigned short>(lasreader->point.get_scan_direction_flag()),
				// static_cast<unsigned short>(lasreader->point.get_edge_of_flight_line()),
				// static_cast<unsigned short>(lasreader->point.get_classification()),
				// static_cast<char>(lasreader->point.get_scan_angle_rank()),
				// static_cast<unsigned short>(lasreader->point.get_user_data()),
				// static_cast<unsigned short>(lasreader->point.get_point_source_ID()),
				// static_cast<unsigned int>(lasreader->point.get_R()),
				// static_cast<unsigned int>(lasreader->point.get_G()),
				// static_cast<unsigned int>(lasreader->point.get_B())
			);
		};

		this->file_reading_loop(terminationCondition, pointInserter, total_points, false);

		// Cleanup
		lasreader->close();
		delete lasreader;
		return points;
	}

	std::pair<Container, std::vector<PointMetadata>> readMeta()
	{
		Container points;
		std::vector<PointMetadata> metadata;

		// LAS File reading
		LASreadOpener lasreadopener;
		lasreadopener.set_file_name(this->path.c_str());
		LASreader* lasreader = lasreadopener.open();
		// TODO: Use extended_point_records para LAS v1.4
		// https://gitlab.citius.usc.es/oscar.garcia/las-shapefile-classifier/-/blob/master/lasreader_wrapper.cc

		// Scale factors for each coordinate
		double xScale = lasreader->header.x_scale_factor;
		double yScale = lasreader->header.y_scale_factor;
		double zScale = lasreader->header.z_scale_factor;

		double xOffset = lasreader->header.x_offset;
		double yOffset = lasreader->header.y_offset;
		double zOffset = lasreader->header.z_offset;

  		// Get total number of points for progress bar
		size_t total_points = lasreader->header.number_of_point_records;

		// Reset reader to beginning
		lasreadopener.set_file_name(this->path.c_str());
		lasreader = lasreadopener.open();

		// Termination condition
		auto terminationCondition = [&lasreader]() { 
			return lasreader->read_point(); 
		};

		// Point creation
		auto pointInserter = [&](size_t& idx) {
			points.push_back(Point(
				idx, 
				static_cast<double>(lasreader->point.get_X() * xScale + xOffset),
				static_cast<double>(lasreader->point.get_Y() * yScale + yOffset),
				static_cast<double>(lasreader->point.get_Z() * zScale + zOffset))
			);
			metadata.emplace_back(
				static_cast<double>(lasreader->point.get_intensity()),
				static_cast<unsigned short>(lasreader->point.get_return_number()),
				static_cast<unsigned short>(lasreader->point.get_number_of_returns()),
				static_cast<unsigned short>(lasreader->point.get_scan_direction_flag()),
				static_cast<unsigned short>(lasreader->point.get_edge_of_flight_line()),
				static_cast<unsigned short>(lasreader->point.get_classification()),
				static_cast<char>(lasreader->point.get_scan_angle_rank()),
				static_cast<unsigned short>(lasreader->point.get_user_data()),
				static_cast<unsigned short>(lasreader->point.get_point_source_ID()),
				static_cast<unsigned int>(lasreader->point.get_R()),
				static_cast<unsigned int>(lasreader->point.get_G()),
				static_cast<unsigned int>(lasreader->point.get_B())
			);
		};

		this->file_reading_loop(terminationCondition, pointInserter, total_points, false);

		// Cleanup
		lasreader->close();
		delete lasreader;
		return std::pair(points, metadata);
	}

};
