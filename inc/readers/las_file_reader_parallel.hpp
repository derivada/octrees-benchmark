//
// Created by miguelyermo on 6/8/21.
//

#pragma once

#include <lasreader.hpp>
#include <vector>
#include <omp.h>
#include <utility>

#include "geometry/point.hpp"
#include "geometry/point_metadata.hpp"
#include "file_reader.hpp"
#include "util.hpp"

/**
 * @author Miguel Yermo and Pablo Díaz
 * @brief Specialization of FileRead to read .las/.laz files. Reads the LAS file in parallel after retrieving the
 * amount of points from it, using LAsreader seek() method to skip ahead for each thread.
 */
template <PointContainer Container>
class LasFileReaderParallel : public FileReader<Container>
{
	public:
	// ***  CONSTRUCTION / DESTRUCTION  *** //
	// ************************************ //
	LasFileReaderParallel(const fs::path& path) : FileReader<Container>(path){};
	~LasFileReaderParallel(){};

	/// @brief Helper to create a new LASreader for each thread and seek the file to the correct position
	LASreader* createReaderAndSeek(size_t start_idx = 0) {
		LASreadOpener opener;
		opener.set_file_name(this->path.c_str());
		LASreader* lasreader = opener.open();
		lasreader->seek(start_idx);
		return lasreader;
	};

	void closeReader(LASreader* lasreader) {
		lasreader->close();
		delete lasreader;
	};

	struct PointCloudInfo {
		double xScale, yScale, zScale;
		double xOffset, yOffset, zOffset;
		size_t totalPoints;
	};

	PointCloudInfo readPointCloudInfo(LASreader* lasreader) {
		PointCloudInfo pci;
		pci.xScale = lasreader->header.x_scale_factor;
		pci.yScale = lasreader->header.y_scale_factor;
		pci.zScale = lasreader->header.z_scale_factor;

		pci.xOffset = lasreader->header.x_offset;
		pci.yOffset = lasreader->header.y_offset;
		pci.zOffset = lasreader->header.z_offset;

  		pci.totalPoints = lasreader->header.number_of_point_records;
		return pci;
	};

	/**
	 * @brief Reads the points contained in the .las/.laz file
	 * @return Vector of point_t
	 */
	Container read() {
		// TODO: Use extended_point_records para LAS v1.4
		// https://gitlab.citius.usc.es/oscar.garcia/las-shapefile-classifier/-/blob/master/lasreader_wrapper.cc
		auto lasreader = createReaderAndSeek();
		auto pci = readPointCloudInfo(lasreader);
		closeReader(lasreader);

		Container points(pci.totalPoints);
		int num_threads = omp_get_max_threads();
		size_t points_per_thread = pci.totalPoints / num_threads;

		#pragma omp parallel num_threads(num_threads)
		{
			int tid = omp_get_thread_num();
			size_t start = tid * points_per_thread;
			size_t end = (tid == num_threads - 1) ? pci.totalPoints : start + points_per_thread;
			LASreader* reader = createReaderAndSeek(start);
			for (size_t i = start; i < end; ++i) {
				if (!reader->read_point()) break;
				points.set(i, Point(
					i,
					static_cast<double>(reader->point.get_X() * pci.xScale + pci.xOffset),
					static_cast<double>(reader->point.get_Y() * pci.yScale + pci.yOffset),
					static_cast<double>(reader->point.get_Z() * pci.zScale + pci.zOffset)
				));

				// points[i] = Point(
				// 	i,
				// 	static_cast<double>(reader->point.get_X() * pci.xScale + pci.xOffset),
				// 	static_cast<double>(reader->point.get_Y() * pci.yScale + pci.yOffset),
				// 	static_cast<double>(reader->point.get_Z() * pci.zScale + pci.zOffset),
				// static_cast<double>(reader->point.get_intensity()),
				// static_cast<unsigned short>(reader->point.get_return_number()),
				// static_cast<unsigned short>(reader->point.get_number_of_returns()),
				// static_cast<unsigned short>(reader->point.get_scan_direction_flag()),
				// static_cast<unsigned short>(reader->point.get_edge_of_flight_line()),
				// static_cast<unsigned short>(reader->point.get_classification()),
				// static_cast<char>(reader->point.get_scan_angle_rank()),
				// static_cast<unsigned short>(reader->point.get_user_data()),
				// static_cast<unsigned short>(reader->point.get_point_source_ID()),
				// static_cast<unsigned int>(reader->point.get_R()),
				// static_cast<unsigned int>(reader->point.get_G()),
				// static_cast<unsigned int>(reader->point.get_B())
				// );
			}
			closeReader(reader);
		}
		return points;
	}
	
	std::pair<Container, std::vector<PointMetadata>> readMeta() {
		// TODO: Use extended_point_records para LAS v1.4
		// https://gitlab.citius.usc.es/oscar.garcia/las-shapefile-classifier/-/blob/master/lasreader_wrapper.cc
		auto lasreader = createReaderAndSeek();
		auto pci = readPointCloudInfo(lasreader);
		closeReader(lasreader);

		Container points(pci.totalPoints);
		std::vector<PointMetadata> metadata(pci.totalPoints);
		int num_threads = omp_get_max_threads();
		size_t points_per_thread = pci.totalPoints / num_threads;

		#pragma omp parallel num_threads(num_threads)
		{
			int tid = omp_get_thread_num();
			size_t start = tid * points_per_thread;
			size_t end = (tid == num_threads - 1) ? pci.totalPoints : start + points_per_thread;
			LASreader* reader = createReaderAndSeek(start);
			for (size_t i = start; i < end; ++i) {
				if (!reader->read_point()) break;
				points.set(i, Point(
					i,
					static_cast<double>(reader->point.get_X() * pci.xScale + pci.xOffset),
					static_cast<double>(reader->point.get_Y() * pci.yScale + pci.yOffset),
					static_cast<double>(reader->point.get_Z() * pci.zScale + pci.zOffset)
				));
				metadata[i] = PointMetadata(
					static_cast<double>(reader->point.get_intensity()),
					static_cast<unsigned short>(reader->point.get_return_number()),
					static_cast<unsigned short>(reader->point.get_number_of_returns()),
					static_cast<unsigned short>(reader->point.get_scan_direction_flag()),
					static_cast<unsigned short>(reader->point.get_edge_of_flight_line()),
					static_cast<unsigned short>(reader->point.get_classification()),
					static_cast<char>(reader->point.get_scan_angle_rank()),
					static_cast<unsigned short>(reader->point.get_user_data()),
					static_cast<unsigned short>(reader->point.get_point_source_ID()),
					static_cast<unsigned int>(reader->point.get_R()),
					static_cast<unsigned int>(reader->point.get_G()),
					static_cast<unsigned int>(reader->point.get_B())
				);
			}
			closeReader(reader);
		}
		return std::make_pair(points, metadata);
	}
};
