#pragma once

#include <string>
#include <iostream>
#include <iomanip>
#include "PointEncoding/point_encoder.hpp"
#include "type_names.hpp"

/// @brief Object with logs of the encoding and build time for both octrees and some basic info
struct EncodingOctreeLog {
    // Data cloud size
    size_t cloudSize = 0;
    std::string pointType = "";

    // The encoder being used
    std::string encoderType = "";

    // Encoding and sorting times
    double boundingBoxTime = 0.0;
    double encodingTime = 0.0;
    double sortingTime = 0.0;

    // Octree general parameters
    size_t max_leaf_points = 0;
    double min_octant_radius = 0.0; // unused
    std::string octreeType = "";

    // Build step times (vary between LinearOctree and Octree)
    double octreeLeafTime = 0.0;
    double octreeInternalTime = 0.0;
    double octreeTime = 0.0;

    // Mem used in bytes
    size_t memoryUsed = 0;

    // Amount of nodes
    size_t totalNodes = 0;
    size_t leafNodes = 0;
    size_t internalNodes = 0;

    // Max depth and min radius at max depth
    size_t maxDepthSeen = 0;
    double minRadiusAtMaxDepth = 0.0;

    friend std::ostream& operator<<(std::ostream& os, const EncodingOctreeLog& log) {
        std::string memoryUsedStr = std::to_string(log.memoryUsed / (1024.0 * 1024)) + " MB";
        os << "Encoding and octree construction log:\n";
        os << std::left << std::setw(LOG_FIELD_WIDTH) << "Point type:" << log.pointType << "\n";
        os << std::left << std::setw(LOG_FIELD_WIDTH) << "Octree type:" << log.octreeType << "\n";
        os << std::left << std::setw(LOG_FIELD_WIDTH) << "Max. points in leaf:" << log.max_leaf_points << "\n";
        os << std::left << std::setw(LOG_FIELD_WIDTH) << "Encoder type:" << log.encoderType << "\n";
        os << std::left << std::setw(LOG_FIELD_WIDTH) << "Cloud size:" << log.cloudSize << "\n";
        os << std::left << std::setw(LOG_FIELD_WIDTH) << "Find bbox. time:" << log.boundingBoxTime << " sec\n";
        os << std::left << std::setw(LOG_FIELD_WIDTH) << "Encoding time:" << log.encodingTime << " sec\n";
        os << std::left << std::setw(LOG_FIELD_WIDTH) << "Sorting time:" << log.sortingTime << " sec\n";
        os << std::left << std::setw(LOG_FIELD_WIDTH) << "Octree leaves time:" << log.octreeLeafTime << " sec\n";
        os << std::left << std::setw(LOG_FIELD_WIDTH) << "Octree internal time:" << log.octreeInternalTime << " sec\n";
        os << std::left << std::setw(LOG_FIELD_WIDTH) << "Octree total time:" << log.octreeTime << " sec\n\n";
        os << std::left << std::setw(LOG_FIELD_WIDTH) << "Octree memory:" << memoryUsedStr << "\n";
        os << std::left << std::setw(LOG_FIELD_WIDTH) << "Number of nodes:" << log.totalNodes << "\n";
        os << std::left << std::setw(LOG_FIELD_WIDTH) << "  Leafs:" << log.leafNodes << "\n";
        os << std::left << std::setw(LOG_FIELD_WIDTH) << "  Internal nodes:" << log.internalNodes << "\n";
        os << std::left << std::setw(LOG_FIELD_WIDTH) << "Max. depth seen:" << log.maxDepthSeen << "\n";
        os << std::left << std::setw(LOG_FIELD_WIDTH) << "Min. radii seen:" << log.minRadiusAtMaxDepth << "\n\n";
        return os;
    }

    void toCSV(std::ostream& out) const {
        out  << pointType << ","
             << octreeType << ","
             << max_leaf_points << ","
             << encoderType << ","
             << cloudSize << ","
             << boundingBoxTime << ","
             << encodingTime << ","
             << sortingTime << ","
             << min_octant_radius << ","
             << octreeLeafTime << ","
             << octreeInternalTime << ","
             << octreeTime << ","
             << memoryUsed << ","
             << totalNodes << ","
             << leafNodes << ","
             << internalNodes << ","
             << maxDepthSeen << ","
             << minRadiusAtMaxDepth
             << "\n";
    }
    
    static void writeCSVHeader(std::ostream& out) {
        out  << "point_type,"
             << "oct_type,"
             << "enc_type,"
             << "max_leaf_points,"
             << "cloud_size,"
             << "bbox_time,"
             << "enc_time,"
             << "sort_time,"
             << "max_leaf_points,"
             << "min_oct_radius,"
             << "octree_leaf_time,"
             << "octree_internal_time,"
             << "octree_time,"
             << "octree_memory,"
             << "number_of_nodes,"
             << "leaf_nodes,"
             << "internal_nodes,"
             << "max_depth_seen,"
             << "min_radii_seen\n";
    }
    
};
