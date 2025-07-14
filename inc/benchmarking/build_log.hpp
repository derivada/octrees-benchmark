#pragma once

#include <string>
#include <iostream>
#include <iomanip>
#include "main_options.hpp"

/// @brief Object with logs of the build time for tested structures
struct BuildLog {
    // Data cloud size
    size_t cloudSize = 0;

    // The encoder used
    EncoderType encoding;

    // Octree general parameters
    size_t maxLeafPoints = 0;
    SearchStructure structure;

    // Build step times (vary between LinearOctree and Octree)
    double linearOctreeLeafTime = 0.0;
    double linearOctreeInternalTime = 0.0;
    double buildTime = 0.0;

    // Mem used in bytes
    size_t memoryUsed = 0;

    // Amount of nodes
    size_t totalNodes = 0;
    size_t leafNodes = 0;
    size_t internalNodes = 0;

    // Max depth and min radius at max depth
    size_t maxDepthSeen = 0;
    double minRadiusAtMaxDepth = 0.0;


    // Cache miss counts (if using --cache-profiling)
    long long l1dMisses = 0;
    long long l2dMisses = 0;
    long long l3Misses = 0;

    friend std::ostream& operator<<(std::ostream& os, const BuildLog& log) {
        std::string memoryUsedStr = std::to_string(log.memoryUsed / (1024.0 * 1024)) + " MB";
        os << "Encoding and octree construction log:\n";
        os << std::left << std::setw(32) << "Point type:" << "Point" << "\n";
        os << std::left << std::setw(32) << "Structure:" << searchStructureToString(log.structure) << "\n";
        os << std::left << std::setw(32) << "Max. points in leaf:" << log.maxLeafPoints << "\n";
        os << std::left << std::setw(32) << "Encoder type:" << encoderTypeToString(log.encoding) << "\n";
        os << std::left << std::setw(32) << "Cloud size:" << log.cloudSize << "\n";
        if(log.structure == SearchStructure::LINEAR_OCTREE) {
            os << std::left << std::setw(32) << "Linear octree leaves time:" << log.linearOctreeLeafTime << " sec\n";
            os << std::left << std::setw(32) << "Linear octree internal time:" << log.linearOctreeInternalTime << " sec\n";
        }
        os << std::left << std::setw(32) << "Total build time:" << log.buildTime << " sec\n\n";
        os << std::left << std::setw(32) << "Memory used:" << memoryUsedStr << "\n";
        os << std::left << std::setw(32) << "Number of nodes:" << log.totalNodes << "\n";
        os << std::left << std::setw(32) << "  Leafs:" << log.leafNodes << "\n";
        os << std::left << std::setw(32) << "  Internal nodes:" << log.internalNodes << "\n";
        os << std::left << std::setw(32) << "Max. depth seen:" << log.maxDepthSeen << "\n";
        os << std::left << std::setw(32) << "Min. radii seen:" << log.minRadiusAtMaxDepth << "\n";
        if(mainOptions.cacheProfiling) {
            os << std::left << std::setw(32) << "L1d cache misses:" << log.l1dMisses << "\n";
            os << std::left << std::setw(32) << "L2d cache misses:" << log.l2dMisses << "\n";
            os << std::left << std::setw(32) << "L3 cache misses:" << log.l3Misses << "\n";
        }
        os << "\n";
        return os;
    }

    void toCSV(std::ostream& out) const {
        out  << "Point" << ","
             << searchStructureToString(structure) << ","
             << maxLeafPoints << ","
             << encoderTypeToString(encoding) << ","
             << cloudSize << ","
             << linearOctreeLeafTime << ","
             << linearOctreeInternalTime << ","
             << buildTime << ","
             << memoryUsed << ","
             << totalNodes << ","
             << leafNodes << ","
             << internalNodes << ","
             << maxDepthSeen << ","
             << minRadiusAtMaxDepth << ","
             << mainOptions.repeats << ","
             << mainOptions.useWarmup << ","
             << l1dMisses << ","
             << l2dMisses << ","
             << l3Misses << "\n"; 
    }
    
    static void writeCSVHeader(std::ostream& out) {
        out  << "point_type,"
             << "structure,"
             << "max_leaf_points,"
             << "enc_type,"
             << "cloud_size,"
             << "linear_octree_leaf_time,"
             << "linear_octree_internal_time,"
             << "build_time,"
             << "memory_used,"
             << "number_of_nodes,"
             << "leaf_nodes,"
             << "internal_nodes,"
             << "max_depth_seen,"
             << "min_radii_seen,"
             << "repeats,"
             << "used_warmup,"
             << "l1d_miss,"
             << "l2d_miss,"
             << "l3_miss\n";
    }
};
