#pragma once

#include <string>
#include <iostream>
#include <iomanip>
#include "main_options.hpp"

/// @brief Object with logs of the encoding time for each encoder
struct EncodingLog {
    // Data cloud size
    size_t cloudSize = 0;

    // The encoder being used
    EncoderType encoding;

    // Encoding and sorting times
    double boundingBoxTime = 0.0;
    double encodingTime = 0.0;
    double sortingTime = 0.0;

    friend std::ostream& operator<<(std::ostream& os, const EncodingLog& log) {
        os << "Encoding log:\n";
        os << std::left << std::setw(32) << "Point type:" << "Point" << "\n";
        os << std::left << std::setw(32) << "Cloud size:" << log.cloudSize << "\n";
        os << std::left << std::setw(32) << "Encoder type:" << encoderTypeToString(log.encoding) << "\n";
        os << std::left << std::setw(32) << "Find bbox. time:" << log.boundingBoxTime << " sec\n";
        os << std::left << std::setw(32) << "Encoding time:" << log.encodingTime << " sec\n";
        os << std::left << std::setw(32) << "Sorting time:" << log.sortingTime << " sec\n";
        return os;
    }

    void toCSV(std::ostream& out) const {
        out << containerTypeToString(mainOptions.containerType) << ","
            << encoderTypeToString(encoding) << ","
            << cloudSize << ","
            << boundingBoxTime << ","
            << encodingTime << ","
            << sortingTime << ","
            << mainOptions.repeats << ","
            << mainOptions.useWarmup << "\n";
    }
    
    static void writeCSVHeader(std::ostream& out) {
        out  << "container,"
             << "enc_type,"
             << "cloud_size,"
             << "bbox_time,"
             << "enc_time,"
             << "sort_time,"
             << "repeats,"
             << "used_warmup\n";
    }
};
