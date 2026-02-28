#pragma once

#include <omp.h>
#include <type_traits>

#include "benchmarking/search_set.hpp"
#include "kernels/kernel_factory.hpp"
#include "structures/nanoflann.hpp"
#include "structures/nanoflann_wrappers.hpp"

#include "benchmarking.hpp"
#include "main_options.hpp"
#include "time_watcher.hpp"

template <PointContainer Container>
class LocalityBenchmark {
    private:
        using PointEncoder = PointEncoding::PointEncoder;
        using key_t = PointEncoding::key_t;
        using coords_t = PointEncoding::coords_t;
        Container& points;
        std::vector<key_t>& codes;
        Box box;
        PointEncoder& enc;
        std::ofstream &outputFile;
        
    public:
        LocalityBenchmark(Container& points, std::vector<key_t>& codes, Box box, PointEncoder& enc,
            std::ofstream &file) :
            points(points), 
            codes(codes),
            box(box),
            enc(enc),
            outputFile(file) {}

        // with knn
        void histogramLocality(size_t k) {
            NanoflannPointCloud<Container> npc(points);
            NanoFlannKDTree<Container> kdtree(3, npc, {mainOptions.maxPointsLeaf});
            std::unordered_map<size_t, size_t> indexDistances;
            std::cout << "Computing locality histogram" << std::endl;
            size_t totalIters = points.size();
            size_t step = std::max<size_t>(1, totalIters / 10);
            std::atomic<size_t> progress(0);
            TimeWatcher tw;
            tw.start();
            #pragma omp parallel
            {
                std::unordered_map<size_t, size_t> localIndexDistances;
                
                #pragma omp for schedule(runtime)
                for(size_t i = 0; i < totalIters; i++) {
                    const double pt[3] = {points[i].getX(), points[i].getY(), points[i].getZ()};
                    // knn
                    std::vector<size_t> indexes(k);
                    std::vector<double> distances(k);
                    const size_t nMatches = kdtree.knnSearch(pt, k, &indexes[0], &distances[0]);
                    for(size_t j = 0; j < nMatches; j++) {
                        size_t diff = (i > indexes[j]) 
                             ? (i - indexes[j]) : (indexes[j] - i);
                        localIndexDistances[diff] += 1;
                    }
            
                    // Progress tracking
                    size_t done = ++progress;
                    if (done % step == 0) {
                        int percent = static_cast<int>(100.0 * done / totalIters);
                        #pragma omp critical
                        {
                            std::cout << "  ... " << percent+1 << "% of knn searches done" << std::endl;
                        }
                    }
                }
                #pragma omp critical
                {
                    for (auto &kv : localIndexDistances) {
                        indexDistances[kv.first] += kv.second;
                    }
                }
            }
            tw.stop();
            std::cout << "Finished computing neighbours in " << tw.getElapsedDecimalSeconds() << " seconds" << std::endl;
            // printPapiResults(events, descriptions, eventValues);
            // Write CSV output
            std::cout << "Writing locality histogram" << std::endl;
            tw.start();
            auto encoderName = enc.getEncoderName();
            outputFile << "encoder,distance,count\n";
            
            size_t total = indexDistances.size();
            step  = std::max<size_t>(1, total / 10); // 10% steps
            size_t counter = 0;
            size_t nextLog = step;
            
            for (const auto &kv : indexDistances) {
                outputFile << encoderName << "," << kv.first << "," << kv.second << "\n";
            
                counter++;
                if (counter >= nextLog) {
                    int percent = static_cast<int>(100.0 * counter / total);
                    std::cout << "  ... " << percent+1 << "% done" << std::endl;
                    nextLog += step;
                }
            }
            
            outputFile << std::flush;
            tw.stop();
            std::cout << "Finished writing histogram (" << total << " entries) in " 
                << tw.getElapsedDecimalSeconds() << " seconds" << std::endl;
        }
};