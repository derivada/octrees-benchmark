#pragma once
#include <omp.h>
#include <papi.h>
#include <type_traits>

#include "geometry/point_containers.hpp"
#include "structures/linear_octree.hpp"
#include "structures/nanoflann.hpp"
#include "structures/nanoflann_wrappers.hpp"
#include "structures/octree.hpp"
#include "structures/unibn_octree.hpp"

#include "benchmarking.hpp"
#include "build_log.hpp"
#include "encoding_log.hpp"
#include "main_options.hpp"
#include "time_watcher.hpp"

#ifdef HAVE_PCL
#include <pcl/point_cloud.h>
#include <pcl/octree/octree_search.h>
#include <pcl/kdtree/kdtree_flann.h>
#include "structures/pcl_wrappers.hpp"
#endif

template <PointContainer Container>
class MemoryBenchmarks {
    using key_t = size_t;
    using PointEncoder = PointEncoding::PointEncoder;
    
    private:
        Container& points;
        std::vector<key_t> codes;
        Box box;
        PointEncoder& enc;

        size_t runPoct() {
            std::shared_ptr<BuildLog> log = std::make_shared<BuildLog>();
            Octree oct(points, box);
            return log->memoryUsed;
        }

        size_t runUnibn() {
            unibn::OctreeParams params;
            std::shared_ptr<BuildLog> log = std::make_shared<BuildLog>();
            unibn::Octree<Point, Container> oct;
            params.bucketSize = mainOptions.maxPointsLeaf;
            oct.initialize(points, params);
            oct.logOctreeData(log);
            return log->memoryUsed;
        }

        size_t runLinoct() {
            unibn::OctreeParams params;
            std::shared_ptr<BuildLog> log = std::make_shared<BuildLog>();
            LinearOctree oct(points, codes, box, enc, log);
            return log->memoryUsed;
        }

        size_t runPCLOct() {
            auto pclCloud = convertCloudToPCL(points);
                pcl::octree::OctreePointCloudSearch<pcl::PointXYZ> oct(mainOptions.pclOctResolution);
                pcl::PointCloud<pcl::PointXYZ>::Ptr cloudPtr = pclCloud.makeShared();
                oct.setInputCloud(cloudPtr);
                oct.addPointsFromInputCloud();
                return estimatePCLOctMemory(oct);
        }

        size_t runPCLKD() {
            auto pclCloud = convertCloudToPCL(points);
            pcl::KdTreeFLANN<pcl::PointXYZ> kdtree = pcl::KdTreeFLANN<pcl::PointXYZ>();
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloudPtr = pclCloud.makeShared();
            kdtree.setInputCloud(cloudPtr);
            return 0;
        }

        size_t runNano() {
            NanoflannPointCloud<Container> npc(points);
            NanoFlannKDTree<Container> kdtree(3, npc, {mainOptions.maxPointsLeaf});
            return kdtree.usedMemory(kdtree);
        }
    
    public:
        MemoryBenchmarks(Container &points, std::vector<key_t> &codes, Box box, PointEncoder &enc): 
            points(points), codes(codes), box(box), enc(enc) {}

        void run() {
            // Sleep for 3 seconds so it is easier to heap profile
            auto structure = mainOptions.memoryStructure.value();
            std::cout << "Running memory benchmark on structure " << searchStructureToString(structure) << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(3));
            size_t theoSize = 0;
            switch(structure) {
                case PTR_OCTREE:
                    theoSize = runPoct();
                    break;
                case LINEAR_OCTREE:
                    theoSize = runLinoct();
                    break;
                case UNIBN_OCTREE:
                    theoSize = runUnibn();
                    break;
                case PCL_OCTREE:
                    theoSize = runPCLOct();
                    break;
                case PCL_KDTREE:
                    theoSize = runPCLKD();
                    break;
                case NANOFLANN_KDTREE:
                    theoSize = runNano();
                    break;
                default:
                    std::cerr << "Unknown SearchStructure type!" << std::endl;
                    break;
            }

            auto theoSizeMB = theoSize / (1024.0 * 1024.0);
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(2)
                << theoSize << " (" << theoSizeMB << " MB)";
            std::string theoSizeStr = oss.str();
            std::cout << "Theoretical size: " << theoSizeStr << std::endl;
        }
};
