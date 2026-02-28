#pragma once
#ifdef HAVE_PCL
#include <pcl/point_cloud.h>
#include "geometry/point_containers.hpp"

template <PointContainer Container>
pcl::PointCloud<pcl::PointXYZ> convertCloudToPCL(Container &points) {
    pcl::PointCloud<pcl::PointXYZ> pclCloud;
    pclCloud.width = points.size();
    pclCloud.height = 1;
    pclCloud.points.resize(points.size());
    #pragma omp parallel for schedule(runtime)
        for (size_t i = 0; i < points.size(); ++i) {
            pclCloud.points[i].x = points[i].getX();
            pclCloud.points[i].y = points[i].getY();
            pclCloud.points[i].z = points[i].getZ();
        }
    return pclCloud;
}

template <typename PointT>
size_t estimatePCLOctMemory(const pcl::octree::OctreePointCloud<PointT>& octree)
{
    using OctreeT   = pcl::octree::OctreePointCloud<PointT>;
    using Iterator  = pcl::octree::OctreeDepthFirstIterator<OctreeT>;
    using LeafNode  = typename OctreeT::LeafNode;
    using BranchNode= typename OctreeT::BranchNode;

    size_t total = sizeof(octree);

    for (auto it = octree.begin(); it != octree.end(); ++it) {
        auto node = it.getCurrentOctreeNode();
        if (it.isBranchNode()) total += sizeof(typename OctreeT::BranchNode);
        else if (it.isLeafNode()) total += sizeof(typename OctreeT::LeafNode);
    }

    return total;
}
#endif