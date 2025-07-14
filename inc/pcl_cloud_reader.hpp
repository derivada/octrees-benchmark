#pragma once
#ifdef HAVE_PCL
#include <pcl/point_cloud.h>
template <typename Point_t>
pcl::PointCloud<pcl::PointXYZ> convertCloudToPCL(std::vector<Point_t> &points) {
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
#endif