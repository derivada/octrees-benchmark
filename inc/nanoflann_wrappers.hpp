#pragma once

// Wrapper around Point
template <typename Point_t>
struct NanoflannPointCloud
{
    using coord_t = double;  //!< The type of each coordinate

    std::vector<Point_t> &pts;
    NanoflannPointCloud(std::vector<Point_t> &points) : pts(points) {};
    
    // Must return the number of data points
    inline size_t kdtree_get_point_count() const { return pts.size(); }

    // Returns the dim'th component of the idx'th point in the class:
    // Since this is inlined and the "dim" argument is typically an immediate
    // value, the
    //  "if/else's" are actually solved at compile time.
    inline double kdtree_get_pt(const size_t idx, const size_t dim) const
    {
        if (dim == 0)
            return pts[idx].getX();
        else if (dim == 1)
            return pts[idx].getY();
        else
            return pts[idx].getZ();
    }

    // Optional bounding-box computation: return false to default to a standard
    // bbox computation loop.
    //   Return true if the BBOX was already computed by the class and returned
    //   in "bb" so it can be avoided to redo it again. Look at bb.size() to
    //   find out the expected dimensionality (e.g. 2 or 3 for point clouds)
    template <class BBOX>
    bool kdtree_get_bbox(BBOX& /* bb */) const
    {
        return false;
    }
};

// nanoflann kd tree
template <typename Point_t>
using NanoFlannKDTree = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<double, NanoflannPointCloud<Point_t>>,
    NanoflannPointCloud<Point_t>,
    3,
    size_t>;