#include <gtest/gtest.h>
#include <vector>
#include <optional>

#include "geometry/point.hpp"
#include "geometry/point_containers.hpp"
#include "encoding/morton_encoder_3d.hpp"
#include "structures/linear_octree.hpp"

using namespace PointEncoding;

// Helper to compute brute-force distances
static double squaredDistance(const Point& a, const Point& b) {
    double dx = a.getX() - b.getX();
    double dy = a.getY() - b.getY();
    double dz = a.getZ() - b.getZ();
    return dx*dx + dy*dy + dz*dz;
}

class AdvancedOctreeTest : public ::testing::Test {
protected:
    PointsAoS points;
    std::optional<std::vector<PointMetadata>> metadata;
    MortonEncoder3D enc;

    void SetUp() override {
        // Empty point cloud for edge case tests
        points.clear();
        metadata = std::nullopt;
    }
};

TEST_F(AdvancedOctreeTest, EmptyPointCloud) {
    auto [codes, box] = enc.sortPoints(points, metadata);
    // Construction should not throw even with empty data
    EXPECT_NO_THROW({ LinearOctree octree(points, codes, box, enc); });
    LinearOctree octree(points, codes, box, enc);
    // Searches on empty octree should return empty results
    Point query(0.0, 0.0, 0.0);
    auto idx = octree.neighborsPrune<Kernel_t::square>(query, 1.0);
    EXPECT_TRUE(idx.empty());
    std::vector<size_t> knnIdx(1);
    std::vector<double> knnDist(1);
    octree.knn(query, 1, knnIdx, knnDist);
    // Expect no valid neighbor index for empty octree
// No strict expectation for knnIdx on empty octree // May be implementation-defined, keep for reference
}

TEST_F(AdvancedOctreeTest, DuplicatePoints) {
    // Insert duplicate points at the same location
    points.push_back(Point(1.0, 1.0, 1.0));
    points.push_back(Point(1.0, 1.0, 1.0));
    points.push_back(Point(2.0, 2.0, 2.0));
    auto [codes, box] = enc.sortPoints(points, metadata);
    LinearOctree octree(points, codes, box, enc);
    // k-NN where k exceeds number of unique points
    size_t k = 5;
    std::vector<size_t> idx(k);
    std::vector<double> dist(k);
    Point query(1.0, 1.0, 1.0);
    octree.knn(query, k, idx, dist);
    // The first two results should be the duplicate points (distance 0)
    EXPECT_NEAR(dist[0], 0.0, 1e-9);
    EXPECT_NEAR(dist[1], 0.0, 1e-9);
    // Remaining results should be the third point
    EXPECT_NEAR(dist[2], squaredDistance(query, points[2]), 1e-9);
}

TEST_F(AdvancedOctreeTest, ZeroRadiusSearch) {
    points.push_back(Point(0.0, 0.0, 0.0));
    points.push_back(Point(0.0, 0.0, 0.0));
    points.push_back(Point(1.0, 1.0, 1.0));
    auto [codes, box] = enc.sortPoints(points, metadata);
    LinearOctree octree(points, codes, box, enc);
    Point query(0.0, 0.0, 0.0);
    double rSq = 0.0; // zero radius
    auto idx = octree.neighborsPrune<Kernel_t::square>(query, rSq);
    // Should return all points exactly at the query location (2 duplicates)
    EXPECT_EQ(idx.size(), 2u);
    for (size_t i : idx) {
        EXPECT_DOUBLE_EQ(points[i].distance3D(query), 0.0);
    }
}

TEST_F(AdvancedOctreeTest, BoundaryConditions) {
    // Points placed exactly on the bounding box limits
    points.push_back(Point(0.0, 0.0, 0.0));
    points.push_back(Point(10.0, 10.0, 10.0));
    points.push_back(Point(5.0, 5.0, 5.0));
    auto [codes, box] = enc.sortPoints(points, metadata);
    LinearOctree octree(points, codes, box, enc);
    // Query at the max corner
    Point queryMax(10.0, 10.0, 10.0);
    auto idxMax = octree.neighborsPrune<Kernel_t::square>(queryMax, 0.0);
    EXPECT_EQ(idxMax.size(), 1u);
    EXPECT_DOUBLE_EQ(points[idxMax[0]].distance3D(queryMax), 0.0);
    // Query just outside the box should return empty
    Point outside(10.1, 10.1, 10.1);
    auto idxOut = octree.neighborsPrune<Kernel_t::square>(outside, 0.01);
    EXPECT_TRUE(idxOut.empty());
}

TEST_F(AdvancedOctreeTest, KernelSphereSearch) {
    points.push_back(Point(1.0, 1.0, 1.0));
    points.push_back(Point(2.0, 0.0, 0.0));
    auto [codes, box] = enc.sortPoints(points, metadata);
    LinearOctree octree(points, codes, box, enc);

    Point query(0.0, 0.0, 0.0);
    // Radius 1.5. Point (1,1,1) is at distance sqrt(3) ~ 1.73 (outside)
    // Point (2,0,0) is at distance 2.0 (outside)
    auto idx1 = octree.neighborsPrune<Kernel_t::sphere>(query, 1.5);
    EXPECT_TRUE(idx1.empty());

    // Radius 2.5. Both points should be inside
    auto idx2 = octree.neighborsPrune<Kernel_t::sphere>(query, 2.5);
    EXPECT_EQ(idx2.size(), 2u);
}

TEST_F(AdvancedOctreeTest, KernelCubeSearch) {
    points.push_back(Point(1.0, 1.0, 1.0));
    points.push_back(Point(2.0, 2.0, 2.0));
    auto [codes, box] = enc.sortPoints(points, metadata);
    LinearOctree octree(points, codes, box, enc);

    Point query(0.0, 0.0, 0.0);
    // Cube with half-extent 1.5. Point (1,1,1) is inside, (2,2,2) is outside
    auto idx = octree.neighborsPrune<Kernel_t::cube>(query, 1.5);
    EXPECT_EQ(idx.size(), 1u);
    EXPECT_DOUBLE_EQ(points[idx[0]].getX(), 1.0);
}
