#include <gtest/gtest.h>
#include <vector>
#include <optional>

#include "geometry/point.hpp"
#include "geometry/point_containers.hpp"
#include "encoding/morton_encoder_3d.hpp"
#include "structures/linear_octree.hpp"

using namespace PointEncoding;

class LinearOctreeTest : public ::testing::Test {
protected:
    PointsAoS points;
    std::optional<std::vector<PointMetadata>> metadata;
    MortonEncoder3D enc;

    void SetUp() override {
        // Form a simple cloud with 8 points that clearly shows proximity
        points.push_back(Point(0.0, 0.0, 0.0));
        points.push_back(Point(0.1, 0.0, 0.0));
        points.push_back(Point(0.0, 0.1, 0.0));
        
        points.push_back(Point(10.0, 10.0, 10.0));
        points.push_back(Point(10.1, 10.0, 10.0));
        points.push_back(Point(10.0, 10.1, 10.0));
        
        points.push_back(Point(5.0, 5.0, 5.0));
        points.push_back(Point(5.0, 5.0, 5.1));
        
        metadata = std::nullopt;
    }
};

TEST_F(LinearOctreeTest, Construction) {
    auto [codes, box] = enc.sortPoints(points, metadata);
    ASSERT_NO_THROW({
        LinearOctree octree(points, codes, box, enc);
    });
}

TEST_F(LinearOctreeTest, KNNSearch) {
    auto [codes, box] = enc.sortPoints(points, metadata);
    LinearOctree octree(points, codes, box, enc);

    size_t k = 2;
    std::vector<size_t> indexes(k);
    std::vector<double> distances(k);
    
    // Search near the cluster around (10, 10, 10)
    Point searchPoint(9.9, 9.9, 9.9);
    octree.knn(searchPoint, k, indexes, distances);
    
    // The closest points should be (10,10,10) and (10.1,10,10) or (10,10.1,10)
    // We expect the distance squared of (10,10,10) to (9.9,9.9,9.9) to be approx:
    // 0.1^2 + 0.1^2 + 0.1^2 = 0.03
    EXPECT_NEAR(distances[0], 0.03, 1e-4);
    
    // Validate we found a point in that cluster
    EXPECT_GT(points[indexes[0]].getX(), 9.0);
}

TEST_F(LinearOctreeTest, RadiusSearch) {
    auto [codes, box] = enc.sortPoints(points, metadata);
    LinearOctree octree(points, codes, box, enc);

    // Search around (5,5,5) with radius 0.5 (radius squared = 0.25)
    Point searchPoint(5.0, 5.0, 5.0);
    double rSq = 0.25;
    
    // Use the basic radius search
    auto indexes = octree.neighborsPrune<Kernel_t::square>(searchPoint, rSq);
    
    // Expect to find exactly 2 points: (5,5,5) and (5,5,5.1)
    EXPECT_EQ(indexes.size(), 2);
    
    // Distances are not returned by neighborsPrune, we verify points correctness
    EXPECT_DOUBLE_EQ(points[indexes[0]].getZ(), 5.0);
    EXPECT_DOUBLE_EQ(points[indexes[1]].getZ(), 5.1);
}
