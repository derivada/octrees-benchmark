#include <gtest/gtest.h>
#include <vector>
#include <random>
#include <algorithm>

#include "geometry/point.hpp"
#include "geometry/point_containers.hpp"
#include "encoding/morton_encoder_3d.hpp"
#include "encoding/hilbert_encoder_3d.hpp"
#include "structures/linear_octree.hpp"
#include "kernels/kernel_factory.hpp"

using namespace PointEncoding;

class ComplexTest : public ::testing::Test {
protected:
    MortonEncoder3D morton;
    HilbertEncoder3D hilbert;
    std::optional<std::vector<PointMetadata>> metadata = std::nullopt;

    // Helper for brute force radius search
    std::vector<size_t> bruteForceRadius(const PointsAoS& points, const Point& center, double radius) {
        std::vector<size_t> result;
        double rSq = radius * radius;
        for (size_t i = 0; i < points.size(); ++i) {
            double dx = points[i].getX() - center.getX();
            double dy = points[i].getY() - center.getY();
            double dz = points[i].getZ() - center.getZ();
            if (dx*dx + dy*dy + dz*dz <= rSq) {
                result.push_back(i);
            }
        }
        return result;
    }
};

TEST_F(ComplexTest, DegenerateLine) {
    PointsAoS points;
    for (int i = 0; i < 100; ++i) {
        points.push_back(Point(i * 0.1, 0.0, 0.0));
    }

    auto sortRes = morton.sortPoints(points, metadata);
    auto codes = sortRes.first;
    auto box = sortRes.second;
    LinearOctree octree(points, codes, box, morton);

    Point searchPoint(5.0, 0.0, 0.0);
    double radius = 0.5;
    auto results = octree.neighborsPrune<Kernel_t::sphere>(searchPoint, radius);
    
    auto expected = bruteForceRadius(points, searchPoint, radius);
    EXPECT_EQ(results.size(), expected.size());
}

TEST_F(ComplexTest, DegeneratePlane) {
    PointsAoS points;
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 10; ++j) {
            points.push_back(Point(i * 0.1, j * 0.1, 0.0));
        }
    }

    auto sortRes = morton.sortPoints(points, metadata);
    auto codes = sortRes.first;
    auto box = sortRes.second;
    LinearOctree octree(points, codes, box, morton);

    Point searchPoint(0.5, 0.5, 0.0);
    double radius = 0.2;
    auto results = octree.neighborsPrune<Kernel_t::sphere>(searchPoint, radius);
    
    auto expected = bruteForceRadius(points, searchPoint, radius);
    EXPECT_EQ(results.size(), expected.size());
}

TEST_F(ComplexTest, LargeCoordinates) {
    PointsAoS points;
    double offset = 1e6;
    points.push_back(Point(offset, offset, offset));
    points.push_back(Point(offset + 0.1, offset, offset));
    points.push_back(Point(offset, offset + 0.1, offset));

    auto sortRes = morton.sortPoints(points, metadata);
    auto codes = sortRes.first;
    auto box = sortRes.second;
    LinearOctree octree(points, codes, box, morton);

    Point searchPoint(offset + 0.05, offset + 0.05, offset);
    double radius = 0.1; 
    auto results = octree.neighborsPrune<Kernel_t::sphere>(searchPoint, radius);
    
    EXPECT_EQ(results.size(), 3);
}

TEST_F(ComplexTest, EncoderDecoderConsistency) {
    Point p(1.23, 4.56, 7.89);
    
    PointsAoS dummy;
    dummy.push_back(Point(0,0,0));
    dummy.push_back(Point(10,10,10));
    morton.sortPoints(dummy, metadata);

    const PointEncoding::key_t c1 = morton.encode(1000, 2000, 3000);
    PointEncoding::coords_t x_dec, y_dec, z_dec;
    morton.decode(c1, x_dec, y_dec, z_dec);
    
    EXPECT_EQ(x_dec, 1000);
    EXPECT_EQ(y_dec, 2000);
    EXPECT_EQ(z_dec, 3000);

    const PointEncoding::key_t c2 = morton.encode(x_dec, y_dec, z_dec);
    EXPECT_EQ(c1, c2);
}

TEST_F(ComplexTest, ApproxSearchBounds) {
    PointsAoS points;
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(0, 10);
    for (int i = 0; i < 1000; ++i) {
        points.push_back(Point(dis(gen), dis(gen), dis(gen)));
    }

    auto sortRes = morton.sortPoints(points, metadata);
    auto codes = sortRes.first;
    auto box = sortRes.second;
    LinearOctree octree(points, codes, box, morton);

    Point searchPoint(5.0, 5.0, 5.0);
    double radius = 2.0;

    auto exact = octree.neighborsStruct<Kernel_t::sphere>(searchPoint, radius);
    
    PointEncoder* encPtr = &morton;
    uint32_t precision = encPtr->maxDepth() / 2;
    
    auto approxUpper = octree.neighborsApprox<Kernel_t::sphere>(searchPoint, radius, precision, true);
    auto approxLower = octree.neighborsApprox<Kernel_t::sphere>(searchPoint, radius, precision, false);
    
    EXPECT_GE(approxUpper.size(), exact.size());
    EXPECT_LE(approxLower.size(), exact.size());
}

TEST_F(ComplexTest, KNNConsistency) {
    PointsAoS points;
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(0, 10);
    for (int i = 0; i < 500; ++i) {
        points.push_back(Point(dis(gen), dis(gen), dis(gen)));
    }

    auto sortRes = morton.sortPoints(points, metadata);
    auto codes = sortRes.first;
    auto box = sortRes.second;
    LinearOctree octree(points, codes, box, morton);

    Point searchPoint(5.0, 5.0, 5.0);
    size_t k = 10;
    std::vector<size_t> indexes(k);
    std::vector<double> distances(k);
    octree.knn(searchPoint, k, indexes, distances);

    for (size_t i = 1; i < k; ++i) {
        EXPECT_GE(distances[i], distances[i-1]);
    }

    for (size_t i = 0; i < k; ++i) {
        double dx = points[indexes[i]].getX() - searchPoint.getX();
        double dy = points[indexes[i]].getY() - searchPoint.getY();
        double dz = points[indexes[i]].getZ() - searchPoint.getZ();
        EXPECT_NEAR(distances[i], dx*dx + dy*dy + dz*dz, 1e-7);
    }
}
