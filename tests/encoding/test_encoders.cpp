#include <gtest/gtest.h>
#include <vector>
#include <optional>

#include "geometry/point.hpp"
#include "geometry/point_containers.hpp"
#include "geometry/point_metadata.hpp"
#include "encoding/morton_encoder_3d.hpp"
#include "encoding/morton_encoder_2d.hpp"
#include "encoding/hilbert_encoder_3d.hpp"
#include "encoding/hilbert_encoder_2d.hpp"

using namespace PointEncoding;

class EncodersTest : public ::testing::Test {
protected:
    PointsAoS points;
    std::optional<std::vector<PointMetadata>> metadata;

    void SetUp() override {
        points.push_back(Point(1.0, 1.0, 1.0));
        points.push_back(Point(9.0, 9.0, 9.0)); 
        points.push_back(Point(5.0, 5.0, 5.0));
        points.push_back(Point(1.0, 9.0, 1.0));
        metadata = std::nullopt;
    }
};

TEST_F(EncodersTest, MortonSorting) {
    MortonEncoder3D morton;
    auto [codes, box] = morton.sortPoints(points, metadata);
    
    EXPECT_EQ(codes.size(), 4);
    EXPECT_EQ(points.size(), 4);
    
    // The point at (1,1,1) should be sorted first than (9,9,9)
    // Points are reordered in-place during sort
    EXPECT_DOUBLE_EQ(points[0].getX(), 1.0);
    EXPECT_DOUBLE_EQ(points[0].getY(), 1.0);
    EXPECT_DOUBLE_EQ(points[0].getZ(), 1.0);
    
    EXPECT_DOUBLE_EQ(points[3].getX(), 9.0);
    EXPECT_DOUBLE_EQ(points[3].getY(), 9.0);
    EXPECT_DOUBLE_EQ(points[3].getZ(), 9.0);

    // Box bounds testing
    EXPECT_DOUBLE_EQ(box.minX(), 1.0);
    EXPECT_DOUBLE_EQ(box.maxX(), 9.0);
    EXPECT_DOUBLE_EQ(box.minY(), 1.0);
    EXPECT_DOUBLE_EQ(box.maxY(), 9.0);
}

TEST_F(EncodersTest, HilbertSorting) {
    HilbertEncoder3D hilbert;
    auto [codes, box] = hilbert.sortPoints(points, metadata);
    
    EXPECT_EQ(codes.size(), 4);
    EXPECT_EQ(points.size(), 4);

    // Bounding Box correctness
    EXPECT_DOUBLE_EQ(box.minX(), 1.0);
    EXPECT_DOUBLE_EQ(box.maxX(), 9.0);
    EXPECT_DOUBLE_EQ(box.minY(), 1.0);
    EXPECT_DOUBLE_EQ(box.maxY(), 9.0);
}

TEST_F(EncodersTest, Morton2DSorting) {
    MortonEncoder2D morton(2); // Z-axis projection (XY plane)
    auto [codes, box] = morton.sortPoints(points, metadata);
    
    EXPECT_EQ(codes.size(), 4);
    // Box bounds should still be 3D since it uses mbb
    EXPECT_DOUBLE_EQ(box.minX(), 1.0);
    EXPECT_DOUBLE_EQ(box.maxX(), 9.0);
}

TEST_F(EncodersTest, Hilbert2DSorting) {
    HilbertEncoder2D hilbert(2); // Z-axis projection
    auto [codes, box] = hilbert.sortPoints(points, metadata);
    
    EXPECT_EQ(codes.size(), 4);
    EXPECT_DOUBLE_EQ(box.minX(), 1.0);
    EXPECT_DOUBLE_EQ(box.maxX(), 9.0);
}
