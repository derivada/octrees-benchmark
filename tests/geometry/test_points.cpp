#include <gtest/gtest.h>
#include "geometry/point.hpp"
#include "geometry/point_containers.hpp"

TEST(PointTest, InitializationAndGetters) {
    Point p(1.5, 2.5, 3.5);
    EXPECT_DOUBLE_EQ(p.getX(), 1.5);
    EXPECT_DOUBLE_EQ(p.getY(), 2.5);
    EXPECT_DOUBLE_EQ(p.getZ(), 3.5);
}

TEST(PointTest, Distance3D) {
    Point p1(0.0, 0.0, 0.0);
    Point p2(3.0, 4.0, 0.0);
    // 3^2 + 4^2 = 9 + 16 = 25 -> sqrt = 5.0
    EXPECT_DOUBLE_EQ(p1.distance3D(p2), 5.0);
}

TEST(PointsAoSContainer, BasicOperations) {
    PointsAoS points;
    EXPECT_EQ(points.size(), 0);
    
    points.push_back(Point(1.0, 1.0, 1.0));
    points.push_back(Point(2.0, 2.0, 2.0));
    
    EXPECT_EQ(points.size(), 2);
    EXPECT_DOUBLE_EQ(points[0].getX(), 1.0);
    EXPECT_DOUBLE_EQ(points[1].getY(), 2.0);
}

TEST(PointsSoAContainer, BasicOperations) {
    PointsSoA points;
    EXPECT_EQ(points.size(), 0);
    
    points.push_back(Point(1.0, 2.0, 3.0));
    points.push_back(Point(4.0, 5.0, 6.0));
    
    EXPECT_EQ(points.size(), 2);
    
    // Check elements via array pointers
    EXPECT_DOUBLE_EQ(points.dataX()[0], 1.0);
    EXPECT_DOUBLE_EQ(points.dataY()[0], 2.0);
    EXPECT_DOUBLE_EQ(points.dataZ()[0], 3.0);
    
    EXPECT_DOUBLE_EQ(points.dataX()[1], 4.0);
    EXPECT_DOUBLE_EQ(points.dataY()[1], 5.0);
    EXPECT_DOUBLE_EQ(points.dataZ()[1], 6.0);
}
