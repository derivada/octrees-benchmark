#include <gtest/gtest.h>
#include "geometry/point.hpp"
#include "geometry/box.hpp"
#include "geometry/point_containers.hpp"

TEST(BoxTest, Initialization) {
    Point center(5.0, 5.0, 5.0);
    Vector radii(2.0, 2.0, 2.0);
    Box box(center, radii);

    EXPECT_DOUBLE_EQ(box.center().getX(), 5.0);
    EXPECT_DOUBLE_EQ(box.minX(), 3.0);
    EXPECT_DOUBLE_EQ(box.maxX(), 7.0);
}

TEST(BoxTest, IsInside) {
    Point center(0.0, 0.0, 0.0);
    Vector radii(1.0, 1.0, 1.0);
    Box box(center, radii);

    EXPECT_TRUE(box.isInside(Point(0.5, 0.5, 0.5)));
    EXPECT_TRUE(box.isInside(Point(-0.5, -0.5, -0.5)));
    EXPECT_FALSE(box.isInside(Point(1.5, 0.0, 0.0)));
    EXPECT_FALSE(box.isInside(Point(0.0, 1.1, 0.0)));
}

TEST(BoxTest, MBB) {
    PointsAoS points;
    points.push_back(Point(0.0, 0.0, 0.0));
    points.push_back(Point(10.0, 10.0, 10.0));
    points.push_back(Point(5.0, 5.0, 5.0));

    Vector maxRadius;
    Point center = mbb(points, maxRadius);

    EXPECT_DOUBLE_EQ(center.getX(), 5.0);
    EXPECT_DOUBLE_EQ(center.getY(), 5.0);
    EXPECT_DOUBLE_EQ(center.getZ(), 5.0);
    EXPECT_DOUBLE_EQ(maxRadius.getX(), 5.0);
}

TEST(BoxTest, Inside2D3D) {
    Point p(5.0, 5.0, 0.0);
    Vector min(0.0, 0.0, 0.0);
    Vector max(10.0, 10.0, 10.0);

    EXPECT_TRUE(insideBox2D(p, min, max));
    EXPECT_TRUE(insideBox3D(Point(5,5,5), min, max));
    EXPECT_FALSE(insideBox3D(Point(5,5,15), min, max));
}
