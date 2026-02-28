#include <gtest/gtest.h>
#include "structures/neighbor_set.hpp"
#include "geometry/point_containers.hpp"

TEST(NeighborSetTest, BasicUsage) {
    PointsAoS points;
    for (int i = 0; i < 10; ++i) {
        points.push_back(Point(i, i, i));
    }

    NeighborSet<PointsAoS> ns(&points);
    ns.addRange(0, 3); // Points 0, 1, 2
    ns.addRange(7, 10); // Points 7, 8, 9

    EXPECT_EQ(ns.size(), 6);
    EXPECT_FALSE(ns.empty());

    std::vector<size_t> indices;
    for (auto it = ns.begin(); it != ns.end(); ++it) {
        indices.push_back((*it).first);
    }

    ASSERT_EQ(indices.size(), 6);
    EXPECT_EQ(indices[0], 0);
    EXPECT_EQ(indices[1], 1);
    EXPECT_EQ(indices[2], 2);
    EXPECT_EQ(indices[3], 7);
    EXPECT_EQ(indices[4], 8);
    EXPECT_EQ(indices[5], 9);
}

TEST(NeighborSetTest, EmptySet) {
    PointsAoS points;
    NeighborSet<PointsAoS> ns(&points);
    EXPECT_TRUE(ns.empty());
    EXPECT_EQ(ns.size(), 0);
    EXPECT_TRUE(ns.begin() == ns.end());
}
