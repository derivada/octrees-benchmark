/**
 * A simple container representing a single node in a Linear Octree, with its points and associated Morton code
 * 
 */
#pragma once
#include "Lpoint.hpp"

// Here we also define a couple shorthands for the integers used in morton encoding
using morton_t = uint_fast64_t;
using coords_t = uint_fast32_t;

class LinearOctreeOldNode {
    private:
        std::vector<Lpoint*> points;
        morton_t code;
        double center[3];
        double radii[3];
    public:
        LinearOctreeOldNode(std::vector<Lpoint*> p, morton_t c, uint8_t depth, coords_t x, coords_t y, coords_t z,
                        Point rootCenter, Vector rootRadii): points(p), code(c) {
            // Returns the physical (approximate) physical center of the node
            radii[0] = rootRadii.getX() * (1.0f / (1 << depth));
            radii[1] = rootRadii.getY() * (1.0f / (1 << depth));
            radii[2] = rootRadii.getZ() * (1.0f / (1 << depth));
            Point nodeRadii = Point(radii[0], radii[1], radii[2]);
            Point lowCorner = (rootCenter - rootRadii) + nodeRadii;
            center[0] = x * nodeRadii.getX() * 2 + lowCorner.getX();
            center[1] = y * nodeRadii.getY() * 2 + lowCorner.getY();
            center[2] = z * nodeRadii.getZ() * 2 + lowCorner.getZ();
        }

        // Geometric information getters
        inline Point getCenter() const {
            return Point(center[0], center[1], center[2]);
        }

        inline Vector getRadii() const {
            return Vector(radii[0], radii[1], radii[2]);
        }


        // Comparators based on Morton code
        bool operator==(const LinearOctreeOldNode& other) const {
            return this->code == other.code;
        }
        bool operator!=(const LinearOctreeOldNode& other) const {
            return !(*this == other);
        }
        bool operator<(const LinearOctreeOldNode& other) const {
            return this->code < other.code;
        }
        
        inline bool isLeaf() {
            return !points.empty();
        }

        // Since this is just a container, we can declare LinearOctree as a friend class
        friend class LinearOctreeOld;
};

