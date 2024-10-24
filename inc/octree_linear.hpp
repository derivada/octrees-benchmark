/**
 * A linear (map-based) implementation of the Octree using Morton codes for quick access with good spacial locality
 * 
 * Pablo Díaz Viñambres 22/10/24
 * 
 * Some implementation details from https://geidav.wordpress.com/2014/08/18/advanced-octrees-2-node-representations/
 */


#pragma once

#include "octree_v2.hpp"
#include "Lpoint.hpp"

class LinearOctreeMap;

class LinearOctree : public OctreeV2 {
private:
    LinearOctreeMap* map;
    uint64_t code;
    uint8_t childMask = 0; // nth bit set to 1 if the nth child exists
public:
    LinearOctree();
    explicit LinearOctree(std::vector<Lpoint>& points);
    explicit LinearOctree(std::vector<Lpoint*>& points);
    LinearOctree(const Point& center, float radius);
    LinearOctree(Point center, float radius, std::vector<Lpoint*>& points);
    LinearOctree(Point center, float radius, std::vector<Lpoint>& points);

    [[nodiscard]] const OctreeV2* getOctant(int index) const override;
    [[nodiscard]] OctreeV2* getOctant(int index) override;
    
    void createOctants() override;
    void clearOctants() override;

    bool maxDepthReached() const override;

    // Friend declaration to allow LinearOctreeMap access
    friend class LinearOctreeMap;
};