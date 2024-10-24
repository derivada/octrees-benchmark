#pragma once
#include <unordered_map>
#include "octree_linear.hpp"

class LinearOctreeMap {
private:
    std::unordered_map<uint64_t, LinearOctree> nodes;

    LinearOctree* getNode(uint64_t locCode);
    void replaceNode(uint64_t locCode, const LinearOctree& node);
    void clearNode(uint64_t locCode);

public:
    static constexpr int MAX_DEPTH = 64 / 3;

    LinearOctreeMap() = default;
    ~LinearOctreeMap() = default;

    LinearOctree* getChild(uint64_t locCode, int index);
    LinearOctree* getParent(uint64_t locCode);
    void replaceChild(uint64_t locCode, int index, const LinearOctree& node);
    void replaceParent(uint64_t locCode, const LinearOctree& node);
    void clearChild(uint64_t locCode, int index);

    static uint64_t getParentCode(uint64_t locCode);
    static uint64_t getChildCode(uint64_t locCode, int index);
    static size_t getDepth(uint64_t locCode);
};