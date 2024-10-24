#include "octree_linear_map.hpp"

// Internal methods
LinearOctree* LinearOctreeMap::getNode(uint64_t locCode) {
    auto iter = nodes.find(locCode);
    return (iter == nodes.end() ? nullptr : &iter->second);
}
void LinearOctreeMap::replaceNode(uint64_t locCode, const LinearOctree& node) {
    nodes[locCode] = node;
}
void LinearOctreeMap::clearNode(uint64_t locCode) {
    nodes.erase(locCode);
}

// Children/parent access methods 
LinearOctree* LinearOctreeMap::getChild(uint64_t locCode, int index) {
    return getNode(getChildCode(locCode, index));
}
LinearOctree* LinearOctreeMap::getParent(uint64_t locCode) {
    if (getDepth(locCode) <= 1) return nullptr;
    return getNode(getParentCode(locCode));
}

void LinearOctreeMap::replaceChild(uint64_t locCode, int index, const LinearOctree& node) {
    replaceNode(getChildCode(locCode, index), node);
}
void LinearOctreeMap::replaceParent(uint64_t locCode, const LinearOctree& node) {
    replaceNode(getParentCode(locCode), node);
}
void LinearOctreeMap::clearChild(uint64_t locCode, int index) {
    clearNode(getChildCode(locCode, index));
}

// Static bitwise methods
uint64_t LinearOctreeMap::getParentCode(uint64_t locCode) {
    return locCode >> 3;
}
uint64_t LinearOctreeMap::getChildCode(uint64_t locCode, int index) {
    return (locCode << 3) | index;
}
size_t LinearOctreeMap::getDepth(uint64_t locCode) {
    if (!locCode) return 0;
    #if defined(__GNUC__)
    return (31 - __builtin_clz(locCode)) / 3;
    #else
    size_t depth = 0;
    for (; locCode != 1; locCode >>= 3, depth++);
    return depth;
    #endif
}
