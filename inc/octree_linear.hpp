#pragma once

#include "Lpoint.hpp"
#include "Box.hpp"
#include <stack>
#include <bitset>
#include <unordered_map>
#include <unordered_set>
#include <filesystem>
#include "libmorton/morton.h"
#include "NeighborKernels/KernelFactory.hpp"
#include "TimeWatcher.hpp"
#include "morton_encoder.hpp"
#include "Box.hpp"

/**
* @class LinearOctree
* 
* @brief Another (more correct) linear octree based on the excellent implementation done 
* for the cornerstone octree project: https://github.com/sekelle/cornerstone-octree/tree/master
* 
* @details This linear octree is built by storing offsets to the positions of an array of points sorted by their morton codes. 
* For each leaf of the octree, there is an element in this array listthat points to the index of the first point in that leaf.
* Since the points are sorted, the next element on the array - 1 contains the index of the last point in that leaf.
* 
* @cite Keller et al. Cornerstone: Octree Construction Algorithms for Scalable Particle Simulations. https://arxiv.org/pdf/2307.06345
* 
* @authors Pablo Díaz Viñambres 
* 
* @date 16/11/2024
* 
*/
class LinearOctree {
private:
    /// @brief The maximum number of points in a leaf
    static constexpr unsigned int MAX_POINTS        = 128;

    /// @brief The minimum octant radius to have in a leaf (TODO: this is still not implemented, and may not be needed)
	static constexpr float        MIN_OCTANT_RADIUS = 0.1;

	/// @brief The default size of the search set in KNN
	static constexpr size_t       DEFAULT_KNN       = 100;

	/// @brief The number of octants per internal node
	static constexpr short        OCTANTS_PER_NODE  = 8;

    /// @brief Number of leaves and internal nodes in the octree. Equal to size of the leaves vector - 1.
    uint32_t nLeaf;

    /// @brief Number of internal nodes in the octree. Equal to (nLeaf-1) / 7.
    uint32_t nInternal;

    /// @brief Total number of nodes in the octree. Equal to nLeaf + nInternal.
    uint32_t nTotal;

    /**
     * @brief The leaves of the octree in cornerstone array format.
     * @details This array contains morton codes (interpreted here as octal digit numbers) satisfying certain constraints:
     * 1. The length of the array is nLeaf + 1
     * 2. The first element is 0 and the last element 8^MAX_DEPTH, where MAX_DEPTH is the maximum depth of the encoding system 
     * (i.e. an upper bound for every possible encoding of a point)
     * 3. The array is sorted in increasing order and the distance between two consecutive elements is 8^l, where l is less or equal
     * to MAX_DEPTH
     * 
     * The array is initialized to {0, 8^MAX_DEPTH} and then subdivided into 8 equally sized bins if the number of points with encoding
     * between two leaves is greater than MAX_POINTS.
     * 
     * For more details about the construction, check the cornerstone paper, section 4.
     */
    std::vector<morton_t> leaves; 

    /// @brief  This array contains how many points have an encoding with a value between two of the leaves
    std::vector<uint32_t> counts;

    /// @brief This array is simply an exclusive scan of counts, and marks the index of the first point for a leaf
    std::vector<uint32_t> layout;

    /**
     * @brief The Warren-Salmon encoding of each node in the octree
     * @details For a given (internal or leaf) node, we store its position on the octree using this array, the position for a node at depth
     * n will be given by 0 000 000 ... 1 x1y1z1 ... xnynzn. This allows for traversals needed in neighbourhood search.
     * 
     * The process to obtain this array and link it with the leaves array is detailed in the cornerstone paper, section 5.
     */
    std::vector<morton_t> prefixes;

    /// @brief Index of the first child of each node (if 0 we have a leaf)
    std::vector<uint32_t> offsets;

    /// @brief The parent index of every group of 8 sibling nodes
    std::vector<uint32_t> parents; // TODO: this may not be needed

    /// @brief First node index of every tree level (L+2 elements where L is MAX_DEPTH)
    std::vector<uint32_t> levelRange = std::vector<uint32_t>(MortonEncoder::MAX_DEPTH + 2);

    /// @brief A map between the internal representation at offsets and the one in cornerstone format in leaves
    std::vector<int32_t> internalToLeaf;

    /// @brief The reverse mapping of internalToLeaf
    std::vector<int32_t> leafToInternal;

    /**
     * @brief A reference to the array of points that we sort
     * @details At the beginning of the octree construction, this points are encoded and then sorted in-place in the order given by their
     * encodings. Therefore, this array is altered inside this class. This is done to  locality that Morton/Hilbert
     */
    std::vector<Lpoint> &points;

    /// @brief The encodings of the points in the octree
    std::vector<morton_t> codes;

    /// @brief The center points of each node in the octree
    std::vector<Point> centers;

    /// @brief The vector of radii of each node in the octree
    std::vector<Vector> radii;

    /// @brief The global bounding box of the octree
    Box bbox = Box(Point(), Vector());

    /// @brief A simple vector containinf the radii of each level in the octree to speed up computations.
    Vector precomputedRadii[MortonEncoder::MAX_DEPTH + 1];

    /// @brief A vector containing the half-lengths of the minimum measure of the encoding.
    float halfLengths[3];
    
    /**
     * @brief Computes the rebalacing decisions as the first process in the subdivision of the leaves array
     * 
     * @details This function implements g1 in the cornerstone paper, for each leaf we calculate the operation
     * that decides whether we merge, split or leave unchanged the leaf.
     * 
     * @param nodeOps The output array of decisions, of length leaves.size()-1
     */
    bool rebalanceDecision(std::vector<uint32_t> &nodeOps) {
        bool converged = true;
        for(int i = 0; i<leaves.size()-1; i++) {
            nodeOps[i] = calculateNodeOp(i);
            if(nodeOps[i] != 1) converged = false;
        }
        return converged;
    } 

    /**
     * @brief Computes the operation on the leaf marked by index
     * 
     * @details The following values can be returned:
     * - 1 if the leaf should remain unchanged
     * - 0 if the leaf should be merged (only if the counts of all of its siblings are <= MAX_POINTS), the siblins
     * are all next to each other and the node is not the first sibling 
     * (because the first sibling is the node that will stay after merge)
     * - 8^L where L goes up to 4, if we need to split the node L times (recursively)
     * 
     * @param index The leaf array index
     */
    uint32_t calculateNodeOp(uint32_t index) {
        auto [sibling, level] = siblingAndLevel(index);

        if(sibling > 0) {
            // We have 8 siblings next to each other, could merge this node if the count of all siblings is less MAX_COUNT
            uint32_t parentIndex = index - sibling;
            // Should not be bigger than 2^32
            size_t parentCount =    counts[parentIndex]   + counts[parentIndex+1]+ 
                                    counts[parentIndex+2] + counts[parentIndex+3]+ 
                                    counts[parentIndex+4] + counts[parentIndex+5]+
                                    counts[parentIndex+6] + counts[parentIndex+7];
            if(parentCount <= MAX_POINTS)
                return 0; // merge
        }
        
        uint32_t nodeCount = counts[index];
        // Decide if we split or not
        // TODO: check MIN_OCTANT_RADIUS
        if (nodeCount > MAX_POINTS * 512 && level + 3 < MortonEncoder::MAX_DEPTH) { return 4096; } // split into 4 layers
        if (nodeCount > MAX_POINTS * 64 && level + 2 < MortonEncoder::MAX_DEPTH) { return 512; }   // split into 3 layers
        if (nodeCount > MAX_POINTS * 8 && level + 1 < MortonEncoder::MAX_DEPTH) { return 64; }     // split into 2 layers
        if (nodeCount > MAX_POINTS && level < MortonEncoder::MAX_DEPTH ) { return 8; } // split into 1 layer
        
        return 1; // dont do anything
    }

    /**
     * @brief Compute the sibling and level of the node in the octree
     * 
     * @details Will return -1 for the sibling if the nodes are not next to each other or if 
     * the level is 0.
     * 
     * @param index The leaf index to find
     */
    inline std::pair<int32_t, uint32_t> siblingAndLevel(uint32_t index) {
        morton_t node = leaves[index];
        morton_t range = leaves[index+1] - node;
        uint32_t level = MortonEncoder::getLevel(range);
        if(level == 0) {
            return {-1, level};
        }

        uint32_t siblingId = MortonEncoder::getSiblingId(node, level);

        // Checks if all siblings are on the tree, to do this, checks if the difference between the two parent nodes corresponding
        // to the code parent and the next parent is the range spanned by two consecutive codes at that level
        bool siblingsOnTree = leaves[index - siblingId + 8] == (leaves[index - siblingId] + MortonEncoder::nodeRange(level - 1));
        if(!siblingsOnTree) siblingId = -1;

        return {siblingId, level};
    }

    /// @brief  Print the bits of the given code in groups of 3 to represent each level
    static void printMortonCode(morton_t code) {
        //
        std::cout << std::bitset<1>(code >> 63) << " ";
        for (int i = 62; i >= 0; i -= 3) {
            std::cout << std::bitset<3>((code >> (i - 2)) & 0b111) << " ";
        }
        std::cout << std::endl;
    }

    /**
     * @brief Computes a new stage of the leaves array by subdividing using the operations given
     * 
     * @details This function implements steps g2 and g3 of the subdivision process
     * 
     * @param newLeaves The new leaves array that will be swapped with the current one after this function execution
     * @param nodeOps The array of operations performed in the first step (g1)
     */
    void rebalanceTree(std::vector<morton_t> &newLeaves, std::vector<uint32_t> &nodeOps) {
        uint32_t n = leaves.size() - 1;

        // Exclusive scan, step g2
        exclusiveScan(nodeOps.data(), n+1);

        // Initialization of the new leafs array
        newLeaves.resize(nodeOps[n] + 1);
        newLeaves.back() = leaves.back();

        // Compute the operations
        for (uint32_t i = 0; i < n; ++i) {
            processNode(i, nodeOps, newLeaves);
        }
    }

    /**
     * @brief Construct the corresponding new indexes of newTree in place
     * 
     * @details Sometimes more than one index is constructed, when nodeOps = 8 or higher, the values are
     * put for all the new siblings that the leaves array subdivides into. This function implements
     * step g3 for each element.
     * 
     * @param index The index of the original tree to be subdivided
     * @param nodeOps The operation to be performed on the index
     * @param newLeaves The new leaves array that will be swapped with the current one after this function execution
     */
    void processNode(uint32_t index, std::vector<uint32_t> &nodeOps, std::vector<morton_t> &newLeaves) {
        // The original value of the opCode (before exclusive scan)
        uint32_t opCode = nodeOps[index + 1] - nodeOps[index]; 
        if(opCode == 0)
            return;
    
        morton_t node = leaves[index];
        morton_t range = leaves[index+1] - node;
        uint32_t level = MortonEncoder::getLevel(range);

        // The new position to put the node into (nodeOps value after exclusive scan)
        uint32_t newNodeIndex = nodeOps[index]; 

        // Copy the old node into the new position
        newLeaves[newNodeIndex] = node;
        if(opCode > 1) {
            // Split the node into 8^L as marked by the opCode, add the adequate codes to the new leaves
            uint32_t levelDiff = MortonEncoder::log8ceil(opCode);
            morton_t gap = MortonEncoder::nodeRange(level + levelDiff);
            for (uint32_t sibling = 1; sibling < opCode; sibling++) {
                newLeaves[newNodeIndex + sibling] = newLeaves[newNodeIndex + sibling - 1] + gap;
            }
        }
    }

    /**
     * @brief Count number of particles in each leafd
     * 
     * @details This functions counts how many particles have encodings at leaf i, that is
     * between leaves[i] and leaves[i+1]
     */
    void computeNodeCounts() {
        uint32_t n = leaves.size() - 1;
        uint32_t codes_size = codes.size();
        uint32_t firstNode = 0;
        uint32_t lastNode = n;

        // Find general bounds for the codes array
        if(codes.size() > 0) {
            firstNode = std::upper_bound(leaves.begin(), leaves.end(), codes[0]) - leaves.begin() - 1;
            lastNode = std::upper_bound(leaves.begin(), leaves.end(), codes[codes_size-1]) - leaves.begin();
            assert(firstNode <= lastNode);
        } else {
            firstNode = n, lastNode = n;
        }

        // Fill non-populated parts of the octree with zeros
        for(uint32_t i = 0; i<firstNode; i++)
            counts[i] = 0;
        for(uint32_t i = lastNode; i<lastNode; i++)
            counts[i] = 0;

        // TODO: Can try using the count guessing algorithm provided in cornerstone code
        size_t nNonZeroNodes = lastNode - firstNode;
        for(uint32_t i = 0; i<nNonZeroNodes; i++) {
            counts[i + firstNode] = calculateNodeCount(leaves[i+firstNode], leaves[i+firstNode+1]);
        }
    }

    /// @brief Since the codes array is sorted, we can use binary search to accelerate the counts computation
    uint32_t calculateNodeCount(morton_t keyStart, morton_t keyEnd) {
        auto rangeStart = std::lower_bound(codes.begin(), codes.end(), keyStart);
        auto rangeEnd   = std::lower_bound(codes.begin(), codes.end(), keyEnd);
        return rangeEnd - rangeStart;
    }

    /// @brief Simple serial implementation of an exclusive scan
    template<class T>
    void exclusiveScan(T* out, size_t numElements) {
        T a = T(0);
        T b = T(0);
        for (size_t i = 0; i < numElements; ++i) {
            a += out[i];
            out[i] = b;
            b = a;
        }
    }

    constexpr uint32_t binaryKeyWeight(morton_t key, unsigned level){
        uint32_t ret = 0;
        for (uint32_t l = 1; l <= level + 1; ++l)
        {
            uint32_t digit = MortonEncoder::octalDigit(key, l);
            ret += digitWeight(digit);
        }
        return ret;
    }

    constexpr int32_t digitWeight(uint32_t digit) {
        int32_t fourGeqMask = -int32_t(digit >= 4);
        return ((7 - digit) & fourGeqMask) - (digit & ~fourGeqMask);
    }

    void createUnsortedLayout() {
        // Create the prefixesand internaltoleaf arrays for the leafs
        for(int i = 0; i<nLeaf; i++) {
            morton_t key = leaves[i];
            uint32_t level = MortonEncoder::getLevel(leaves[i+1] - key);
            prefixes[i + nInternal] = MortonEncoder::encodePlaceholderBit(key, 3*level);
            internalToLeaf[i + nInternal] = i + nInternal;

            uint32_t prefixLength = MortonEncoder::commonPrefix(key, leaves[i+1]);
            if(prefixLength % 3 == 0 && i < nLeaf - 1) {
                uint32_t octIndex = (i + binaryKeyWeight(key, prefixLength / 3)) / 7;
                prefixes[octIndex] = MortonEncoder::encodePlaceholderBit(key, prefixLength);
                internalToLeaf[octIndex] = octIndex;
            }
        }
    }

    // Determine octree subdivision level boundaries
    void getLevelRange() {
        for(uint32_t level = 0; level <= MortonEncoder::MAX_DEPTH; level++) {
            auto it = std::lower_bound(prefixes.begin(), prefixes.end(), MortonEncoder::encodePlaceholderBit(0, 3 * level));
            levelRange[level] = std::distance(prefixes.begin(), it);
        }
        levelRange[MortonEncoder::MAX_DEPTH + 1] = nTotal;
    }

    // Extract parent/child relationships from binary tree and translate to sorted order
    void linkTree() {
        for(int i = 0; i<nInternal; i++) {
            uint32_t idxA = leafToInternal[i];
            morton_t prefix = prefixes[idxA];
            morton_t nodeKey = MortonEncoder::decodePlaceholderBit(prefix);
            unsigned prefixLength = MortonEncoder::decodePrefixLength(prefix);
            unsigned level = prefixLength / 3;
            assert(level < MortonEncoder::MAX_DEPTH);

            morton_t childPrefix = MortonEncoder::encodePlaceholderBit(nodeKey, prefixLength + 3);

            uint32_t leafSearchStart = levelRange[level + 1];
            uint32_t leafSearchEnd   = levelRange[level + 2];
            uint32_t childIdx = std::distance(prefixes.begin(), 
                std::lower_bound(prefixes.begin() + leafSearchStart, prefixes.begin() + leafSearchEnd, childPrefix));

            if (childIdx != leafSearchEnd && childPrefix == prefixes[childIdx]) {
                offsets[idxA] = childIdx;
                // We only store the parent once for every group of 8 siblings.
                // This works as long as each node always has 8 siblings.
                // Subtract one because the root has no siblings.
                parents[(childIdx - 1) / 8] = idxA;
            }
        } 
    }

public:
    LinearOctree() = default;
    
    /**
     * @brief Builds the linear octree given an array of points, also reporting how much time each step takes
     * 
     * @details The points will be sorted in-place by the order given by the encoding to allow
     * spatial data locality
     */
    explicit LinearOctree(std::vector<Lpoint> &points): points(points) {
        std::cout << "Linear octree build summary:\n";
        double total_time = 0.0;
        TimeWatcher tw;
        auto buildStep = [&](auto &&step, const std::string action) {
            tw.start();
            step();
            tw.stop();
            total_time += tw.getElapsedDecimalSeconds();
            std::cout << "  Time to " << action << ": " << tw.getElapsedDecimalSeconds() << " seconds\n";
        };

        buildStep([&] { setupBbox(); }, "find bounding box");
        buildStep([&] { sortPoints(); }, "sort the points by their Morton codes");
        buildStep([&] { buildOctreeLeaves(); }, "build the octree leaves");
        buildStep([&] { resize(); }, "allocate space for internal variables");
        buildStep([&] { buildOctreeInternal(); }, "build internal part of the octree and link it");
        buildStep([&] { computeGeometry(); }, "compute octree geometry");

        std::cout << "Total time to build linear octree: " << total_time << " seconds\n";
    }

    /**
     * @brief Computes essential geometric information about the octree
     * 
     * @details This function computes tree things:
     * 1. Global bounding box of the octree
     * 2. Compute the half-lengths vector that indicates how much we displace in the physical step for
     * each unit of the morton encoded integer coordinates
     * 3. Precomputes radii for all the possible levels
     */
    void setupBbox() {
        Vector radii;
        Point center = mbb(points, radii);
        bbox = Box(center, radii);

        // Compute the physical half lengths for multiplying with the encoded coordinates
        halfLengths[0] = 0.5f * MortonEncoder::EPS * (bbox.maxX() - bbox.minX());
        halfLengths[1] = 0.5f * MortonEncoder::EPS * (bbox.maxY() - bbox.minY());
        halfLengths[2] = 0.5f * MortonEncoder::EPS * (bbox.maxZ() - bbox.minZ());

        for(int i = 0; i<= MortonEncoder::MAX_DEPTH; i++) {
            coords_t sideLength = (1u << (MortonEncoder::MAX_DEPTH - i));
            precomputedRadii[i] = Vector(
                sideLength * halfLengths[0],
                sideLength * halfLengths[1],
                sideLength * halfLengths[2]
            );
        }
    }

    /**
     * @brief This function computes the morton encodings of the points and sorts them in
     * the given order
     * 
     * @details The points array is changed after this step
     */
    void sortPoints() {
        // Temporal vector of pairs
        std::vector<std::pair<morton_t, Lpoint>> encoded_points;
        encoded_points.reserve(points.size());
        for(size_t i = 0; i < points.size(); i++) {
            encoded_points.emplace_back(MortonEncoder::encodeMortonPoint(points[i], bbox), points[i]);
        }

        // TODO: implement parallel radix sort
        std::sort(encoded_points.begin(), encoded_points.end(),
            [](const auto& a, const auto& b) {
                return a.first < b.first;  // Compare only the morton codes
        });
        
        // Copy back sorted codes and points
        codes.resize(points.size());
        for(size_t i = 0; i < points.size(); i++) {
            codes[i] = encoded_points[i].first;
            points[i] = encoded_points[i].second;
        }
    }

    /// @brief Builds the octeee leaves array by repeatingly calling @ref updateOctreeLeaves()
    void buildOctreeLeaves() {
        // Builds the octree sequentially using the cornerstone algorithm

        // We start with 0, 7777...777 (in octal)
        leaves = {0, MortonEncoder::UPPER_BOUND};
        counts = {(uint32_t) codes.size()};

        while(!updateOctreeLeaves())
            ;
        

    }

    /**
     * @brief Computes the node operations to be done on the leaves and modifies the tree if necessary
     * 
     * @details Convergence is achieved when all the node operations to be done are equal to 1
     */
    bool updateOctreeLeaves() {
        std::vector<uint32_t> nodeOps(leaves.size());
        bool converged = rebalanceDecision(nodeOps);
        if(!converged) {
            std::vector<morton_t> newTree;
            rebalanceTree(newTree, nodeOps);
            counts.resize(newTree.size()-1);
            swap(leaves, newTree);

            computeNodeCounts();
        }
        return converged;
    }

    void resize() {
        // Compute the final sizes of the octree
        nLeaf = leaves.size() - 1;
        nInternal = (nLeaf - 1) / 7;
        nTotal = nLeaf + nInternal;

        // Resize the other fields
        prefixes.resize(nTotal);
        offsets.resize(nTotal+1);
        parents.resize((nTotal-1) / 8);
        internalToLeaf.resize(nTotal);
        leafToInternal.resize(nTotal);
        centers.resize(nTotal);
        radii.resize(nTotal);
        layout.resize(leaves.size());

        // Perform the exclusive scan to get the layout indices (first index in the codes for each leaf)
        std::exclusive_scan(counts.begin(), counts.end() + 1, layout.begin(), 0);
    }

    /**
     * @brief Builds the internal part of the octree and links the nodes
     * 
     * @details Follows the process indicated in the cornerstone octree paper, section 5. 
     */
    void buildOctreeInternal() {
        createUnsortedLayout();
        // Sort by key where the keys are the prefixes and the values to sort internalToLeaf
        std::vector<std::pair<morton_t, uint32_t>> prefixes_internalToLeaf(nTotal);
        for(int i = 0; i<nTotal; i++) {
            prefixes_internalToLeaf[i] = {prefixes[i], internalToLeaf[i]};
        }
        std::stable_sort(prefixes_internalToLeaf.begin(), prefixes_internalToLeaf.end(), [](const auto &t1, const auto &t2) {
            return t1.first < t2.first;
        });

        for(int i = 0; i<nTotal; i++) {
            prefixes[i] = prefixes_internalToLeaf[i].first;
            internalToLeaf[i] = prefixes_internalToLeaf[i].second;
        }

        // Compute the reverse mapping leafToInternal
        for (uint32_t i = 0; i < nTotal; ++i) {
            leafToInternal[internalToLeaf[i]] = i;
        }

        // Offset by the number of internal nodes
        for (uint32_t i = 0; i < nTotal; ++i) {
            internalToLeaf[i] -= nInternal;
        }

        // Find the LO array
        getLevelRange();

        // Clear child offsets
        std::fill(offsets.begin(), offsets.end(), 0);

        // Compute the links
        linkTree();
    }

    /**
     * @brief Computes the octree geometry (the centers and radii of each internal node and leaf)
     * 
     * @details We do this to allow for faster traversals, however this is not strictly necessary
     * and could be removed if memory becomes a constraint. This stuff can be computed on the fly
     * during neighbourhood searches.
     */
    void computeGeometry() {
        for(uint32_t i = 0; i<prefixes.size(); i++) {
            morton_t prefix = prefixes[i];
            morton_t startKey = MortonEncoder::decodePlaceholderBit(prefix);
            uint32_t level = MortonEncoder::decodePrefixLength(prefix) / 3;
            std::tie(centers[i], radii[i]) = 
                MortonEncoder::getCenterAndRadii(startKey, level, bbox, halfLengths, precomputedRadii);
        }
    }

    /**
     * @brief Traverse the octree in a single pass
     * 
     * @details This function is used to traverse the octree in a single pass, calling the continuationCriterion
     * function to decide whether to descend into a node or not, and the endpointAction function to perform an action
     * when a leaf node is reached.
     * 
     * @param continuationCriterion A function that takes the index of an internal node indicates when to prune the tree during the search
     * @param endpointAction A function that takes the index of a leaf node and computes an action over it
     */
    template<class C, class A>
    void singleTraversal(C&& continuationCriterion, A&& endpointAction) const {
        bool descend = continuationCriterion(0);
        if (!descend) return;

        if (offsets[0] == 0) {
            // Root node is already a leaf
            endpointAction(0);
            return;
        }

        uint32_t stack[128];
        stack[0] = 0;

        uint32_t stackPos = 1;
        uint32_t node = 0; // Start at the root

        do {
            for (int octant = 0; octant < OCTANTS_PER_NODE; ++octant) {
                uint32_t child = offsets[node] + octant;
                bool descend = continuationCriterion(child);
                if (descend) {
                    if (offsets[child] == 0) {
                        // Leaf node reached
                        endpointAction(child);
                    } else {
                        assert(stackPos < 128);
                        stack[stackPos++] = child; // Push into the stack
                    }
                }
            }
            node = stack[--stackPos];

        } while (node != 0); // The root node is obtained, search finished
    }

    /**
     * @brief Search neighbors function. Given kernel that already contains a point and a radius, return the points inside the region.
     * @param k specific kernel that contains the data of the region (center and radius)
     * @param condition function that takes a candidate neighbor point and imposes an additional condition (should return a boolean).
     * The signature of the function should be equivalent to `bool cnd(const Lpoint &p);`
     * @param root The morton code for the node to start (usually the tree root which is 0)
     * @return Points inside the given kernel type. Actually the same as ptsInside.
     */
    template<typename Kernel, typename Function>
    [[nodiscard]] std::vector<Lpoint*> neighbors(const Kernel& k, Function&& condition, morton_t root = 0) const {
        std::vector<Lpoint*> ptsInside;
        auto center_id = k.center().id();

        auto intersectsKernel = [&](uint32_t nodeIndex) {
            return k.boxOverlap(this->centers[nodeIndex], this->radii[nodeIndex]);
        };
        
        auto findAndInsertPoints = [&](uint32_t nodeIndex) {
            uint32_t leafIdx = this->internalToLeaf[nodeIndex];
            auto pointsStart = this->layout[leafIdx], pointsEnd = this->layout[leafIdx+1];
            for (int32_t j = pointsStart; j < pointsEnd; j++) {
                Lpoint& p = this->points[j];  // Now we can get a non-const reference
                if (k.isInside(p) && center_id != p.id() && condition(p)) {
                    ptsInside.push_back(&p);
                }
            }
        };
        
        singleTraversal(intersectsKernel, findAndInsertPoints);
        return ptsInside;
	}

    /**
     * @brief Search neighbors function. Given a point and a radius, return the number of points inside a given kernel type
     * @param p Center of the kernel to be used
     * @param radius Radius of the kernel to be used
     * @param condition function that takes a candidate neighbor point and imposes an additional condition (should return a boolean).
     * The signature of the function should be equivalent to `bool cnd(const Lpoint &p);`
     * @return Points inside the given kernel
     */
	template<typename Kernel, typename Function>
	[[nodiscard]] size_t numNeighbors(const Kernel& k, Function&& condition) const {
        size_t ptsInside = 0;
        auto center_id = k.center().id();

        auto intersectsKernel = [&](uint32_t nodeIndex) {
            return k.boxOverlap(this->centers[nodeIndex], this->radii[nodeIndex]);
        };
        
        auto findAndIncrementPointsCount = [&](uint32_t nodeIndex) {
            uint32_t leafIdx = this->internalToLeaf[nodeIndex];
            auto pointsStart = this->layout[leafIdx], pointsEnd = this->layout[leafIdx+1];
            for (int32_t j = pointsStart; j < pointsEnd; j++) {
                Lpoint& p = this->points[j];  // Now we can get a non-const reference
                if (k.isInside(p) && center_id != p.id() && condition(p)) {
                    ptsInside++;
                }
            }
        };
        
        singleTraversal(intersectsKernel, findAndIncrementPointsCount);
        return ptsInside;
	}

    /**
     * @brief KNN algorithm. Returns the min(k, maxNeighs) nearest neighbors of a given point p
     * @param p
     * @param k
     * @param maxNeighs
     * @return
     */
    std::vector<Lpoint*> KNN(const Point& p, const size_t k, const size_t maxNeighs) const {
        std::vector<Lpoint*> knn{};
        std::unordered_map<size_t, bool> wasAdded{};

        double r = 1.0;

        size_t nmax = std::min(k, maxNeighs);
        const double rMax = bbox.radii().getMaxCoordinate(); // Use maximum radius as an upper bound

        while (knn.size() <= nmax && r <= rMax)
        {
            auto neighs = searchNeighbors<Kernel_t::sphere>(p, r);

            // Add all the points if there is room for them on proximity order
            if (knn.size() + neighs.size() > nmax) {
                std::sort(neighs.begin(), neighs.end(),
                        [&p](Lpoint* a, Lpoint* b) { return a->distance3D(p) < b->distance3D(p); });
            }

            for (const auto& n : neighs)  {
                if (!wasAdded[n->id()]) {
                    wasAdded[n->id()] = true;
                    knn.push_back(n); // Conditional inserting?
                }
            }
            r *= 2;
        }
        return knn;
    }

	/**
     * @brief Search neighbors function. Given a point and a radius, return the points inside a given kernel type
     * @param p Center of the kernel to be used
     * @param radius Radius of the kernel to be used
     * @return Points inside the given kernel type
     */
	template<Kernel_t kernel_type = Kernel_t::square>
	[[nodiscard]] inline std::vector<Lpoint*> searchNeighbors(const Point& p, double radius) const {
		const auto kernel = kernelFactory<kernel_type>(p, radius);
		// Dummy condition that always returns true, so we can use the same function for all cases
		// The compiler should optimize this away
		constexpr auto dummyCondition = [](const Lpoint&) { return true; };
		return neighbors(kernel, dummyCondition);
	}
	/**
     * @brief Search neighbors function. Given a point and a radius, return the points inside a given kernel type
     * @param p Center of the kernel to be used
     * @param radii Radii of the kernel to be used
     * @return Points inside the given kernel type
     */
	template<Kernel_t kernel_type = Kernel_t::cube>
	[[nodiscard]] inline std::vector<Lpoint*> searchNeighbors(const Point& p, const Vector& radii) const {
		const auto kernel = kernelFactory<kernel_type>(p, radii);
		// Dummy condition that always returns true, so we can use the same function for all cases
		// The compiler should optimize this away
		constexpr auto dummyCondition = [](const Lpoint&) { return true; };
		return neighbors(kernel, dummyCondition);
	}
    /**
     * @brief Search neighbors function. Given a point and a radius, return the points inside a given kernel type
     * @param p Center of the kernel to be used
     * @param radius Radius of the kernel to be used
     * @param condition function that takes a candidate neighbor point and imposes an additional condition (should return a boolean).
     * The signature of the function should be equivalent to `bool cnd(const Lpoint &p);`
     * @return Points inside the given kernel type
     */
	template<Kernel_t kernel_type = Kernel_t::square, class Function>
	[[nodiscard]] inline std::vector<Lpoint*> searchNeighbors(const Point& p, double radius, Function&& condition) const {
		const auto kernel = kernelFactory<kernel_type>(p, radius);
		return neighbors(kernel, std::forward<Function&&>(condition));
	}

	/**
     * @brief Search neighbors function. Given a point and a radius, return the points inside a given kernel type
     * @param p Center of the kernel to be used
     * @param radii Radii of the kernel to be used
     * @param condition function that takes a candidate neighbor point and imposes an additional condition (should return a boolean).
     * The signature of the function should be equivalent to `bool cnd(const Lpoint &p);`
     * @return Points inside the given kernel type
     */
	template<Kernel_t kernel_type = Kernel_t::square, class Function>
	[[nodiscard]] inline std::vector<Lpoint*> searchNeighbors(const Point& p, const Vector& radii,
	                                                          Function&& condition) const {
		const auto kernel = kernelFactory<kernel_type>(p, radii);
		return neighbors(kernel, std::forward<Function&&>(condition));
	}

	/**
     * Searching neighbors in 3D using a different radius for each direction
     * @param p Point around the neighbors will be search
     * @param radius Vector of radiuses: one per spatial direction
     * @param flags Vector of flags: return only points which flags[pointId] == false
     * @return Points inside the given kernel
     */
	[[nodiscard]] inline std::vector<Lpoint*> searchNeighbors3D(const Point& p, const double radius,
	                                                            const std::vector<bool>& flags) const {
		const auto condition = [&](const Point& point) { return !flags[point.id()]; };
		return searchNeighbors<Kernel_t::cube>(p, radius, condition);
	}

    [[nodiscard]] inline std::vector<Lpoint*> searchSphereNeighbors(const Point& point, const float radius) const {
		return searchNeighbors<Kernel_t::sphere>(point, radius);
	}

	/**
	 * A point is considered to be inside a Ring around a point if its outside the innerRing and inside the outerRing
	 * @param p Center of the kernel to be used
	 * @param innerRingRadii Radii of the inner part of the ring. Points within this part will be excluded
	 * @param outerRingRadii Radii of the outer part of the ring
	 * @return The points located between the inner ring and the outer ring
	 */
	[[nodiscard]] std::vector<Lpoint*> searchNeighborsRing(const Lpoint& p, const Vector& innerRingRadii,
	                                                       const Vector& outerRingRadii) const {
		// Search points within "outerRingRadii"
		const auto outerKernel = kernelFactory<Kernel_t::cube>(p, outerRingRadii);
		// But not too close (within "innerRingRadii")
		const auto innerKernel = kernelFactory<Kernel_t::cube>(p, innerRingRadii);
		const auto condition   = [&](const Point& point) { return !innerKernel.isInside(point); };

		return neighbors(outerKernel, condition);
	}

    // Other search functions
    [[nodiscard]] inline std::vector<Lpoint*> searchNeighbors2D(const Point& p, const double radius) const {
		return searchNeighbors<Kernel_t::square>(p, radius);
	}

	[[nodiscard]] inline std::vector<Lpoint*> searchCylinderNeighbors(const Lpoint& p, const double radius,
	                                                                  const double zMin, const double zMax) const {
		return searchNeighbors<Kernel_t::circle>(p, radius,
		                                         [&](const Lpoint& p) { return p.getZ() >= zMin && p.getZ() <= zMax; });
	}

	[[nodiscard]] inline std::vector<Lpoint*> searchCircleNeighbors(const Lpoint& p, const double radius) const {
		return searchNeighbors<Kernel_t::circle>(p, radius);
	}

	[[nodiscard]] inline std::vector<Lpoint*> searchCircleNeighbors(const Lpoint* p, const double radius) const {
		return searchCircleNeighbors(*p, radius);
	}

	[[nodiscard]] inline std::vector<Lpoint*> searchNeighbors3D(const Point& p, const Vector& radii) const {
		return searchNeighbors<Kernel_t::cube>(p, radii);
	}

	[[nodiscard]] inline std::vector<Lpoint*> searchNeighbors3D(const Point& p, double radius) const {
		return searchNeighbors<Kernel_t::cube>(p, radius);
	}

	/**
     * @brief Search neighbors function. Given a point and a radius, return the number of points inside a given kernel type
     * @param p Center of the kernel to be used
     * @param radius Radius of the kernel to be used
     * @return Points inside the given kernel
     */	
    template<Kernel_t kernel_type = Kernel_t::square>
	[[nodiscard]] inline size_t numNeighbors(const Point& p, const double radius) const {
		const auto kernel = kernelFactory<kernel_type>(p, radius);
        constexpr auto dummyCondition = [](const Lpoint&) { return true; };
		return numNeighbors(kernel, dummyCondition);
	}
	/**
     * @brief Search neighbors function. Given a point and a radius, return the number of points inside a given kernel type
     * @param p Center of the kernel to be used
     * @param radius Radius of the kernel to be used
     * @param condition function that takes a candidate neighbor point and imposes an additional condition (should return a boolean).
     * The signature of the function should be equivalent to `bool cnd(const Lpoint &p);`
     * @return Points inside the given kernel
     */
	template<Kernel_t kernel_type = Kernel_t::square, class Function>
	[[nodiscard]] inline size_t numNeighbors(const Point& p, const double radius, Function&& condition) const {
		const auto kernel = kernelFactory<kernel_type>(p, radius);
		return numNeighbors(kernel, std::forward<Function&&>(condition));
	}
};