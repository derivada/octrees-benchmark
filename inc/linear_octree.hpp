#pragma once

#include "Geometry/Box.hpp"
#include <optional>
#include <bitset>
#include <unordered_map>
#include <filesystem>
#include "PointEncoding/point_encoder.hpp"
#include "PointEncoding/no_encoding.hpp"
#include "NeighborKernels/KernelFactory.hpp"
#include "TimeWatcher.hpp"
#include "Geometry/Box.hpp"
#include "type_names.hpp"
#include "neighbor_set.hpp"
#include "encoding_octree_log.hpp"

/**
* @class LinearOctree
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

template <typename Point_t>
class LinearOctree {

private:
    using PointEncoder = PointEncoding::PointEncoder;
    using key_t = PointEncoding::key_t;
    using coords_t = PointEncoding::coords_t;

    PointEncoder& enc;

    /// @brief The maximum number of points in a leaf
    static constexpr size_t MAX_POINTS = 128;

    /// @brief The minimum octant radius to have in a leaf (TODO: this could be implemented in halfLength compuutation, but do we really need it?)
	static constexpr double MIN_OCTANT_RADIUS = 0;

	/// @brief The default size of the search set in KNN
	static constexpr size_t DEFAULT_KNN = 100;

	/// @brief The number of octants per internal node
	static constexpr uint8_t OCTANTS_PER_NODE  = 8;

    /// @brief Number of leaves and internal nodes in the octree. Equal to size of the leaves vector - 1.
    uint32_t nLeaf;

    /// @brief Number of internal nodes in the octree. Equal to (nLeaf-1) / 7.
    uint32_t nInternal;

    /// @brief Total number of nodes in the octree. Equal to nLeaf + nInternal.
    uint32_t nTotal;

    /**
     * @brief The leaves of the octree in cornerstone array format.
     * @details This array contains encoded (Hilbert or Morton) points (interpreted here as octal digit numbers) satisfying certain constraints:
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
    std::vector<key_t> leaves; 

    /// @brief  This array contains how many points have an encoding with a value between two of the leaves
    std::vector<size_t> counts;

    /// @brief  This array is built from counts via a traversal, and counts how many points internal nodes and leaves have
    std::vector<size_t> internalCounts;

    /// @brief This array is simply an exclusive scan of counts, and marks the index of the first point for a leaf
    std::vector<size_t> layout;

    /// @brief This array is built from exclusiveScan via a traversal, and marks the index of the first and last points for a leaf or internal node
    std::vector<std::pair<size_t, size_t>> internalLayoutRanges;

    /**
     * @brief The Warren-Salmon encoding of each node in the octree
     * @details For a given (internal or leaf) node, we store its position on the octree using this array, the position for a node at depth
     * n will be given by 0 000 000 ... 1 x1y1z1 ... xnynzn. This allows for traversals needed in neighbourhood search.
     * 
     * The process to obtain this array and link it with the leaves array is detailed in the cornerstone paper, section 5.
     */
    std::vector<key_t> prefixes;

    /// @brief Index of the first child of each node (if 0 we have a leaf)
    std::vector<uint32_t> offsets;

    /// @brief The parent index of every group of 8 sibling nodes
    std::vector<uint32_t> parents; // TODO: this may not be needed

    /// @brief First node index of every tree level (L+2 elements where L is MAX_DEPTH)
    std::vector<size_t> levelRange;

    /// @brief A map between the internal representation at offsets and the one in cornerstone format in leaves
    std::vector<int32_t> internalToLeaf;

    /// @brief The reverse mapping of internalToLeaf
    std::vector<int32_t> leafToInternal;

    /**
     * @brief A reference to the array of points that we sort
     * @details At the beginning of the octree construction, this points are encoded and then sorted in-place in the order given by their
     * encodings. Therefore, this array is altered inside this class. This is done to  locality that Morton/Hilbert
     */
    std::vector<Point_t> &points;

    /// @brief The encodings of the points in the octree
    std::vector<key_t> codes;

    /// @brief The center points of each node in the octree
    std::vector<Point> centers;

    /// @brief The vector of radii of each node in the octree
    std::vector<Vector> radii;

    /// @brief The global bounding box of the octree
    Box box;

    /// @brief A simple vector containinf the radii of each level in the octree to speed up computations.
    std::vector<Vector> precomputedRadii;

    /// @brief A vector containing the half-lengths of the minimum measure of the encoding.
    double halfLengths[3];
    
    /// @brief The maximum depth seen in the octree
    uint32_t maxDepthSeen = 0;

    size_t vectorMemorySize(const auto& vec) {
        return sizeof(vec) + vec.size() * sizeof(typename std::decay_t<decltype(vec)>::value_type);
    }

    /// @brief Returns the memory footprint of the octree (without counting points)
    size_t computeMemorySize() {
        size_t total_size = 0;
        total_size += vectorMemorySize(leaves);
        total_size += vectorMemorySize(counts);
        total_size += vectorMemorySize(internalCounts);
        total_size += vectorMemorySize(layout);
        total_size += vectorMemorySize(internalLayoutRanges);
        total_size += vectorMemorySize(prefixes);
        total_size += vectorMemorySize(offsets);
        total_size += vectorMemorySize(parents);
        total_size += vectorMemorySize(levelRange);
        total_size += vectorMemorySize(internalToLeaf);
        total_size += vectorMemorySize(leafToInternal);
        total_size += vectorMemorySize(codes);
        total_size += vectorMemorySize(centers);
        total_size += vectorMemorySize(radii);
        total_size += sizeof(precomputedRadii) + sizeof(halfLengths) + sizeof(box) + sizeof(maxDepthSeen) + sizeof(points);
        return total_size;
    }

    /**
     * @brief Computes the rebalacing decisions as the first process in the subdivision of the leaves array
     * 
     * @details This function implements g1 in the cornerstone paper, for each leaf we calculate the operation
     * that decides whether we merge, split or leave unchanged the leaf.
     * 
     * @param nodeOps The output array of decisions, of length leaves.size()-1
     */
    bool rebalanceDecision(std::vector<size_t> &nodeOps) {
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
            size_t parentCount =    counts[parentIndex]   + counts[parentIndex+1]+ 
                                    counts[parentIndex+2] + counts[parentIndex+3]+ 
                                    counts[parentIndex+4] + counts[parentIndex+5]+
                                    counts[parentIndex+6] + counts[parentIndex+7];
            if(parentCount <= MAX_POINTS)
                return 0; // merge
        }
        
        uint32_t nodeCount = counts[index];
        // Decide if we split this leaf or not
        // split into 4 layers
        if (nodeCount > MAX_POINTS * 512 && level + 3 < enc.maxDepth()) { 
            maxDepthSeen = std::max(maxDepthSeen, level + 4);
            return 4096; 
        }
        // split into 3 layers
        if (nodeCount > MAX_POINTS * 64 && level + 2 < enc.maxDepth()) { 
            maxDepthSeen = std::max(maxDepthSeen, level + 3);
            return 512;
        }
        // split into 2 layers
        if (nodeCount > MAX_POINTS * 8 && level + 1 < enc.maxDepth()) { 
            maxDepthSeen = std::max(maxDepthSeen, level + 2);
            return 64; 
        }
        // split into 1 layer
        if (nodeCount > MAX_POINTS && level < enc.maxDepth()) { 
            maxDepthSeen = std::max(maxDepthSeen, level + 1);
            return 8; 
        }
        // Don't do anything with this leaf
        return 1;
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
        key_t node = leaves[index];
        key_t range = leaves[index+1] - node;
        uint32_t level = enc.getLevel(range);
        if(level == 0) {
            return {-1, level};
        }

        uint32_t siblingId = enc.getSiblingId(node, level);

        // Checks if all siblings are on the tree, to do this, checks if the difference between the two parent nodes corresponding
        // to the code parent and the next parent is the range spanned by two consecutive codes at that level
        bool siblingsOnTree = leaves[index - siblingId + 8] == (leaves[index - siblingId] + enc.nodeRange(level - 1));
        if(!siblingsOnTree) siblingId = -1;

        return {siblingId, level};
    }

    /// @brief  Print the bits of the given code in groups of 3 to represent each level
    static void printMortonCode(key_t code) {
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
    void rebalanceTree(std::vector<key_t> &newLeaves, std::vector<size_t> &nodeOps) {
        size_t n = leaves.size() - 1;

        // Exclusive scan, step g2
        exclusiveScan(nodeOps.data(), n+1);

        // Initialization of the new leafs array
        newLeaves.resize(nodeOps[n] + 1);
        newLeaves.back() = leaves.back();

        // Compute the operations
        for (size_t i = 0; i < n; ++i) {
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
    void processNode(size_t index, std::vector<size_t> &nodeOps, std::vector<key_t> &newLeaves) {
        // The original value of the opCode (before exclusive scan)
        size_t opCode = nodeOps[index + 1] - nodeOps[index]; 
        if(opCode == 0)
            return;
    
        key_t node = leaves[index];
        key_t range = leaves[index+1] - node;
        uint32_t level = enc.getLevel(range);

        // The new position to put the node into (nodeOps value after exclusive scan)
        size_t newNodeIndex = nodeOps[index]; 

        // Copy the old node into the new position
        newLeaves[newNodeIndex] = node;
        if(opCode > 1) {
            // Split the node into 8^L as marked by the opCode, add the adequate codes to the new leaves
            uint32_t levelDiff = enc.log8ceil(opCode);
            key_t gap = enc.nodeRange(level + levelDiff);
            for (size_t sibling = 1; sibling < opCode; sibling++) {
                newLeaves[newNodeIndex + sibling] = newLeaves[newNodeIndex + sibling - 1] + gap;
            }
        }
    }

    /**
     * @brief Count number of particles in each leaf
     * 
     * @details This functions counts how many particles have encodings at leaf i, that is
     * between leaves[i] and leaves[i+1]
     */
    void computeNodeCounts() {
        size_t n = leaves.size() - 1;
        size_t codes_size = codes.size();
        size_t firstNode = 0;
        size_t lastNode = n;

        // Find general bounds for the codes array
        if(codes.size() > 0) {
            firstNode = std::upper_bound(leaves.begin(), leaves.end(), codes[0]) - leaves.begin() - 1;
            lastNode = std::upper_bound(leaves.begin(), leaves.end(), codes[codes_size-1]) - leaves.begin();
            assert(firstNode <= lastNode);
        } else {
            firstNode = n, lastNode = n;
        }

        // Fill non-populated parts of the octree with zeros
        for(size_t i = 0; i<firstNode; i++)
            counts[i] = 0;
        for(size_t i = lastNode; i<lastNode; i++)
            counts[i] = 0;

        size_t nNonZeroNodes = lastNode - firstNode;
        for(size_t i = 0; i<nNonZeroNodes; i++) {
            counts[i + firstNode] = calculateNodeCount(leaves[i+firstNode], leaves[i+firstNode+1]);
        }
    }

    /// @brief Since the codes array is sorted, we can use binary search to accelerate the counts computation
    size_t calculateNodeCount(key_t keyStart, key_t keyEnd) {
        auto rangeStart = std::lower_bound(codes.begin(), codes.end(), keyStart);
        auto rangeEnd   = std::lower_bound(codes.begin(), codes.end(), keyEnd);
        return rangeEnd - rangeStart;
    }

    /// @brief Simple serial implementation of an exclusive scan
    template<class Time_t>
    void exclusiveScan(Time_t* out, size_t numElements) {
        Time_t a = Time_t(0);
        Time_t b = Time_t(0);
        for (size_t i = 0; i < numElements; ++i) {
            a += out[i];
            out[i] = b;
            b = a;
        }
    }

    /// @brief Computes the key weight mapping for conversion between internal and leaf nodes
    constexpr int32_t binaryKeyWeight(key_t key, unsigned level){
        int32_t ret = 0;
        for (uint32_t l = 1; l <= level + 1; ++l)
        {
            uint32_t digit = enc.octalDigit(key, l);
            ret += digitWeight(digit);
        }
        return ret;
    }

    /// @brief Helper function for binaryKeyWeight
    constexpr int32_t digitWeight(uint32_t digit) {
        int32_t fourGeqMask = -int32_t(digit >= 4);
        return ((7 - digit) & fourGeqMask) - (digit & ~fourGeqMask);
    }

    /// @brief Create the prefixes and internaltoleaf arrays for the leafs
    void createUnsortedLayout() {
        for(size_t i = 0; i<nLeaf; i++) {
            key_t key = leaves[i];
            uint32_t level = enc.getLevel(leaves[i+1] - key);
            prefixes[i + nInternal] = enc.encodePlaceholderBit(key, level);
            internalToLeaf[i + nInternal] = i + nInternal;

            uint32_t prefixLength = enc.commonPrefix(key, leaves[i+1]);
            if(prefixLength % 3 == 0 && i < nLeaf - 1) {
                uint32_t octIndex = (i + binaryKeyWeight(key, prefixLength / 3)) / 7;
                prefixes[octIndex] = enc.encodePlaceholderBit(key, prefixLength / 3);
                internalToLeaf[octIndex] = octIndex;
            }
        }
    }

    /// @brief Determine octree subdivision level boundaries
    void getLevelRange() {
        for(uint32_t level = 0; level <= enc.maxDepth(); level++) {
            auto it = std::lower_bound(prefixes.begin(), prefixes.end(), enc.encodePlaceholderBit(0, level));
            levelRange[level] = std::distance(prefixes.begin(), it);
        }
        levelRange[enc.maxDepth() + 1] = nTotal;
    }

    /// @brief Extract parent/child relationships from binary tree and translate to sorted order
    void linkTree() {
        for(int i = 0; i<nInternal; i++) {
            size_t idxA = leafToInternal[i];
            key_t prefix = prefixes[idxA];
            key_t nodeKey = enc.decodePlaceholderBit(prefix);
            unsigned prefixLength = enc.decodePrefixLength(prefix);
            unsigned level = prefixLength / 3;
            assert(level < enc.maxDepth());

            key_t childPrefix = enc.encodePlaceholderBit(nodeKey, level + 1);

            size_t leafSearchStart = levelRange[level + 1];
            size_t leafSearchEnd   = levelRange[level + 2];
            auto childIdx = std::distance(prefixes.begin(), 
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
    
    /// @brief Computes the amount of points under an internal or leaf node
    size_t computeInternalNodeCounts(uint32_t node = 0) {
        // If node is a leaf, get its count from the array using itL mapping
        if(offsets[node] == 0) {
            internalCounts[node] = counts[internalToLeaf[node]];
            return counts[internalToLeaf[node]];
        }

        // Compute recursively (post-order DFS) the count of the internal node. It will be the sum of the counts of its children.
        size_t count = 0;
        for(uint8_t octant = 0; octant < OCTANTS_PER_NODE; octant++) {
            uint32_t child = offsets[node] + octant;
            count += computeInternalNodeCounts(child);   
        }
        internalCounts[node] = count;
        return count;
    }

    /// @brief Computes the ranges of point indexes covered by internal or leafs nodes
    std::pair<size_t, size_t> computeInternalNodeLayouts(uint32_t node = 0) {
        // If node is a leaf, get its internal layout from the two consecutive leafs on the layout array
        if(offsets[node] == 0) {
            internalLayoutRanges[node] = std::make_pair(layout[internalToLeaf[node]], layout[internalToLeaf[node] + 1]);
            return internalLayoutRanges[node];
        }

        // Compute recursively (post-order DFS) the count of the internal node. It will be the total range spanned by its children.
        for(uint8_t octant = 0; octant < OCTANTS_PER_NODE; octant++) {
            uint32_t child = offsets[node] + octant;
            auto childLayout = computeInternalNodeLayouts(child);
            if(octant == 0)
                internalLayoutRanges[node].first = childLayout.first;
            else if(octant == OCTANTS_PER_NODE-1)
                internalLayoutRanges[node].second = childLayout.second;
        }
        return internalLayoutRanges[node];
    }

public:    
    /**
     * @brief Builds the linear octree given an array of points, also reporting how much time each step takes
     * 
     * @details The points will be sorted in-place by the order given by the encoding to allow
     * spatial data locality
     */
    explicit LinearOctree(std::vector<Point_t>& points,
                        std::vector<key_t>& codes,
                        Box box,
                        PointEncoder& enc,
                        std::shared_ptr<EncodingOctreeLog> log = nullptr)
        : points(points), enc(enc), codes(codes), box(box)  {
        
        static_assert(!std::is_same_v<std::decay_t<PointEncoder>, PointEncoding::NoEncoding>,
                    "Encoder cannot be an instance of NoEncoding when using LinearOctree.");

        setupBbox();

        TimeWatcher tw;
        // Leaf construction
        tw.start();
        buildOctreeLeaves();
        tw.stop();
        if(log)
            log->octreeLeafTime = tw.getElapsedDecimalSeconds();
        
        // Internal part construction
        tw.start();
        resize();
        buildOctreeInternal();
        computeGeometry();
        tw.stop();
        if(log)
            log->octreeInternalTime = tw.getElapsedDecimalSeconds();


        // Output extra info from the build
        if (log) {
            log->octreeTime = log->octreeLeafTime + log->octreeInternalTime;
            log->max_leaf_points = MAX_POINTS;
            log->min_octant_radius = MIN_OCTANT_RADIUS;
            log->octreeType = "LinearOctree";
            log->memoryUsed = computeMemorySize();
            log->totalNodes = nTotal;
            log->leafNodes = nLeaf;
            log->internalNodes = nInternal;
            log->maxDepthSeen = maxDepthSeen;
            log->minRadiusAtMaxDepth = precomputedRadii[maxDepthSeen].getX();
        }
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
        precomputedRadii = std::vector<Vector>(enc.maxDepth() + 1);
        levelRange = std::vector<size_t>(enc.maxDepth() + 2);
        // Compute the physical half lengths for multiplying with the encoded coordinates
        halfLengths[0] = 0.5f * enc.eps() * (box.maxX() - box.minX());
        halfLengths[1] = 0.5f * enc.eps() * (box.maxY() - box.minY());
        halfLengths[2] = 0.5f * enc.eps() * (box.maxZ() - box.minZ());

        for(int i = 0; i <= enc.maxDepth(); i++) {
            coords_t sideLength = (1u << (enc.maxDepth() - i));
            precomputedRadii[i] = Vector(
                sideLength * halfLengths[0],
                sideLength * halfLengths[1],
                sideLength * halfLengths[2]
            );
        }
    }

    /// @brief Builds the octeee leaves array by repeatingly calling @ref updateOctreeLeaves()
    void buildOctreeLeaves() {
        // Builds the octree sequentially using the cornerstone algorithm
        // We start with 0, UPPER_BOUND on the leaves. Remember that UPPER_BOUND is 100000...000 with as many 0s as MAX_DEPTH, and it can never be reached by a code
        leaves = std::vector<key_t>{0, enc.upperBound()};
        counts = {codes.size()};

        while(!updateOctreeLeaves())
            ;
    }

    /**
     * @brief Computes the node operations to be done on the leaves and modifies the tree if necessary
     * 
     * @details Convergence is achieved when all the node operations to be done are equal to 1
     */
    bool updateOctreeLeaves() {
        std::vector<size_t> nodeOps(leaves.size());
        bool converged = rebalanceDecision(nodeOps);
        if(!converged) {
            std::vector<key_t> newTree;
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
        internalCounts.resize(nTotal);
        internalLayoutRanges.resize(nTotal);

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
        std::vector<std::pair<key_t, uint32_t>> prefixes_internalToLeaf(nTotal);
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

        // Compute internal node counts
        computeInternalNodeCounts();

        // Compute internal node layouts
        computeInternalNodeLayouts();
    }

    void printKey(uint64_t key) const {
        for(int i=20; i>=0; i--) {
            std::cout << std::bitset<3>(key >> (3*i)) << " ";
        }

        std::cout << std::endl;
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
            key_t prefix = prefixes[i];
            key_t startKey = enc.decodePlaceholderBit(prefix);
            uint32_t level = enc.decodePrefixLength(prefix) / 3;
            std::tie(centers[i], radii[i]) = 
                enc.getCenterAndRadii(startKey, level, box, halfLengths, precomputedRadii);
        }
    }

    /// @brief Computes the density of the point cloud as number of points / dataset
    double getDensity() {
        return (double) points.size() / (box.radii().getX() * box.radii().getY() * box.radii().getZ() * 8.0f);
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
     * The signature of the function should be equivalent to `bool cnd(const typename &p);`
     * @param root The morton code for the node to start (usually the tree root which is 0)
     * @return Points inside the given kernel type. Actually the same as ptsInside.
     */
    template<typename Kernel>
    [[nodiscard]] std::vector<Point_t*> neighbors(const Kernel& k) const {
        std::vector<Point_t*> ptsInside;
        auto checkBoxIntersect = [&](uint32_t nodeIndex) {
            auto nodeCenter = this->centers[nodeIndex];
            auto nodeRadii = this->radii[nodeIndex];
            switch (k.boxIntersect(nodeCenter, nodeRadii)) {
                case KernelAbstract::IntersectionJudgement::INSIDE: {
                    // Completely inside, all add points and prune
                    size_t startIndex = this->internalLayoutRanges[nodeIndex].first;
                    size_t endIndex = this->internalLayoutRanges[nodeIndex].second;
                    auto* startPtr = points.data() + startIndex;
                    auto* endPtr = points.data() + endIndex;
                    for (; startPtr != endPtr; ++startPtr) {
                        ptsInside.push_back(startPtr);
                    }
                    return false;
                }
                case KernelAbstract::IntersectionJudgement::OVERLAP:
                    // Overlaps but not inside, keep descending
                    return true;
                default:
                    // Completely outside, prune
                    return false;
            }
        };
        
        auto findAndInsertPoints = [&](uint32_t nodeIndex) {
            // Reached a leaf, add all points inside the kernel
            size_t startIndex = this->internalLayoutRanges[nodeIndex].first;
            size_t endIndex = this->internalLayoutRanges[nodeIndex].second;
            auto* startPtr = points.data() + startIndex;
            auto* endPtr = points.data() + endIndex;
            for (; startPtr != endPtr; ++startPtr) {
                if (k.isInside(*startPtr)) {
                    ptsInside.push_back(startPtr);
                }
            }
        };
        
        singleTraversal(checkBoxIntersect, findAndInsertPoints);
        return ptsInside;
	}

    /**
     * @brief Search neighbors function. Similar to neighbors(), but returns a list-of-ranges wrapper structure containing the neighbours. 
     * This structure implements a forward iterator and can thus be used in a loop. It is way faster as it does not need to copy
     * pointers to each individual point found. 
     * @param k specific kernel that contains the data of the region (center and radius)
     * @param condition function that takes a candidate neighbor point and imposes an additional condition (should return a boolean).
     * The signature of the function should be equivalent to `bool cnd(const typename &p);`
     * @param root The morton code for the node to start (usually the tree root which is 0)
     * @return Points inside the given kernel type. Actually the same as ptsInside.
     */
    template<typename Kernel>
    [[nodiscard]] NeighborSet<Point_t> neighborsStruct(const Kernel& k) {
        NeighborSet<Point_t> result(&points);
        auto checkBoxIntersect = [&](uint32_t nodeIndex) {
            auto nodeCenter = this->centers[nodeIndex];
            auto nodeRadii = this->radii[nodeIndex];
            switch (k.boxIntersect(nodeCenter, nodeRadii)) {
                case KernelAbstract::IntersectionJudgement::INSIDE: {
                    // Completely inside, add octant to the result
                    result.addRange(internalLayoutRanges[nodeIndex].first, internalLayoutRanges[nodeIndex].second);
                    return false;
                }
                case KernelAbstract::IntersectionJudgement::OVERLAP:
                    // Overlaps but not inside, keep descending
                    return true;
                default:
                    // Completely outside, prune
                    return false;
            }
        };
        
        auto findAndInsertPoints = [&](uint32_t nodeIndex) {
            // Reached a leaf, add all points inside the kernel
            size_t startIndex = this->internalLayoutRanges[nodeIndex].first;
            size_t endIndex = this->internalLayoutRanges[nodeIndex].second;
            auto* startPtr = points.data() + startIndex;
            auto* endPtr = points.data() + endIndex;
            size_t rangeStart = startIndex, rangeEnd = startIndex;
            for (; startPtr != endPtr; ++startPtr, ++rangeEnd) {
                if (!k.isInside(*startPtr)) {
                    if (rangeStart < rangeEnd) {
                        // Store the last valid range. Ranges are exclusive on the right, i.e. (2,4) has points 2 and 3.
                        result.addRange(rangeStart, rangeEnd); 
                    }
                    rangeStart = rangeEnd + 1; // Reset the range start to the next point
                }
            }
            // Insert the last range if it was open
            if (rangeStart < rangeEnd) {
                result.addRange(rangeStart, rangeEnd);
            }
        };
        
        singleTraversal(checkBoxIntersect, findAndInsertPoints);
        return result;
	}

    uint32_t getPrecisionLevel(const Vector &toleranceVector) const {
        assert(toleranceVector.getX() > 0 && toleranceVector.getY() > 0 && toleranceVector.getZ() > 0 && "tolerance vector must be >0 in every coordinate");
        uint32_t precisionLevel = maxDepthSeen;
        for(uint32_t i = 0; i<maxDepthSeen; i++) {
            Vector radii = precomputedRadii[i];
            if(radii.getX() < toleranceVector.getX() && radii.getY() < toleranceVector.getY() && radii.getZ() && toleranceVector.getZ()) {
                precisionLevel = i;
                // std::cout << "for tolerance vector " << toleranceVector << " set precisionLevel to " << precisionLevel << "\n after comparing with " << radii << "\n";
                break;
            }
        }
        return precisionLevel;
    }

    uint32_t getPrecisionLevel(double tolerance) const {
        assert(tolerance > 0 && "tolerance must be greater than 0");
        return getPrecisionLevel(Vector{tolerance, tolerance, tolerance});
    }

    uint32_t getPrecisionLevel(const Vector &kernelRadii, double tolerancePercentage) const {
        assert(tolerancePercentage > 0.0 && "tolerance percentage must be greater than 0");
        return getPrecisionLevel(Vector{
                kernelRadii.getX() * tolerancePercentage / 100.0, 
                kernelRadii.getY() * tolerancePercentage / 100.0, 
                kernelRadii.getZ() * tolerancePercentage / 100.0
        });
    }

    template<typename Kernel>
    [[nodiscard]] NeighborSet<Point_t> neighborsApprox(const Kernel& k, uint32_t precisionLevel, bool upperBound) {
        // std::cout << "approximate neigh search with max precision level " << precisionLevel << "and mode " << (upperBound ? "upper bound": "lower bound") << std::endl;
        NeighborSet<Point_t> result(&points);
        auto checkBoxIntersect = [&](uint32_t nodeIndex) {
            auto nodeCenter = this->centers[nodeIndex];
            auto nodeRadii = this->radii[nodeIndex];
            uint32_t nodeDepth = enc.decodePrefixLength(this->prefixes[nodeIndex]) / 3;
            if(nodeDepth > precisionLevel) {
                // If we are beyond precision and in upper bound mode, add octant to the result
                if(upperBound) {
                    result.addRange(internalLayoutRanges[nodeIndex].first, internalLayoutRanges[nodeIndex].second);
                }
                // Prune either way
                return false;
            } 

            switch (k.boxIntersect(nodeCenter, nodeRadii)) {
                case KernelAbstract::IntersectionJudgement::INSIDE: {
                    // Completely inside, add octant to the result
                    result.addRange(internalLayoutRanges[nodeIndex].first, internalLayoutRanges[nodeIndex].second);
                    return false;
                }
                case KernelAbstract::IntersectionJudgement::OVERLAP:
                    // Overlaps but not inside, keep descending
                    return true;
                default:
                    // Completely outside, prune
                    return false;
            }
        };
        
        auto findAndInsertPoints = [&](uint32_t nodeIndex) {
            // Reached a leaf, add all points inside the kernel
            size_t startIndex = this->internalLayoutRanges[nodeIndex].first;
            size_t endIndex = this->internalLayoutRanges[nodeIndex].second;
            auto* startPtr = points.data() + startIndex;
            auto* endPtr = points.data() + endIndex;
            size_t rangeStart = startIndex, rangeEnd = startIndex;
            for (; startPtr != endPtr; ++startPtr, ++rangeEnd) {
                if (!k.isInside(*startPtr)) {
                    if (rangeStart < rangeEnd) {
                        // Store the last valid range. Ranges are exclusive on the right, i.e. (2,4) has points 2 and 3.
                        result.addRange(rangeStart, rangeEnd); 
                    }
                    rangeStart = rangeEnd + 1; // Reset the range start to the next point
                }
            }
            // Insert the last range if it was open
            if (rangeStart < rangeEnd) {
                result.addRange(rangeStart, rangeEnd);
            }
        };
        
        singleTraversal(checkBoxIntersect, findAndInsertPoints);
        return result;
	}
    
    template<typename Kernel>
    [[nodiscard]] NeighborSet<Point_t> neighborsApprox(const Kernel& k, const Vector &toleranceVector, bool upperBound) {
        return neighborsApprox(k, getPrecisionLevel(toleranceVector), upperBound);
    }

    template<Kernel_t kernel_type = Kernel_t::square>
	[[nodiscard]] inline NeighborSet<Point_t> searchNeighborsApprox(const Point& p, double radius, const Vector &toleranceVector, bool upperBound) {
		const auto kernel = kernelFactory<kernel_type>(p, radius);
		return neighborsApprox(kernel, toleranceVector, upperBound);
	}
	template<Kernel_t kernel_type = Kernel_t::cube>
	[[nodiscard]] inline NeighborSet<Point_t> searchNeighborsApprox(const Point& p, const Vector& radii, const Vector &toleranceVector, bool upperBound) {
		const auto kernel = kernelFactory<kernel_type>(p, radii);
		return neighborsApprox(kernel, toleranceVector, upperBound);
	}

    template<typename Kernel>
    [[nodiscard]] NeighborSet<Point_t> neighborsApprox(const Kernel& k, double tolerancePercentage, bool upperBound) {
        return neighborsApprox(k, getPrecisionLevel(k.radii(), tolerancePercentage), upperBound);
    }

    template<Kernel_t kernel_type = Kernel_t::square>
	[[nodiscard]] inline NeighborSet<Point_t> searchNeighborsApprox(const Point& p, double radius, double tolerancePercentage, bool upperBound) {
		const auto kernel = kernelFactory<kernel_type>(p, radius);
		return neighborsApprox(kernel, tolerancePercentage, upperBound);
	}
	template<Kernel_t kernel_type = Kernel_t::cube>
	[[nodiscard]] inline NeighborSet<Point_t> searchNeighborsApprox(const Point& p, const Vector& radii, double tolerancePercentage, bool upperBound) {
		const auto kernel = kernelFactory<kernel_type>(p, radii);
		return neighborsApprox(kernel, tolerancePercentage, upperBound);
	}

    /**
     * @brief Search neighbors function. Given a point and a radius, return the number of points inside a given kernel type
     * @param p Center of the kernel to be used
     * @param radius Radius of the kernel to be used
     * The signature of the function should be equivalent to `bool cnd(const typename &p);`
     * @return Points inside the given kernel
     */
	template<typename Kernel>
	[[nodiscard]] size_t numNeighbors(const Kernel& k) const {
        size_t ptsInside = 0;
        auto checkBoxIntersect = [&](uint32_t nodeIndex) {
            auto nodeCenter = this->centers[nodeIndex];
            auto nodeRadii = this->radii[nodeIndex];
            switch (k.boxIntersect(nodeCenter, nodeRadii)) {
                case KernelAbstract::IntersectionJudgement::INSIDE:
                    // Completely inside, all add points except center and prune
                    ptsInside += this->internalCounts[nodeIndex];
                    return false;
                case KernelAbstract::IntersectionJudgement::OVERLAP:
                    // Overlaps but not inside, keep descending
                    return true;
                case  KernelAbstract::IntersectionJudgement::OUTSIDE:
                    // Completely outside, prune
                    return false;
                default:
                    return false;
            }
        };
        
        auto findAndIncrementPointsCount = [&](uint32_t nodeIndex) {
            // Reached a leaf, add all points inside the kernel
            size_t startIndex = this->internalLayoutRanges[nodeIndex].first;
            size_t endIndex = this->internalLayoutRanges[nodeIndex].second;
            auto* startPtr = points.data() + startIndex;
            auto* endPtr = points.data() + endIndex;
            for (; startPtr != endPtr; ++startPtr) {
                if (k.isInside(*startPtr)) {
                    ptsInside++;
                }
            }
        };
        singleTraversal(checkBoxIntersect, findAndIncrementPointsCount);
        return ptsInside;
	}

    /**
     * @brief KNN algorithm. Returns the min(k, maxNeighs) nearest neighbors of a given point p
     * @param p
     * @param k
     * @param maxNeighs
     * @return
     */
    std::vector<Point_t*> KNN(const Point& p, const size_t k, const size_t maxNeighs) const {
        std::vector<Point_t*> knn{};
        std::unordered_map<size_t, bool> wasAdded{};

        double r = 1.0;

        size_t nmax = std::min(k, maxNeighs);
        const double rMax = box.radii().getMaxCoordinate(); // Use maximum radius as an upper bound

        while (knn.size() <= nmax && r <= rMax)
        {
            auto neighs = searchNeighbors<Kernel_t::sphere>(p, r);

            // Add all the points if there is room for them on proximity order
            if (knn.size() + neighs.size() > nmax) {
                std::sort(neighs.begin(), neighs.end(),
                        [&p](Point_t* a, Point_t* b) { return a->distance3D(p) < b->distance3D(p); });
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
	[[nodiscard]] inline std::vector<Point_t*> searchNeighbors(const Point& p, double radius) const {
		const auto kernel = kernelFactory<kernel_type>(p, radius);
		return neighbors(kernel);
	}
	/**
     * @brief Search neighbors function. Given a point and a radius, return the points inside a given kernel type
     * @param p Center of the kernel to be used
     * @param radii Radii of the kernel to be used
     * @return Points inside the given kernel type
     */
	template<Kernel_t kernel_type = Kernel_t::cube>
	[[nodiscard]] inline std::vector<Point_t*> searchNeighbors(const Point& p, const Vector& radii) const {
		const auto kernel = kernelFactory<kernel_type>(p, radii);
		return neighbors(kernel);
	}

    template<Kernel_t kernel_type = Kernel_t::square>
	[[nodiscard]] inline NeighborSet<Point_t> searchNeighborsStruct(const Point& p, double radius) {
		const auto kernel = kernelFactory<kernel_type>(p, radius);
		return neighborsStruct(kernel);
	}

	template<Kernel_t kernel_type = Kernel_t::cube>
	[[nodiscard]] inline NeighborSet<Point_t> searchNeighborsStruct(const Point& p, const Vector& radii) {
		const auto kernel = kernelFactory<kernel_type>(p, radii);
		return neighborsStruct(kernel);
	}

	/**
	 * A point is considered to be inside a Ring around a point if its outside the innerRing and inside the outerRing
	 * @param p Center of the kernel to be used
	 * @param innerRingRadii Radii of the inner part of the ring. Points within this part will be excluded
	 * @param outerRingRadii Radii of the outer part of the ring
	 * @return The points located between the inner ring and the outer ring
	 */
	[[nodiscard]] std::vector<Point_t*> searchNeighborsRing(const Point_t& p, const Vector& innerRingRadii,
	                                                       const Vector& outerRingRadii) const {
		// Search points within "outerRingRadii"
		const auto outerKernel = kernelFactory<Kernel_t::cube>(p, outerRingRadii);
		// But not too close (within "innerRingRadii")
		const auto innerKernel = kernelFactory<Kernel_t::cube>(p, innerRingRadii);
		const auto condition   = [&](const Point& point) { return !innerKernel.isInside(point); };

		return neighbors(outerKernel, condition);
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
        // constexpr auto dummyCondition = [](const Point_t&) { return true; };
		return numNeighbors(kernel);
	}

    // OLD IMPLEMENTATIONS KEPT FOR COMPARISON AND TESTING PURPOSES
    // ALSO THE OLD IMPL CAN TAKE AN ARBITRARY CONDITION ON THE SEARCHES, WHILE THE NEW CAN'T
    template<typename Kernel, typename Function>
    [[nodiscard]] std::vector<Point_t*> neighborsOld(const Kernel& k, Function&& condition) const {
        std::vector<Point_t*> ptsInside;

        auto intersectsKernel = [&](uint32_t nodeIndex) {
            return k.boxOverlap(this->centers[nodeIndex], this->radii[nodeIndex]);
        };
        
        auto findAndInsertPoints = [&](uint32_t nodeIndex) {
            uint32_t leafIdx = this->internalToLeaf[nodeIndex];
            auto pointsStart = this->layout[leafIdx], pointsEnd = this->layout[leafIdx+1];
            for (int32_t j = pointsStart; j < pointsEnd; j++) {
                Point_t& p = this->points[j];  // Now we can get a non-const reference
                if (k.isInside(p) && condition(p)) {
                    ptsInside.push_back(&p);
                }
            }
        };
        singleTraversal(intersectsKernel, findAndInsertPoints);
        return ptsInside;
	}
    template<Kernel_t kernel_type = Kernel_t::square>
	[[nodiscard]] inline std::vector<Point_t*> searchNeighborsOld(const Point& p, double radius) const {
		const auto kernel = kernelFactory<kernel_type>(p, radius);
		constexpr auto dummyCondition = [](const Point_t&) { return true; };
		return neighborsOld(kernel, dummyCondition);
	}
	template<Kernel_t kernel_type = Kernel_t::cube>
	[[nodiscard]] inline std::vector<Point_t*> searchNeighborsOld(const Point& p, const Vector& radii) const {
		const auto kernel = kernelFactory<kernel_type>(p, radii);
		constexpr auto dummyCondition = [](const Point_t&) { return true; };
		return neighborsOld(kernel, dummyCondition);
	}
	template<Kernel_t kernel_type = Kernel_t::square, class Function>
	[[nodiscard]] inline std::vector<Point_t*> searchNeighborsOld(const Point& p, double radius, Function&& condition) const {
		const auto kernel = kernelFactory<kernel_type>(p, radius);
		return neighborsOld(kernel, std::forward<Function&&>(condition));
	}
	template<Kernel_t kernel_type = Kernel_t::square, class Function>
	[[nodiscard]] inline std::vector<Point_t*> searchNeighborsOld(const Point& p, const Vector& radii,
	                                                          Function&& condition) const {
		const auto kernel = kernelFactory<kernel_type>(p, radii);
		return neighborsOld(kernel, std::forward<Function&&>(condition));
	}

	[[nodiscard]] std::vector<Point_t*> searchNeighborsRingOld(const Point_t& p, const Vector& innerRingRadii,
	                                                       const Vector& outerRingRadii) const {
		const auto outerKernel = kernelFactory<Kernel_t::cube>(p, outerRingRadii);
		const auto innerKernel = kernelFactory<Kernel_t::cube>(p, innerRingRadii);
		const auto condition   = [&](const Point& point) { return !innerKernel.isInside(point); };

		return neighborsOld(outerKernel, condition);
	}

    template<typename Kernel, typename Function>
	[[nodiscard]] size_t numNeighborsOld(const Kernel& k, Function&& condition) const {
        size_t ptsInside = 0;

        auto intersectsKernel = [&](uint32_t nodeIndex) {
            return k.boxOverlap(this->centers[nodeIndex], this->radii[nodeIndex]);
        };
        
        auto findAndIncrementPointsCount = [&](uint32_t nodeIndex) {
            uint32_t leafIdx = this->internalToLeaf[nodeIndex];
            auto pointsStart = this->layout[leafIdx], pointsEnd = this->layout[leafIdx+1];
            for (int32_t j = pointsStart; j < pointsEnd; j++) {
                Point_t& p = this->points[j];
                if (k.isInside(p) && condition(p)) {
                    ptsInside++;
                }
            }
        };
        
        singleTraversal(intersectsKernel, findAndIncrementPointsCount);
        return ptsInside;
	}
    template<Kernel_t kernel_type = Kernel_t::square>
	[[nodiscard]] inline size_t numNeighborsOld(const Point& p, const double radius) const {
		const auto kernel = kernelFactory<kernel_type>(p, radius);
        constexpr auto dummyCondition = [](const Point_t&) { return true; };
		return numNeighborsOld(kernel, dummyCondition);
	}
	template<Kernel_t kernel_type = Kernel_t::square, class Function>
	[[nodiscard]] inline size_t numNeighborsOld(const Point& p, const double radius, Function&& condition) const {
		const auto kernel = kernelFactory<kernel>(p, radius);
		return numNeighborsOld(kernel, std::forward<Function&&>(condition));
	}

    // Misc. functions for debugging
    template <typename Time_t>
    void writeVector(std::ofstream &file, std::vector<Time_t> &v, std::string name = "v") {
        file << "Printing vector " << name << " with " << v.size() << "elements\n";
        for(size_t i = 0; i<v.size(); i++)
            file << name << "[" << i << "] = " << v[i] << "\n";
    }

    template <typename U, typename V>
    void writeVectorPairs(std::ofstream &file, std::vector<std::pair<U, V>> &v, std::string name = "v") {
        file << "Printing vector " << name << " with " << v.size() << "elements\n";
        for(size_t i = 0; i<v.size(); i++)
            file << name << "[" << i << "] = " << v[i].first << ", " << v[i].second << "\n";
    }

    template <typename Time_t>
    void writeVectorBinary(std::ofstream &file, std::vector<Time_t> &v, std::string name = "v") {
        file << "Printing vector " << name << " with " << v.size() << "elements\n";
        for(size_t i = 0; i<v.size(); i++)
            file << name << "[" << i << "] = " << std::bitset<64>(v[i]) << "\n";
    }
    
    void writePointsAndCodes(std::ofstream &file, const std::string &encoder_name) {
        file << std::fixed << std::setprecision(3); 
        file << encoder_name << " " << "x y z\n";
        assert(codes.size() == points.size());
        for(size_t i = 0; i<codes.size(); i++) 
            file << codes[i] << " " << points[i].getX() << " " << points[i].getY() << " " << points[i].getZ() << "\n";
    }
    
    void logOctree(std::ofstream &file, std::ofstream &pointsFile) {
        std::cout << "(1/2) Logging octree parameters and structure" << std::endl;
        std::string pointTypeName = getPointName<Point_t>();
        std::string encoderTypename = enc.getEncoderName();
        file << "---- Linear octree parameters ----";
        file << "Encoder: " << encoderTypename << "\n";
        file << "Point type: " << pointTypeName << "\n";
        file << "Max. points per leaf: " << MAX_POINTS << "\n";
        file << "Total number of nodes = " << nTotal << "\n Leafs = " << nLeaf << "\n Internal nodes = " << nInternal << "\n";
        file << "---- Full structure ----";
        writeVectorBinary(file, leaves, "leaves");
        writeVector(file, counts, "counts");
        writeVector(file, layout, "layout");
        writeVectorBinary(file, prefixes, "prefixes");
        writeVector(file, offsets, "offsets");
        writeVector(file, parents, "parents");
        writeVector(file, levelRange, "levelRange");
        writeVector(file, internalToLeaf, "internalToLeaf");
        writeVector(file, leafToInternal, "leafToInternal");
        writeVector(file, internalCounts, "internalCounts");
        writeVectorPairs(file, internalLayoutRanges, "internalLayoutRanges");
        file << std::flush;
        
        // write codes and point coordinates
        std::cout << "(2/2) Logging encoded points" << std::endl;
        writePointsAndCodes(pointsFile, encoderTypename);
        pointsFile << std::flush;
        std::cout << "Done! Octree and points logged" << std::endl;
    }

    void logOctreeBounds(std::ofstream &outputFile, int max_level) {
        outputFile << "level,upx,upy,upz,downx,downy,downz\n";
        auto logBounds = [&](uint32_t nodeIndex) {
            auto nodeCenter = this->centers[nodeIndex];
            auto nodeRadii = this->radii[nodeIndex];
            uint32_t nodeDepth = enc.decodePrefixLength(this->prefixes[nodeIndex]) / 3;
            auto up = nodeCenter + nodeRadii;
            auto down = nodeCenter - nodeRadii;
            outputFile << nodeDepth << "," << up.getX() << "," << up.getY() << "," << up.getZ() << "," 
                << down.getX() << "," << down.getY() << "," << down.getZ() << "\n";

            return nodeDepth+1 <= max_level;
        };
        
        auto logBoundsLeaf = [&](uint32_t nodeIndex) {
            
        };
        
        singleTraversal(logBounds, logBoundsLeaf);
	}

};