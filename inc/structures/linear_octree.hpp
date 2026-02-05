#pragma once

#include <optional>
#include <bitset>
#include <unordered_map>
#include <filesystem>
#include "benchmarking/build_log.hpp"
#include "benchmarking/time_watcher.hpp"
#include "encoding/no_encoding.hpp"
#include "encoding/point_encoder.hpp"
#include "geometry/box.hpp"
#include "geometry/point_containers.hpp"
#include "kernels/kernel_factory.hpp"
#include "main_options.hpp"
#include "neighbor_set.hpp"

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

template <PointContainer Container>
class LinearOctree {

protected:
    using PointEncoder = PointEncoding::PointEncoder;
    using key_t = PointEncoding::key_t;
    using coords_t = PointEncoding::coords_t;
    
    /// @brief The minimum octant radius to have in a leaf (TODO: this could be implemented in halfLength compuutation, but do we really need it?)
	static constexpr double MIN_OCTANT_RADIUS = 0;

	/// @brief The number of octants per internal node
	static constexpr uint8_t OCTANTS_PER_NODE  = 8;

    /// @brief Leaves of the Octree, destroyed after build
    struct LeafPart {
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
         * between two leaves is greater than mainOptions.maxPointsLeaf.
         * 
         * For more details about the construction, check the cornerstone paper, section 4.
         */
        std::vector<key_t> leaves; 

        /// @brief  This array contains how many points have an encoding with a value between two of the leaves
        std::vector<size_t> counts;

        /// @brief This array is simply an exclusive scan of counts, and marks the index of the first point for a leaf
        std::vector<size_t> layout;
    };

    /// @brief Internal part of the Octree, destroyed after build
    struct InternalPart {
        /**
        * @brief The Warren-Salmon encoding of each node in the octree
        * @details For a given (internal or leaf) node, we store its position on the octree using this array, the position for a node at depth
        * n will be given by 0 000 000 ... 1 x1y1z1 ... xnynzn. This allows for traversals needed in neighbourhood search.
        * 
        * The process to obtain this array and link it with the leaves array is detailed in the cornerstone paper, section 5.
        */
        std::vector<key_t> prefixes;

        /// @brief The parent index of every group of 8 sibling nodes
        std::vector<uint32_t> parents; // TODO: this may not be needed

        /// @brief First node index of every tree level (L+2 elements where L is MAX_DEPTH)
        std::vector<size_t> levelRange;

        /// @brief A map between the internal representation at offsets and the one in cornerstone format in leaves
        std::vector<int32_t> internalToLeaf;

        /// @brief The reverse mapping of internalToLeaf
        std::vector<int32_t> leafToInternal;
    };
    
    /// @brief Number of leaves and internal nodes in the octree. Equal to size of the leaves vector - 1.
    uint32_t nLeaf;

    /// @brief Number of internal nodes in the octree. Equal to (nLeaf-1) / 7.
    uint32_t nInternal;

    /// @brief Total number of nodes in the octree. Equal to nLeaf + nInternal.
    uint32_t nTotal;

    /// @brief The maximum depth seen in the octree
    uint32_t maxDepthSeen = 0;

    /**
     * The next 4 arrays contain the important information for neighbor searches
     */
    /// @brief Index of the first child of each node (if 0 we have a leaf). This is the array used during DFS Octree searches.
    std::vector<uint32_t> offsets;

    /// @brief This array is built from exclusiveScan via a traversal, and marks the index of the first and last points for a leaf or internal node.
    std::vector<std::pair<size_t, size_t>> internalRanges;

    /// @brief The center points of each node in the octree
    std::vector<Point> centers;

    /// @brief A simple vector containinf the radii of each level in the octree to speed up computations.
    std::vector<Vector> precomputedRadii;

    /**
     * @brief A reference to the array of points that we sort
     * @details At the beginning of the octree construction, this points are encoded and then sorted in-place in the order given by their
     * encodings. Therefore, this array is altered inside this class. This is done to  locality that Morton/Hilbert
     */
    Container &points;

    /// @brief The global bounding box of the octree
    Box box;

    /// @brief A reference to the PointEncoder used to sort the cloud before starting the LinearOctree build.
    PointEncoder& enc;

    /// @brief The encodings of the points in the octree, not needed during searches so we don't count towards total memory overhead
    std::vector<key_t> &codes;

    /// @brief A vector containing the half-lengths of the minimum measure of the encoding.
    double halfLengths[3];


    /**
     * @brief Computes the rebalacing decisions as the first process in the subdivision of the leaves array
     * 
     * @details This function implements g1 in the cornerstone paper, for each leaf we calculate the operation
     * that decides whether we merge, split or leave unchanged the leaf.
     * 
     * @param nodeOps The output array of decisions, of length leaves.size()-1
     */
    bool rebalanceDecision(LeafPart &leaf, std::vector<size_t> &nodeOps) {
        bool converged = true;
        #pragma omp parallel for schedule(static)
            for(int i = 0; i<leaf.leaves.size()-1; i++) {
                nodeOps[i] = calculateNodeOp(leaf, i);
                if(nodeOps[i] != 1) converged = false;
            }
        return converged;
    } 

    /**
     * @brief Computes the operation on the leaf marked by index
     * 
     * @details The following values can be returned:
     * - 1 if the leaf should remain unchanged
     * - 0 if the leaf should be merged (only if the counts of all of its siblings are <= mainOptions.maxPointsLeaf), the siblins
     * are all next to each other and the node is not the first sibling 
     * (because the first sibling is the node that will stay after merge)
     * - 8^L where L goes up to 4, if we need to split the node L times (recursively)
     * 
     * @param index The leaf array index
     */
    uint32_t calculateNodeOp(LeafPart &leaf, uint32_t index) {
        auto [sibling, level] = siblingAndLevel(leaf, index);

        if(sibling > 0) {
            // We have 8 siblings next to each other, could merge this node if the count of all siblings is less MAX_COUNT
            uint32_t parentIndex = index - sibling;
            size_t parentCount =    leaf.counts[parentIndex]   + leaf.counts[parentIndex+1]+ 
                                    leaf.counts[parentIndex+2] + leaf.counts[parentIndex+3]+ 
                                    leaf.counts[parentIndex+4] + leaf.counts[parentIndex+5]+
                                    leaf.counts[parentIndex+6] + leaf.counts[parentIndex+7];
            if(parentCount <= mainOptions.maxPointsLeaf)
                return 0; // merge
        }
        
        uint32_t nodeCount = leaf.counts[index];
        // Decide if we split this leaf or not
        if (nodeCount > mainOptions.maxPointsLeaf && level < enc.maxDepth()) { 
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
    inline std::pair<int32_t, uint32_t> siblingAndLevel(LeafPart &leaf, uint32_t index) {
        key_t node = leaf.leaves[index];
        key_t range = leaf.leaves[index+1] - node;
        uint32_t level = enc.getLevel(range);
        if(level == 0) {
            return {-1, level};
        }

        uint32_t siblingId = enc.getSiblingId(node, level);

        // Checks if all siblings are on the tree, to do this, checks if the difference between the two parent nodes corresponding
        // to the code parent and the next parent is the range spanned by two consecutive codes at that level
        bool siblingsOnTree = leaf.leaves[index - siblingId + 8] == (leaf.leaves[index - siblingId] + enc.nodeRange(level - 1));
        if(!siblingsOnTree) siblingId = -1;

        return {siblingId, level};
    }

    /**
     * @brief Computes a new stage of the leaves array by subdividing using the operations given
     * 
     * @details This function implements steps g2 and g3 of the subdivision process
     * 
     * @param newLeaves The new leaves array that will be swapped with the current one after this function execution
     * @param nodeOps The array of operations performed in the first step (g1)
     */
    void rebalanceTree(LeafPart &leaf, std::vector<key_t> &newLeaves, std::vector<size_t> &nodeOps) {
        size_t n = leaf.leaves.size() - 1;

        // Exclusive scan, step g2
        exclusiveScan(nodeOps.data(), n+1);

        // Initialization of the new leafs array
        newLeaves.resize(nodeOps[n] + 1);
        newLeaves.back() = leaf.leaves.back();

        // Compute the operations
        #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < n; ++i) {
                processNode(leaf, i, nodeOps, newLeaves);
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
    void processNode(LeafPart &leaf, size_t index, std::vector<size_t> &nodeOps, std::vector<key_t> &newLeaves) {
        // The original value of the opCode (before exclusive scan)
        size_t opCode = nodeOps[index + 1] - nodeOps[index]; 
        if(opCode == 0)
            return;
    
        key_t node = leaf.leaves[index];
        key_t range = leaf.leaves[index+1] - node;
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
    void computeNodeCounts(LeafPart &leaf) {
        size_t n = 0;
        if(leaf.leaves.size()) n = leaf.leaves.size() - 1;
        size_t codes_size = codes.size();
        size_t firstNode = 0;
        size_t lastNode = n;

        // Find general bounds for the codes array
        if(codes.size() > 0) {
            firstNode = std::upper_bound(leaf.leaves.begin(), leaf.leaves.end(), codes[0]) - leaf.leaves.begin() - 1;
            lastNode = std::upper_bound(leaf.leaves.begin(), leaf.leaves.end(), codes[codes_size-1]) - leaf.leaves.begin();
        } else {
            firstNode = n, lastNode = n;
        }

        // Fill non-populated parts of the octree with zeros and the inside with calculateNodeCount
        #pragma omp parallel for schedule(static)
            for(size_t i = 0; i<n; i++) {
                if(i < firstNode || i >= lastNode)
                    leaf.counts[i] = 0;
                else
                    leaf.counts[i] = calculateNodeCount(leaf.leaves[i], leaf.leaves[i+1]);
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
    void createUnsortedLayout(LeafPart &leaf, InternalPart &inter) {
        for(size_t i = 0; i<nLeaf; i++) {
            key_t key = leaf.leaves[i];
            uint32_t level = enc.getLevel(leaf.leaves[i+1] - key);
            inter.prefixes[i + nInternal] = enc.encodePlaceholderBit(key, level);
            inter.internalToLeaf[i + nInternal] = i + nInternal;

            uint32_t prefixLength = enc.commonPrefix(key, leaf.leaves[i+1]);
            if(prefixLength % 3 == 0 && i < nLeaf - 1) {
                uint32_t octIndex = (i + binaryKeyWeight(key, prefixLength / 3)) / 7;
                inter.prefixes[octIndex] = enc.encodePlaceholderBit(key, prefixLength / 3);
                inter.internalToLeaf[octIndex] = octIndex;
            }
        }
    }

    /// @brief Determine octree subdivision level boundaries
    void getLevelRange(InternalPart &inter) {
        for(uint32_t level = 0; level <= enc.maxDepth(); level++) {
            auto it = std::lower_bound(inter.prefixes.begin(), inter.prefixes.end(), enc.encodePlaceholderBit(0, level));
            inter.levelRange[level] = std::distance(inter.prefixes.begin(), it);
        }
        inter.levelRange[enc.maxDepth() + 1] = nTotal;
    }

    /// @brief Extract parent/child relationships from binary tree and translate to sorted order
    void linkTree(InternalPart &inter) {
        #pragma omp parallel for schedule(static)
            for(int i = 0; i<nInternal; i++) {
                size_t idxA = inter.leafToInternal[i];
                key_t prefix = inter.prefixes[idxA];
                key_t nodeKey = enc.decodePlaceholderBit(prefix);
                unsigned prefixLength = enc.decodePrefixLength(prefix);
                unsigned level = prefixLength / 3;
                if(level >= enc.maxDepth()) {
                    std::cerr << "Max depth exceeded on linkTree" << std::endl;
                    exit(-1);
                }

                key_t childPrefix = enc.encodePlaceholderBit(nodeKey, level + 1);

                size_t leafSearchStart = inter.levelRange[level + 1];
                size_t leafSearchEnd   = inter.levelRange[level + 2];
                auto childIdx = std::distance(inter.prefixes.begin(), 
                    std::lower_bound(inter.prefixes.begin() + leafSearchStart, inter.prefixes.begin() + leafSearchEnd, childPrefix));

                if (childIdx != leafSearchEnd && childPrefix == inter.prefixes[childIdx]) {
                    offsets[idxA] = childIdx;
                    // We only store the parent once for every group of 8 siblings.
                    // This works as long as each node always has 8 siblings.
                    // Subtract one because the root has no siblings.
                    inter.parents[(childIdx - 1) / 8] = idxA;
                }
            } 
    }
    
    /// @brief Computes the ranges of point indexes covered by internal or leafs nodes
    std::pair<size_t, size_t> computeInternalRanges(LeafPart &leaf, InternalPart &inter, uint32_t node = 0) {
        // If node is a leaf, get its internal layout from the two consecutive leafs on the layout array
        if(offsets[node] == 0) {
            internalRanges[node] = std::make_pair(leaf.layout[inter.internalToLeaf[node]], leaf.layout[inter.internalToLeaf[node] + 1]);
            size_t range_size = internalRanges[node].second - internalRanges[node].first;
            return internalRanges[node];
        }

        // Compute recursively (post-order DFS) the count of the internal node. It will be the total range spanned by its children.
        for(uint8_t octant = 0; octant < OCTANTS_PER_NODE; octant++) {
            uint32_t child = offsets[node] + octant;
            auto childLayout = computeInternalRanges(leaf, inter, child);
            if(octant == 0)
                internalRanges[node].first = childLayout.first;
            else if(octant == OCTANTS_PER_NODE-1)
                internalRanges[node].second = childLayout.second;
        }
        return internalRanges[node];
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
    void setupBbox(InternalPart &inter) {
        precomputedRadii = std::vector<Vector>(enc.maxDepth() + 1);
        inter.levelRange = std::vector<size_t>(enc.maxDepth() + 2);
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
    void buildOctreeLeaves(LeafPart &leaf) {
        // Builds the octree sequentially using the cornerstone algorithm
        // We start with 0, UPPER_BOUND on the leaves. Remember that UPPER_BOUND is 100000...000 with as many 0s as MAX_DEPTH, and it can never be reached by a code
        leaf.leaves = std::vector<key_t>{0, enc.upperBound()};
        leaf.counts = {codes.size()};

        while(!updateOctreeLeaves(leaf));

        nLeaf = leaf.leaves.size() - 1;

        // Perform the exclusive scan to get the layout indices (first index in the codes for each leaf)
        leaf.layout.resize(nLeaf+1);
        std::exclusive_scan(leaf.counts.begin(), leaf.counts.end() + 1, leaf.layout.begin(), 0);
    }

    /**
     * @brief Computes the node operations to be done on the leaves and modifies the tree if necessary
     * 
     * @details Convergence is achieved when all the node operations to be done are equal to 1
     */
    bool updateOctreeLeaves(LeafPart &leaf) {
        std::vector<size_t> nodeOps(leaf.leaves.size());
        bool converged = rebalanceDecision(leaf, nodeOps);
        if(!converged) {
            std::vector<key_t> newLeaves;
            rebalanceTree(leaf, newLeaves, nodeOps);
            leaf.counts.resize(newLeaves.size()-1);
            swap(leaf.leaves, newLeaves);
            computeNodeCounts(leaf);
        }
        return converged;
    }

    void resize(InternalPart &inter) {
        // Compute the final sizes of the octree
        nInternal = (nLeaf - 1) / 7;
        nTotal = nLeaf + nInternal;
        // Resize the other fields
        inter.prefixes.resize(nTotal);
        offsets.resize(nTotal+1);
        inter.parents.resize((nTotal-1) / 8);
        inter.internalToLeaf.resize(nTotal);
        inter.leafToInternal.resize(nTotal);
        centers.resize(nTotal);
        internalRanges.resize(nTotal);
    }

    /**
     * @brief Builds the internal part of the octree and links the nodes
     * 
     * @details Follows the process indicated in the cornerstone octree paper, section 5. 
     */
    void buildOctreeInternal(LeafPart &leaf, InternalPart &inter) {
        createUnsortedLayout(leaf, inter);
        // Sort by key where the keys are the prefixes and the values to sort internalToLeaf
        std::vector<std::pair<key_t, uint32_t>> prefixes_internalToLeaf(nTotal);
        for(int i = 0; i<nTotal; i++) {
            prefixes_internalToLeaf[i] = {inter.prefixes[i], inter.internalToLeaf[i]};
        }
        std::stable_sort(prefixes_internalToLeaf.begin(), prefixes_internalToLeaf.end(), [](const auto &t1, const auto &t2) {
            return t1.first < t2.first;
        });

        for(int i = 0; i<nTotal; i++) {
            inter.prefixes[i] = prefixes_internalToLeaf[i].first;
            inter.internalToLeaf[i] = prefixes_internalToLeaf[i].second;
        }
        prefixes_internalToLeaf.clear();

        // Compute the reverse mapping leafToInternal
        for (uint32_t i = 0; i < nTotal; ++i) {
            inter.leafToInternal[inter.internalToLeaf[i]] = i;
        }

        // Offset by the number of internal nodes
        for (uint32_t i = 0; i < nTotal; ++i) {
            inter.internalToLeaf[i] -= nInternal;
        }

        // Find the LO array
        getLevelRange(inter);

        // Clear child offsets
        std::fill(offsets.begin(), offsets.end(), 0);

        // Compute the links
        linkTree(inter);

        // Compute internal node layouts
        computeInternalRanges(leaf, inter);
    }

    /**
     * @brief Computes the octree geometry (the centers and radii of each internal node and leaf)
     * 
     * @details We do this to allow for faster traversals, however this is not strictly necessary
     * and could be removed if memory becomes a constraint. This stuff can be computed on the fly
     * during neighbourhood searches.
     */
    void computeGeometry(InternalPart &inter) {
        #pragma omp parallel for schedule(static)
            for(uint32_t i = 0; i<inter.prefixes.size(); i++) {
                key_t prefix = inter.prefixes[i];
                key_t startKey = enc.decodePlaceholderBit(prefix);
                uint32_t level = enc.decodePrefixLength(prefix) / 3;
                centers[i] = enc.getCenter(startKey, level, box, halfLengths, precomputedRadii);
            }
    }

    void buildOctree(std::shared_ptr<BuildLog> log = nullptr) {
        // Temporal structs for the variables in the leaf and internal parts of the octree that we wont use during searches
        LeafPart leaf;
        InternalPart inter;

        setupBbox(inter);

        TimeWatcher tw;

        // Leaf construction
        tw.start();
        buildOctreeLeaves(leaf);
        tw.stop();
        if(log)
            log->linearOctreeLeafTime = tw.getElapsedDecimalSeconds();
        
        // Internal part construction
        tw.start();
        resize(inter);
        buildOctreeInternal(leaf, inter);
        computeGeometry(inter);
        tw.stop();
        if(log)
            log->linearOctreeInternalTime = tw.getElapsedDecimalSeconds();

        // Output extra info from the build
        if (log) {
            log->buildTime = log->linearOctreeLeafTime + log->linearOctreeInternalTime;
            log->maxLeafPoints = mainOptions.maxPointsLeaf;
            log->memoryUsed = computeMemoryFootprint();
            log->totalNodes = nTotal;
            log->leafNodes = nLeaf;
            log->internalNodes = nInternal;
            log->maxDepthSeen = maxDepthSeen;
            log->minRadiusAtMaxDepth = std::min(
                precomputedRadii[maxDepthSeen].getX(), std::min(
                precomputedRadii[maxDepthSeen].getY(), 
                precomputedRadii[maxDepthSeen].getZ()
            ));
        }
    }

public:    
    /**
     * @brief Builds the linear octree given an array of points, also reporting how much time each step takes
     * 
     * @details The points will be sorted in-place by the order given by the encoding to allow
     * spatial data locality
     */
    explicit LinearOctree(Container& points,
                        std::vector<key_t>& codes,
                        Box box,
                        PointEncoder& enc,
                        std::shared_ptr<BuildLog> log = nullptr)
        : points(points), enc(enc), codes(codes), box(box)  {
        static_assert(!std::is_same_v<std::decay_t<PointEncoder>, PointEncoding::NoEncoding>,
            "Encoder cannot be an instance of NoEncoding when using LinearOctree.");
        buildOctree(log);
    }

    /// @brief Returns the memory overhead of the octree (points cloud or codes are not counted)
    size_t computeMemoryFootprint() const {
        size_t memory = 0;
        // Base size of the structure
        memory += sizeof(LinearOctree);
        // Size of each of the 4 arrays used in neighbourhood searches (the others are deleted after build) 
        memory += vectorMemorySize(internalRanges);
        memory += vectorMemorySize(offsets);
        memory += vectorMemorySize(centers);
        memory += vectorMemorySize(precomputedRadii);
        return memory;
    }

    /// @brief Computes the density of the point cloud as number of points / dataset
    double getDensity() const {
        return (double) points.size() / (box.radii().getX() * box.radii().getY() * box.radii().getZ() * 8.0f);
    }

    /**
     * @brief Helper function to traverse the octree in a single pass
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
        bool descend = continuationCriterion(0, 0);
        if (!descend) return;

        if (offsets[0] == 0) {
            // Root node is already a leaf
            endpointAction(0);
            return;
        }

        std::pair<uint32_t, uint32_t> stack[128]; // node, depth
        stack[0] = {0, 0};

        uint32_t stackPos = 1;
        uint32_t node = 0; // Start at the root
        uint32_t currDepth = 0;
        do {
            for (int octant = 0; octant < OCTANTS_PER_NODE; ++octant) {
                uint32_t child = offsets[node] + octant;
                bool descend = continuationCriterion(child, currDepth + 1);
                if (descend) {
                    if (offsets[child] == 0) {
                        // Leaf node reached
                        endpointAction(child);
                    } else {
                        assert(stackPos < 128);
                        stack[stackPos++] = {child, currDepth + 1};
                    }
                }
            }
            std::tie(node, currDepth) = stack[--stackPos];
        } while (node != 0); // The root node is obtained, search finished
    }

    /**
     * @brief Search neighbors function. Similar to neighborsPrune(), but returns a list-of-ranges wrapper structure containing the neighbours. 
     * This structure implements a forward iterator and can thus be used in a loop. It is way faster as it does not need to copy
     * pointers to each individual point found. 
     * @param k specific kernel that contains the data of the region (center and radius)
     * @return Points inside the given kernel type.
     */
    template<typename Kernel>
    [[nodiscard]] NeighborSet<Container> neighborsStruct(const Kernel& k) const {
        NeighborSet<Container> result(&points);
        auto checkBoxIntersect = [&](uint32_t nodeIndex, uint32_t currDepth) {
            auto nodeCenter = this->centers[nodeIndex];
            auto nodeRadii = this->precomputedRadii[currDepth];
            switch (k.boxIntersect(nodeCenter, nodeRadii)) {
                case KernelAbstract::IntersectionJudgement::INSIDE: {
                    // Completely inside, add octant to the result
                    result.addRange(internalRanges[nodeIndex].first, internalRanges[nodeIndex].second);
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
            size_t startIndex = this->internalRanges[nodeIndex].first;
            size_t endIndex = this->internalRanges[nodeIndex].second;
            size_t rangeStart = startIndex;
            
            for (size_t i = startIndex; i < endIndex; ++i) {
                if (!k.isInside(points[i])) {
                    if (rangeStart < i)
                        result.addRange(rangeStart, i);
                    rangeStart = i + 1;
                }
            }
            
            if (rangeStart < endIndex)
                result.addRange(rangeStart, endIndex);
        };
        
        singleTraversal(checkBoxIntersect, findAndInsertPoints);
        return result;
	}


    /**
     * @brief Search neighbors function. Given kernel that already contains a point and a radius, return the points inside the region.
     * Note that neighborsStruct is generally faster, so only use this function if you don't want your result encapsultated on NeighborSet.
     * 
     * @param k specific kernel that contains the data of the region (center and radius)
     * @return Points inside the given kernel type.
     */
    template<typename Kernel>
    [[nodiscard]] std::vector<size_t> neighborsPrune(const Kernel& k) const {
        std::vector<size_t> ptsInside;
        auto checkBoxIntersect = [&](uint32_t nodeIndex, uint32_t currDepth) {
            auto nodeCenter = this->centers[nodeIndex];
            auto nodeRadii = this->precomputedRadii[currDepth];
            switch (k.boxIntersect(nodeCenter, nodeRadii)) {
                case KernelAbstract::IntersectionJudgement::INSIDE: {
                    // Completely inside, all add points and prune
                    size_t startIndex = this->internalRanges[nodeIndex].first;
                    size_t endIndex = this->internalRanges[nodeIndex].second;
                    for (size_t i = startIndex; i < endIndex; ++i) {
                        ptsInside.push_back(i);
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
            size_t startIndex = this->internalRanges[nodeIndex].first;
            size_t endIndex = this->internalRanges[nodeIndex].second;
            for (size_t i = startIndex; i < endIndex; ++i) {
                if (k.isInside(points[i])) {
                    ptsInside.push_back(i);
                }
            }
        };
        
        singleTraversal(checkBoxIntersect, findAndInsertPoints);
        return ptsInside;
	}

    /**
     * @brief Search neighbors function. Basic implementation without "octant contained in kernel" check. Note that this implementation
     * is slower than all the others, only use if you wanna apply an arbitrary condition predicate apart from the kernel.
     * neighborsPrune could also be adapted to take an arbitrary predicate if needed.
     * 
     * @param k specific kernel that contains the data of the region (center and radius)
     * @param condition function that takes a candidate neighbor point and imposes an additional condition (should return a boolean).
     * The signature of the function should be equivalent to `bool cnd(const typename &p);`
     * @return Points inside the given kernel type and satisfying condition.
     */
    template<typename Kernel, typename Function>
    [[nodiscard]] std::vector<size_t> neighbors(const Kernel& k, Function&& condition) const {
        std::vector<size_t> ptsInside;

        auto intersectsKernel = [&](uint32_t nodeIndex, uint32_t nodeDepth) {
            return k.boxOverlap(this->centers[nodeIndex], this->precomputedRadii[nodeDepth]);
        };
        
        auto findAndInsertPoints = [&](uint32_t nodeIndex) {
            // Reached a leaf, add all points inside the kernel
            size_t startIndex = this->internalRanges[nodeIndex].first;
            size_t endIndex = this->internalRanges[nodeIndex].second;
            for (size_t i = startIndex; i < endIndex; ++i) {
                if (k.isInside(points[i]) && condition(points[i])) {
                    ptsInside.push_back(i);
                }
            }
        };
        singleTraversal(intersectsKernel, findAndInsertPoints);
        return ptsInside;
	}

    // Helper for kNN searches
    struct OctantPointIndex {
        uint64_t index        : 58;
        uint64_t depth        : 5;
        uint64_t isOctant     : 1;
    
        double sqDist = 0.0;
    
        OctantPointIndex(size_t idx, bool octant, uint8_t depth_, double dist) 
            : index(0), depth(0), isOctant(0), sqDist(dist) {
            set(idx, octant, depth_);
        }
    
        void set(size_t idx, bool octant, uint8_t octantDepth = 0) {
            index = idx & ((1ULL << 58) - 1);
            depth = octantDepth & 0x1F;
            isOctant = octant ? 1 : 0;
        }
    
        bool operator<(const OctantPointIndex& other) const {
            // Note: std::priority_queue uses max heap by default, so invert comparison for min-heap behavior
            return this->sqDist > other.sqDist;
        }
    };
    
    // Helper for kNN searches
    inline static double distPointsSquared(const Point &p, const Point &q) {
        // compute sq distance between the points
        double dx = p.getX() - q.getX();
        double dy = p.getY() - q.getY();
        double dz = p.getZ() - q.getZ();

        return dx * dx + dy * dy + dz * dz;
    }

    // Helper for kNN searches
    inline static double distPointOctantSquared(const Point& p, const Point& octCenter, const Vector& octRadius) {
        // Extract octant bounds
        const double minX = octCenter.getX() - octRadius.getX();
        const double minY = octCenter.getY() - octRadius.getY();
        const double minZ = octCenter.getZ() - octRadius.getZ();
        const double maxX = octCenter.getX() + octRadius.getX();
        const double maxY = octCenter.getY() + octRadius.getY();
        const double maxZ = octCenter.getZ() + octRadius.getZ();
    
        const double px = p.getX();
        const double py = p.getY();
        const double pz = p.getZ();
    
        // Clamp point to octant bounds to get the intersection with it
        const double cx = std::max(minX, std::min(px, maxX));
        const double cy = std::max(minY, std::min(py, maxY));
        const double cz = std::max(minZ, std::min(pz, maxZ));
        
        // Compute the distance to that intersection point
        const double dx = px - cx;
        const double dy = py - cy;
        const double dz = pz - cz;
    
        return dx * dx + dy * dy + dz * dz;
    }

    /**
     * 
     * @brief kNN search function. A priority queue with distances to center, ascending order, both octants and points can be inside it.
     * 
     * @details For octants, store the cube-to-point distance
     *  -> this is computed by observing that the closest point to it is clamp(p_c, c_min, c_max) (for each coordinate c \in {x,y,z})
     * for points, store the regular point-to-point distance 
     * 
     * algo:
     *  1. extract head of queue
     *  2. if head is point, insert it into result
     *  3. if head is octant, then:
     *  3.1 if it is a leaf, push every point into queue
     *  3.2 if it is an internal node, push the 8 suboctants with their min distances to p
     * 
     * @param[in] Point p the center of the searches
     * @param[out] indexes the vector of point cloud indexes returned
     * @param[out] dists the vector of point cloud distances returned
     */
    size_t knn(const Point& p, const size_t k, std::vector<size_t> &indexes, std::vector<double> &dists) const {
        // Initialize the min-distances heap with the root octant
        std::vector<OctantPointIndex> heap;
        heap.reserve(std::max((size_t) 512, k / 2));
        std::make_heap(heap.begin(), heap.end());
        heap.emplace_back(0, true, 0, 0.0);

        // Extract nearest octant/point until we insert k points
        // NOTE: No need to check heap.empty() since 
        // we have, at the very least, maxPoints points and 1 octant
        size_t inserted = 0, maxPoints = std::min(points.size(), k);
        while(inserted < maxPoints) {
            std::pop_heap(heap.begin(), heap.end());
            OctantPointIndex top = heap.back();
            heap.pop_back();
            if(top.isOctant) {
                if(offsets[top.index] == 0) {
                    // Leaf node, push points into the queue
                    size_t startIndex = this->internalRanges[top.index].first;
                    size_t endIndex = this->internalRanges[top.index].second;
                    for (size_t i = startIndex; i < endIndex; ++i) {
                        double sqDist = distPointsSquared(p, points[i]);
                        heap.emplace_back(i, false, 0, sqDist);
                        std::push_heap(heap.begin(), heap.end());
                    }
                } else {
                    // Internal node, push child octants into the queue
                    uint8_t childDepth = top.depth + 1;
                    for (int octant = 0; octant < OCTANTS_PER_NODE; ++octant) {
                        size_t childOctIndex = offsets[top.index] + octant;
                        double sqDist = distPointOctantSquared(p, centers[childOctIndex], precomputedRadii[childDepth]);
                        heap.emplace_back(childOctIndex, true, childDepth, sqDist);
                        std::push_heap(heap.begin(), heap.end());
                    }
                }   
            } else {
                // Insert point directly into result
                indexes[inserted] = top.index;
                dists[inserted] = top.sqDist;
                inserted++;
            }
        }
        return inserted;
	}

    // Helper for approx searches
    uint32_t getPrecisionLevel(const Vector &toleranceVector) const {
        assert(toleranceVector.getX() > 0 && toleranceVector.getY() > 0 && toleranceVector.getZ() > 0 && "tolerance vector must be >0 in every coordinate");
        uint32_t precisionLevel = maxDepthSeen;
        for(uint32_t i = 0; i<maxDepthSeen; i++) {
            Vector radii = precomputedRadii[i];
            if(radii.getX() < toleranceVector.getX() && radii.getY() < toleranceVector.getY() && radii.getZ() && toleranceVector.getZ()) {
                precisionLevel = i;
                break;
            }
        }
        return precisionLevel;
    }

    // Helper for approx searches
    uint32_t getPrecisionLevel(double tolerance) const {
        assert(tolerance > 0 && "tolerance must be greater than 0");
        return getPrecisionLevel(Vector{tolerance, tolerance, tolerance});
    }

    // Helper for approx searches
    uint32_t getPrecisionLevel(const Vector &kernelRadii, double tolerancePercentage) const {
        assert(tolerancePercentage > 0.0 && "tolerance percentage must be greater than 0");
        return getPrecisionLevel(Vector{
                kernelRadii.getX() * tolerancePercentage / 100.0, 
                kernelRadii.getY() * tolerancePercentage / 100.0, 
                kernelRadii.getZ() * tolerancePercentage / 100.0
        });
    }

    /**
     * @brief Search neighbors function. Same algorithm as neighborsStruct(), but the search is cut until a certain depth is reached.
     * @param[in] precisionLevel The maximum depth of the search
     * @param[in] Whether to include the points found an an intersecting leaf on the maximum level, or not.
     * @return Points inside the given kernel type.
     */
    template<typename Kernel>
    [[nodiscard]] NeighborSet<Container> neighborsApprox(const Kernel& k, uint32_t precisionLevel, bool upperBound) const {
        NeighborSet<Container> result(&points);
        auto checkBoxIntersect = [&](uint32_t nodeIndex, uint32_t currDepth) {
            auto nodeCenter = this->centers[nodeIndex];
            auto nodeRadii = this->precomputedRadii[currDepth];
            if(currDepth > precisionLevel) {
                // If we are beyond precision and in upper bound mode, add octant to the result
                if(upperBound) {
                    result.addRange(internalRanges[nodeIndex].first, internalRanges[nodeIndex].second);
                }
                // Prune either way
                return false;
            } 

            switch (k.boxIntersect(nodeCenter, nodeRadii)) {
                case KernelAbstract::IntersectionJudgement::INSIDE: {
                    // Completely inside, add octant to the result
                    result.addRange(internalRanges[nodeIndex].first, internalRanges[nodeIndex].second);
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
            size_t startIndex = this->internalRanges[nodeIndex].first;
            size_t endIndex = this->internalRanges[nodeIndex].second;

            size_t rangeStart = startIndex;
            size_t rangeEnd = startIndex;

            for (size_t i = startIndex; i < endIndex; ++i, ++rangeEnd) {
                if (!k.isInside(points[i])) {
                    if (rangeStart < rangeEnd) {
                        // Store the last valid range [rangeStart, rangeEnd)
                        result.addRange(rangeStart, rangeEnd);
                    }
                    rangeStart = rangeEnd + 1;  // start next range after current point
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

    // Overrides for working with a given radius/vector of radii before calling the actual search methods
    template<Kernel_t kernel_type = Kernel_t::square>
	[[nodiscard]] inline NeighborSet<Container> neighborsStruct(const Point& p, double radius) const {
		const auto kernel = kernelFactory<kernel_type>(p, radius);
		return neighborsStruct(kernel);
	}

	template<Kernel_t kernel_type = Kernel_t::cube>
	[[nodiscard]] inline NeighborSet<Container> neighborsStruct(const Point& p, const Vector& radii) const {
		const auto kernel = kernelFactory<kernel_type>(p, radii);
		return neighborsStruct(kernel);
	}

	template<Kernel_t kernel_type = Kernel_t::square>
	[[nodiscard]] inline std::vector<size_t> neighborsPrune(const Point& p, double radius) const {
		const auto kernel = kernelFactory<kernel_type>(p, radius);
		return neighborsPrune(kernel);
	}

	template<Kernel_t kernel_type = Kernel_t::cube>
	[[nodiscard]] inline std::vector<size_t> neighborsPrune(const Point& p, const Vector& radii) const {
		const auto kernel = kernelFactory<kernel_type>(p, radii);
		return neighborsPrune(kernel);
	}

    template<Kernel_t kernel_type = Kernel_t::square>
	[[nodiscard]] inline std::vector<size_t> neighbors(const Point& p, double radius) const {
		const auto kernel = kernelFactory<kernel_type>(p, radius);
		constexpr auto dummyCondition = [](const Point&) { return true; };
		return neighbors(kernel, dummyCondition);
	}

	template<Kernel_t kernel_type = Kernel_t::cube>
	[[nodiscard]] inline std::vector<size_t> neighbors(const Point& p, const Vector& radii) const {
		const auto kernel = kernelFactory<kernel_type>(p, radii);
		constexpr auto dummyCondition = [](const Point&) { return true; };
		return neighbors(kernel, dummyCondition);
	}

	template<Kernel_t kernel_type = Kernel_t::square, class Function>
	[[nodiscard]] inline std::vector<size_t> neighbors(const Point& p, double radius, Function&& condition) const {
		const auto kernel = kernelFactory<kernel_type>(p, radius);
		return neighbors(kernel, std::forward<Function&&>(condition));
	}

	template<Kernel_t kernel_type = Kernel_t::square, class Function>
	[[nodiscard]] inline std::vector<size_t> neighbors(const Point& p, const Vector& radii,
	                                                          Function&& condition) const {
		const auto kernel = kernelFactory<kernel_type>(p, radii);
		return neighbors(kernel, std::forward<Function&&>(condition));
	}

    template<Kernel_t kernel_type = Kernel_t::square>
	[[nodiscard]] inline NeighborSet<Container> neighborsApprox(const Point& p, double radius, double tolerancePercentage, bool upperBound) const {
		const auto kernel = kernelFactory<kernel_type>(p, radius);
		return neighborsApprox(kernel, tolerancePercentage, upperBound);
	}

	template<Kernel_t kernel_type = Kernel_t::cube>
	[[nodiscard]] inline NeighborSet<Container> neighborsApprox(const Point& p, const Vector& radii, double tolerancePercentage, bool upperBound) const {
		const auto kernel = kernelFactory<kernel_type>(p, radii);
		return neighborsApprox(kernel, tolerancePercentage, upperBound);
	}

    template<Kernel_t kernel_type = Kernel_t::square>
	[[nodiscard]] inline NeighborSet<Container> neighborsApprox(const Point& p, double radius, const Vector &toleranceVector, bool upperBound) const {
		const auto kernel = kernelFactory<kernel_type>(p, radius);
		return neighborsApprox(kernel, toleranceVector, upperBound);
	}

	template<Kernel_t kernel_type = Kernel_t::cube>
	[[nodiscard]] inline NeighborSet<Container> neighborsApprox(const Point& p, const Vector& radii, const Vector &toleranceVector, bool upperBound) const {
		const auto kernel = kernelFactory<kernel_type>(p, radii);
		return neighborsApprox(kernel, toleranceVector, upperBound);
	}



    // Debugging utilities
    void printKey(key_t key) {
        for(int i=20; i>=0; i--) {
            std::cout << std::bitset<3>(key >> (3*i)) << " ";
        }

        std::cout << std::endl;
    }

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

    void logOctree(std::ofstream &file, std::ofstream &pointsFile, LeafPart &leaf, InternalPart &inter) {
        std::cout << "(1/2) Logging octree parameters and structure" << std::endl;
        std::string encoderTypename = enc.getEncoderName();
        file << "---- Linear octree parameters ----";
        file << "Encoder: " << encoderTypename << "\n";
        file << "Max. points per leaf: " << mainOptions.maxPointsLeaf << "\n";
        file << "Total number of nodes = " << nTotal << "\n Leafs = " << nLeaf << "\n Internal nodes = " << nInternal << "\n";
        file << "---- Full structure ----";
        writeVectorBinary(file, leaf.leaves, "leaves");
        writeVector(file, leaf.counts, "counts");
        writeVector(file, leaf.layout, "layout");
        writeVectorBinary(file, inter.prefixes, "prefixes");
        writeVector(file, offsets, "offsets");
        writeVector(file, inter.parents, "parents");
        writeVector(file, inter.levelRange, "levelRange");
        writeVector(file, inter.internalToLeaf, "internalToLeaf");
        writeVector(file, inter.leafToInternal, "leafToInternal");
        writeVectorPairs(file, internalRanges, "internalRanges");
        file << std::flush;
        
        // write codes and point coordinates
        std::cout << "(2/2) Logging encoded points" << std::endl;
        writePointsAndCodes(pointsFile, encoderTypename);
        pointsFile << std::flush;
        std::cout << "Done! Octree and points logged" << std::endl;
    }
    
    void logOctreeBounds(std::ofstream &outputFile, int max_level) {
        outputFile << "level,upx,upy,upz,downx,downy,downz\n";
        auto logBounds = [&](uint32_t nodeIndex, uint32_t currDepth) {
            auto nodeCenter = this->centers[nodeIndex];
            auto nodeRadii = this->precomputedRadii[currDepth];
            auto up = nodeCenter + nodeRadii;
            auto down = nodeCenter - nodeRadii;
            outputFile << currDepth << "," << up.getX() << "," << up.getY() << "," << up.getZ() << "," 
                << down.getX() << "," << down.getY() << "," << down.getZ() << "\n";

            return currDepth+1 <= max_level;
        };
        
        auto logBoundsLeaf = [&](uint32_t nodeIndex) {
            
        };
        
        singleTraversal(logBounds, logBoundsLeaf);
	}
};