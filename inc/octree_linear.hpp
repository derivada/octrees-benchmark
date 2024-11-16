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
* @brief Another (more correct) linear octree implementation based on the excellent implementation done 
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
    
    // Compute the array of rebalancing decisions (g1)
    bool rebalanceDecision(std::vector<uint32_t> &nodeOps) {
        bool converged = true;
        for(int i = 0; i<leaves.size()-1; i++) {
            nodeOps[i] = calculateNodeOp(i);
            if(nodeOps[i] != 1) converged = false;
        }
        return converged;
    } 

    // Calculate the operation to be done in this node
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

    // Get the sibling ID and level of the node in the octree
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

    
    static void printMortonCode(morton_t code) {
        // Print the bits in groups of 3 to represent each level
        std::cout << std::bitset<1>(code >> 63) << " ";
        for (int i = 62; i >= 0; i -= 3) {
            std::cout << std::bitset<3>((code >> (i - 2)) & 0b111) << " ";
        }
        std::cout << std::endl;
    }

    // Build the new tree using the rebalance decision array
    void rebalanceTree(std::vector<morton_t> &newTree, std::vector<uint32_t> &nodeOps) {
        uint32_t n = leaves.size() - 1;

        // g2, exclusive scan
        exclusiveScan(nodeOps.data(), n+1);

        newTree.resize(nodeOps[n] + 1);
        newTree.back() = leaves.back();
        for (uint32_t i = 0; i < n; ++i) {
            processNode(i, nodeOps, newTree);
        }
    }

    // Construct new octree value for the given index
    void processNode(uint32_t index, std::vector<uint32_t> &nodeOps, std::vector<morton_t> &newTree) {
        morton_t node = leaves[index];
        morton_t range = leaves[index+1] - node;

        uint32_t level = MortonEncoder::getLevel(range);

        uint32_t opCode       = nodeOps[index + 1] - nodeOps[index]; // The original value of the opCode (before exclusive scan)
        uint32_t newNodeIndex = nodeOps[index]; // The new position to put the node into (nodeOps value after exclusive scan)

        if(opCode == 1) {
            // do nothing, just copy node into new position
            newTree[newNodeIndex] = node;
            // assert(MortonEncoder::isPowerOf8(newTree[newNodeIndex + 1] - newTree[newNodeIndex]));
        } else if(opCode == 8) {
            // Split the node into 8
            for(int sibling = 0; sibling < OCTANTS_PER_NODE; sibling++) {
                newTree[newNodeIndex + sibling] = node + sibling * MortonEncoder::nodeRange(level + 1);
            }
            // assert(MortonEncoder::isPowerOf8(newTree[newNodeIndex + 8] - newTree[newNodeIndex + 7]));
        } else {
            // TODO: higher order splits
            uint32_t levelDiff = MortonEncoder::log8ceil(opCode);
            for (int sibling = 0; sibling < opCode; ++sibling) {
                newTree[newNodeIndex + sibling] = node + sibling * MortonEncoder::nodeRange(level + levelDiff);
            }
        }
    }

    // Count number of particles in each octree node
    void computeNodeCounts() {
        uint32_t n = leaves.size() - 1;
        uint32_t codes_size = codes.size();
        uint32_t firstNode = 0;
        uint32_t lastNode = n;

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

        // TODO: count guessing, parallelizing
        size_t nNonZeroNodes = lastNode - firstNode;
        exclusiveScan(counts.data() + firstNode, nNonZeroNodes);

        for(uint32_t i = 0; i<nNonZeroNodes; i++) {
            counts[i + firstNode] = calculateNodeCount(leaves[i+firstNode], leaves[i+firstNode+1]);
        }
    }
    unsigned calculateNodeCount(morton_t keyStart, morton_t keyEnd) {
        auto rangeStart = std::lower_bound(codes.begin(), codes.end(), keyStart);
        auto rangeEnd   = std::lower_bound(codes.begin(), codes.end(), keyEnd);
        size_t count    = rangeEnd - rangeStart;
        // TODO: should we use maxCount??
        return count;
    }

    template<class T>
    void exclusiveScan(T* out, size_t numElements) {
        exclusiveScanSerialInplace(out, numElements, T(0));
    }

    template<class T>
    T exclusiveScanSerialInplace(T* out, size_t num_elements, T init)
    {
        T a = init;
        T b = init;
        for (size_t i = 0; i < num_elements; ++i)
        {
            a += out[i];
            out[i] = b;
            b      = a;
        }
        return b;
    }

    constexpr uint32_t binaryKeyWeight(morton_t key, unsigned level)
    {
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
     * @brief Build 
     * @
     */
    explicit LinearOctree(std::vector<Lpoint> &points): points(points) {
        std::cout << "Linear octree build summary:\n";
        double total_time;
        TimeWatcher tw;

        tw.start();
        setupBbox();
        tw.stop();
        total_time += tw.getElapsedDecimalSeconds();
        std::cout << "  Time to find bounding box: " << tw.getElapsedDecimalSeconds() << " seconds\n";

        tw.start();
        sortPoints();
        tw.stop();
        total_time += tw.getElapsedDecimalSeconds();
        std::cout << "  Time to sort the points by their morton codes: " << tw.getElapsedDecimalSeconds() << " seconds\n";

        tw.start();
        buildOctreeLeaves();
        tw.stop();
        total_time += tw.getElapsedDecimalSeconds();
        std::cout << "  Time to build the octree leaves: " << tw.getElapsedDecimalSeconds() << " seconds\n";
        
        tw.start();
        resize();
        tw.stop();
        total_time += tw.getElapsedDecimalSeconds();
        std::cout << "  Time to allocate space for internal variables: " << tw.getElapsedDecimalSeconds() << " seconds\n";

        tw.start();
        buildOctreeInternal();
        tw.stop();
        total_time += tw.getElapsedDecimalSeconds();
        std::cout << "  Time to build internal part of the octree and link it: " << tw.getElapsedDecimalSeconds() << " seconds\n";

        tw.start();
        computeGeometry();
        tw.stop();
        total_time += tw.getElapsedDecimalSeconds();
        std::cout << "  Time to compute octree geometry (centers and radii): " << tw.getElapsedDecimalSeconds() << " seconds\n";

        std::cout << "Total time to build linear octree: " << total_time << " seconds\n";
    }

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

    void sortPoints() {
        std::vector<std::pair<morton_t, Lpoint>> encoded_points;
        encoded_points.reserve(points.size());
        for(size_t i = 0; i < points.size(); i++) {
            encoded_points.emplace_back(MortonEncoder::encodeMortonPoint(points[i], bbox), points[i]);
        }

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

    void buildOctreeLeaves() {
        // Builds the octree sequentially using the cornerstone algorithm

        // We start with 0, 7777...777 (in octal)
        leaves = {0, MortonEncoder::UPPER_BOUND};
        counts = {(uint32_t) codes.size()};

        while(!updateOctreeLeaves())
            ;
        

        // Compute the final sizes of the octree
        nLeaf = leaves.size() - 1; // TODO: shouldnt this be -1?
        nInternal = (nLeaf - 1) / 7;
        nTotal = nLeaf + nInternal;

        // Perform the exclusive scan to get the layout indices
        layout.resize(leaves.size());
        std::exclusive_scan(counts.begin(), counts.end() + 1, layout.begin(), 0);
    }

    bool updateOctreeLeaves() {
        std::vector<uint32_t> nodeOps(leaves.size());
        bool converged = rebalanceDecision(nodeOps);

        std::vector<morton_t> newTree;
        rebalanceTree(newTree, nodeOps);
        counts.resize(newTree.size()-1);
        swap(leaves, newTree);

        computeNodeCounts();
        return converged;
    }

    void resize() {
        // Resize the other fields
        prefixes.resize(nTotal);
        offsets.resize(nTotal+1);
        parents.resize((nTotal-1) / 8);
        internalToLeaf.resize(nTotal);
        leafToInternal.resize(nTotal);
        centers.resize(nTotal);
        radii.resize(nTotal);
    }

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

    // Computes the node centers and radii
    void computeGeometry() {
        for(uint32_t i = 0; i<prefixes.size(); i++) {
            morton_t prefix = prefixes[i];
            morton_t startKey = MortonEncoder::decodePlaceholderBit(prefix);
            uint32_t level = MortonEncoder::decodePrefixLength(prefix) / 3;
            std::tie(centers[i], radii[i]) = MortonEncoder::getCenterAndRadii(startKey, level, bbox, halfLengths, precomputedRadii);
        }
    }

    void printArray(std::vector<uint32_t> &arr) {
        for(int i = 0; i<arr.size();i++) {
            std::cout << i << " -> " << arr[i] << std::endl;
        }
    }
    void printArrayMortonCodes(std::vector<morton_t> &arr){
        for(int i = 0; i<arr.size();i++) {
            std::cout << i << " -> "; printMortonCode(arr[i]);
        }
    }

    // A generic traversal along the tree
    template<class C, class A>
    void singleTraversal(C&& continuationCriterion, A&& endpointAction) const {
        bool descend = continuationCriterion(0);
        if (!descend) return;

        if (offsets[0] == 0) {
            // root node is already the endpoint
            endpointAction(0);
            return;
        }

        uint32_t stack[128];
        stack[0] = 0;

        uint32_t stackPos = 1;
        uint32_t node = 0; // start at the root

        do {
            for (int octant = 0; octant < OCTANTS_PER_NODE; ++octant) {
                uint32_t child = offsets[node] + octant;
                bool descend = continuationCriterion(child);
                if (descend) {
                    if (offsets[child] == 0) {
                        // endpoint reached with child is a leaf node
                        endpointAction(child);
                    } else {
                        assert(stackPos < 128);
                        stack[stackPos++] = child; // push
                    }
                }
            }
            node = stack[--stackPos];

        } while (node != 0); // the root can only be obtained when the tree has been fully traversed
    }

    template<typename Kernel, typename Function>
    [[nodiscard]] std::vector<Lpoint*> neighbors(const Kernel& k, Function&& condition, morton_t root = 0) const
    /**
     * @brief Search neighbors function. Given kernel that already contains a point and a radius, return the points inside the region.
     * @param k specific kernel that contains the data of the region (center and radius)
     * @param condition function that takes a candidate neighbor point and imposes an additional condition (should return a boolean).
     * The signature of the function should be equivalent to `bool cnd(const Lpoint &p);`
     * @param root The morton code for the node to start (usually the tree root which is 0)
     * @return Points inside the given kernel type. Actually the same as ptsInside.
     */
	{
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

    template<Kernel_t kernel_type = Kernel_t::square>
	[[nodiscard]] inline std::vector<Lpoint*> searchNeighbors(const Point& p, double radius) const
	/**
     * @brief Search neighbors function. Given a point and a radius, return the points inside a given kernel type
     * @param p Center of the kernel to be used
     * @param radius Radius of the kernel to be used
     * @return Points inside the given kernel type
     */
	{
		const auto kernel = kernelFactory<kernel_type>(p, radius);
		// Dummy condition that always returns true, so we can use the same function for all cases
		// The compiler should optimize this away
		constexpr auto dummyCondition = [](const Lpoint&) { return true; };

		return neighbors(kernel, dummyCondition);
	}

    [[nodiscard]] inline std::vector<Lpoint*> searchSphereNeighbors(const Point& point, const float radius) const
	{
		return searchNeighbors<Kernel_t::sphere>(point, radius);
	}
};