/**
 * A linear (map-based) implementation of the Octree using Morton codes for quick access with good spacial locality
 * 
 * Pablo Díaz Viñambres 22/10/24
 * 
 */


#pragma once

#include "Lpoint.hpp"
#include "Box.hpp"
#include <stack>
#include <bitset>
#include <unordered_map>
#include <filesystem>
#include "libmorton/morton.h"
#include "octree_linear_node.hpp"
#include "NeighborKernels/KernelFactory.hpp"
#include "TimeWatcher.hpp"

class LinearOctree {
private:
    static constexpr unsigned int MAX_POINTS        = 100;
	static constexpr float        MIN_OCTANT_RADIUS = 0.1;
    static constexpr unsigned int MAX_DEPTH         = 19;
	static constexpr size_t       DEFAULT_KNN       = 100;
	static constexpr short        OCTANTS_PER_NODE  = 8;

    /* 
     * Linear Octree
     * 
     * To construct the Morton encodings for a point (p1,p2,p3) on a depth d, we first find its anchor (its scaled integer coordinates that
     * indicate the node center it belongs to) by scaling the point into the cube [0,1]^3, multiplying each coordinate by 2^d and flooring.
     * 
     * Once we have the anchor (x,y,z) at depth d, we Morton encode it by interleaving its most significant 21 bits on each coordinate
     * For this purpose we use libmorton, https://github.com/Forceflow/libmorton/tree/main/include/libmorton
     * 
     * The encoding produces a 64 bit number, however the maximum depth is only 19 instead of 21. This is because we also need to store the depth
     * of the node to form the octree. To do this, we need at least 5 bits, but since another two are unused, we shift a total of 7 bits and put
     * the current depth at those last 7 bits.   
     * 
     * In summary, the format of the codes is:
     * x18y18z18x17y17z17...x1y1z1x0y0z0d6...d0
     * 
     * This implementation is also different from octree.h in the fact that we use a vector of radii instead of a single radius. 
     * 
     * A reference implementation was followed: https://github.com/Excalibur-SLE/AdaptOctree/blob/master/adaptoctree/tree.py
     * 
     * Although many adaptations were done (conversion to C++, 7 depth bits instead of 15, etc.)
     * 
     */

    // The map storing the nodes by their morton codes
    // It contains both internal and leaf nodes, internal nodes don't have any points on them
    // TODO: there are probably better ways to do this
    std::unordered_map<morton_t, LinearOctreeNode*> nodes; 
    
    // Center of the point cloud at depth level 0
    Point center;

    // Vector of radii of the point cloud at depth level 0
    Vector radii; 
    
    /**
     * Some constants and bitmasks useful for working with Morton codes
     */
    static constexpr uint8_t DEPTH_BITS = 7;
    static constexpr uint8_t DEPTH_MASK = 0x7f;
    static constexpr morton_t NOT_DEPTH_MASK = 0xffffffffffffffff ^ DEPTH_MASK;
    static constexpr morton_t LAST_DEPTH_BITS_MASK = 0x0000000000000380;
    static constexpr morton_t NOT_LAST_DEPTH_BITS_MASK = 0xffffffffffffffff ^ LAST_DEPTH_BITS_MASK;
    static constexpr morton_t X_MASK = 0x9249249249249200;
    static constexpr morton_t Y_MASK = 0x4924924924924900;
    static constexpr morton_t Z_MASK = 0x2492492492492480;
    static constexpr morton_t XY_MASK = X_MASK | Y_MASK;
    static constexpr morton_t YZ_MASK = Y_MASK | Z_MASK;
    static constexpr morton_t XZ_MASK = X_MASK | Z_MASK;
    
    /*
     * Method to convert a point to its anchor, this operation approximates its coordinates and so it is not reversible
     */
    inline void getAnchorCoords(const Point& p, uint8_t depth, coords_t &x, coords_t &y, coords_t &z) {
        // TODO: it should be 2^depth and not have to add this case, investigate why it wasn't working
        if(depth == 0) {
            x = 0, y = 0, z = 0;
            return;
        }
        float x_transf = ((p.getX() - center.getX())  + radii.getX()) / (2 * radii.getX());
        float y_transf = ((p.getY() - center.getY())  + radii.getY()) / (2 * radii.getY());
        float z_transf = ((p.getZ() - center.getZ())  + radii.getZ()) / (2 * radii.getZ());

        // Get the integer coordinates by multiplying by 2^(depth-1) and then taking floor
        x = (coords_t) (x_transf * (1 << (depth-1)));
        y = (coords_t) (y_transf * (1 << (depth-1)));
        z = (coords_t) (z_transf * (1 << (depth-1)));
    }

    /**
     * Methods for encoding and decoding of points into morton codes
     */
    static inline morton_t encodeMorton(uint8_t depth, coords_t x, coords_t y, coords_t z) {
        // Compute the morton code and push point into corresponding bin
        morton_t code = libmorton::morton3D_64_encode(x, y, z);

        // Pack depth into the key by shifting and then putting key into the tail bits
        // In an octree, it is needed to distinguish nodes in different depths to allow traversals and so on
        return (code << DEPTH_BITS) | depth;
    }

    inline morton_t encodeMortonPoint(const Point& p, uint8_t depth) {
        // Utility method combining the two above
        coords_t x, y, z;
        getAnchorCoords(p, depth, x, y, z);
        return encodeMorton(depth, x, y, z);
    }

    static inline void decodeMorton(morton_t code, coords_t &x, coords_t &y, coords_t &z) {
        // First we unshift to remove the depth bits and get the original code we passed to libmorton
        code = code >> DEPTH_BITS;

        // Now we can recover the anchor coordinates
        libmorton::morton3D_64_decode(code, x, y, z);
    }

    /**
     * Utility methods for working with Morton codes
     * TODO: maybe remove the assertions here and do some better error handling
     */
    static inline uint8_t getDepth(morton_t code) {
        return (uint8_t) (code & DEPTH_MASK);
    }

    static inline morton_t getParentCode(morton_t code) {
        // To get parent morton code, shift 3 bits to the right and put level-1 in the trailing bits
        uint8_t depth = getDepth(code);
        assert(depth > 0);
        morton_t parent = (code >> 3) & NOT_DEPTH_MASK;
        return parent | (depth - 1);
    }

    static inline morton_t getSiblingCode(morton_t code, uint8_t index) {
        // To get a sibling morton code, just return the code with the last 3 bits before depth bits set to sibling index
        assert(index >= 0b000 && index <= 0b111);
        return (code & NOT_LAST_DEPTH_BITS_MASK) | (((morton_t) index) << DEPTH_BITS);
    }

    static inline morton_t getChildrenCode(morton_t code, uint8_t index) {
        // To get a child morton code, up the level, shift 3 bits to the right and then or the last 3 bits before
        // trailing to the sibling index
        uint8_t depth = getDepth(code);
        assert(depth <= MAX_DEPTH && index >= 0b000 && index <= 0b111);
        // Shift code one layer to the right, by masking first we make sure the 3 bits where we are going
        // to put the children are already empty
        morton_t children = (code & NOT_DEPTH_MASK) << 3;
        // Put children bits and new level
        return children | (((morton_t) index) << DEPTH_BITS) | (depth + 1);
    }
    
    static void printMortonCode(coords_t x, coords_t y, coords_t z, morton_t code, bool formatted = false) {
        std::cout << "Anchor center " << x << ", " << y << ", " << z << "\n";
        printMortonCode(code, formatted);
    }

    static void printMortonCode(Point &p, morton_t code, bool formatted = false) {
        std::cout << "Physical center " << p.getX() << ", " << p.getY() << ", " << p.getZ() << "\n";
        printMortonCode(code, formatted);
    }

    static void printMortonCode(morton_t code, bool formatted = false) {
        // Print the bits in groups of 3 to represent each level
        if(formatted) {
            for (int i = 63; i >= 7; i -= 3) {
                std::cout << std::bitset<3>((code >> (i - 2)) & 0b111) << " ";
            }
            // Print the last 7 bits together
            std::cout << "| " << std::bitset<7>(code & 0b1111111) << "\n";
        } else {
            std::cout << std::bitset<64>(code) << "\n";
        }
    }

    /**
     * Utility methods for getting geometric information (center, radius, inside check) from a morton code
     */
    inline Point getNodeCenter(morton_t code) {
        // Returns the physical (approximate) physical center of the node
        coords_t x, y, z;
        uint8_t depth = getDepth(code);
        decodeMorton(code, x, y, z);
        double x_d, y_d, z_d;
        Vector nodeRadii = getNodeRadii(code);
        Point lowCorner = center - radii;
        x_d = (x + 0.5) * nodeRadii.getX() + lowCorner.getX();
        y_d = (y + 0.5) * nodeRadii.getY() + lowCorner.getY();
        z_d = (z + 0.5) * nodeRadii.getZ() + lowCorner.getZ();
        return Point(x_d, y_d, z_d);
    }

    inline Vector getNodeRadii(morton_t code) {
        // Returns the vector of (approximate) physical radii of the node
        uint8_t depth = getDepth(code);
        return Vector(radii.getX() * (1.0f / (1 << depth)), 
                        radii.getY()* (1.0f / (1 << depth)), 
                        radii.getZ()* (1.0f / (1 << depth)));
    }


    void printNodeGeometry(morton_t code) {
        Box bbox = Box(getNodeCenter(code), getNodeRadii(code));
        std::cout << "Node: ";
        printMortonCode(code, true);
        std::cout << "Center: " << bbox.center() << "\n";
        std::cout << "Radii: " << bbox.radii() << "\n";
        std::cout << "Lower corner: " << bbox.min() << "\nUpper corner: " << bbox.max() << "\n"; 
    }


    inline bool isInside(Point &p, morton_t code) {
        // To check if a node is inside a given code, we compute its morton code at the depth of the node
        // and check whether it is the same
        // The "physical" approach of getting the node center and radii and computing the box would not be
        // accurate since we are only approximating those
        return isNode(code) && (encodeMortonPoint(p, getDepth(code)) == code);
    } 



    // Insert points into the octree by computing their bins, and adds nodes to keep processing to the queue
    void insertPoints(std::vector<Lpoint*>& points, uint8_t depth, std::stack<LinearOctreeNode*>& subdivision_stack) {
        std::unordered_map<morton_t, std::vector<Lpoint*>> bins;
        for(int i = 0; i<points.size(); i++) {
            // Shift and scale coordinates into [0, 1]^3
            // x' = ((x - c_x) + r_x) / (2*r_x)
            morton_t code = encodeMortonPoint(*points[i], depth);
            bins[code].push_back(points[i]);
        } 
        
        // Add good nodes to the octree, reject and put into subdivision stack the others
        for (auto& [code, binPoints] : bins) {
            auto node = new LinearOctreeNode(binPoints, code, depth);
            // Insert into octree map (we also keep internal nodes, 
            // though their vectors of points will be cleared in case we subdivide)
            nodes[code] = node;

            // TODO: also add min physical size condition
            if(binPoints.size() > MAX_POINTS && depth < MAX_DEPTH) {
                // Push into queue for future subdivision until we have small amount of points
                subdivision_stack.push(node);
            }
        }
    }

public:
    LinearOctree() = default;

    explicit LinearOctree(std::vector<Lpoint>& points) {
        center = mbb(points, radii);
        std::vector<Lpoint*> points_p;
        points_p.reserve(points.size());
        for (auto& point : points) {
            points_p.push_back(&point);
        }
        buildOctree(points_p);
    }

    explicit LinearOctree(std::vector<Lpoint*>& points) {
        buildOctree(points);
    }

    // TODO: getters of center, radii, bbox, etc.

    [[nodiscard]] inline double getDensity() const
	/*
    * @brief Computes the point density of the given Octree as nPoints / Volume
    */
	{
		// TODO
        return 0.0f;
	}

    void writeDensities(const std::filesystem::path& path) const;
	void writeNumPoints(const std::filesystem::path& path) const;

    // Called this findNode instead of findOctant but they do the same
    [[nodiscard]] const LinearOctreeNode* findNode(const Lpoint* p) const;

    // TODO implement at cpp the ones that make sense for linear octrees
    [[nodiscard]] std::vector<std::pair<Point, double>> computeDensities() const;
	[[nodiscard]] std::vector<std::pair<Point, size_t>> computeNumPoints() const;
    [[nodiscard]] bool isInside2D(const Point& p) const;
    void   insertPoints(std::vector<Lpoint>& points);
	void   insertPoints(std::vector<Lpoint*>& points);
	void   insertPoint(Lpoint* p);
	void   createOctants();
	void   fillOctants();
	size_t octantIdx(const Lpoint* p) const;

    /**
     * Methods for checking if the node is in the tree, is a leaf or is an inner node
     * 
     * "isEmpty()" doesnt make much sense in a Linear Octree
     */
    [[nodiscard]] inline bool isNode(morton_t code) const {
        return nodes.find(code) != nodes.end();
    }
    [[nodiscard]] inline bool isLeaf(morton_t code) const { 
        auto it = nodes.find(code);
        if(it == nodes.end()) {
            std::cout << code << " not found" << std::endl;
            return false;
        }
        return it->second->isLeaf();
    }
    [[nodiscard]] inline bool isInner(morton_t code) const {
        auto it = nodes.find(code);
        if(it == nodes.end()) {
            return false;
        }
        return !(it->second->isLeaf());
    }

    void buildOctree(std::vector<Lpoint*>& points) {
        // TODO: usually build process its different, points are sorted globally and then put into bins
        // Maybe doing something similar but simpler than cornerstone paper https://dl.acm.org/doi/abs/10.1145/3592979.3593417
        // this is faster and more parallelizable in the future
        
        // Add all points to the root node and add to subdivision stack
        std::stack<LinearOctreeNode*> subdivision_stack;
        auto node = new LinearOctreeNode(points, 0, 0);
        nodes[0] = node;

        if(points.size() > MAX_POINTS){
            subdivision_stack.push(node);
        }

        // Process nodes that we still need to subdivide
        while(!subdivision_stack.empty()) {
            node = subdivision_stack.top();
            subdivision_stack.pop();
            // Reprocess the points in the node
            insertPoints(node->points, LinearOctree::getDepth(node->code) + 1, subdivision_stack);

            // Clear the old node points array
            node->points.clear();
        }
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

	template<Kernel_t kernel_type = Kernel_t::cube>
	[[nodiscard]] inline std::vector<Lpoint*> searchNeighbors(const Point& p, const Vector& radii) const
	/**
   * @brief Search neighbors function. Given a point and a radius, return the points inside a given kernel type
   * @param p Center of the kernel to be used
   * @param radii Radii of the kernel to be used
   * @return Points inside the given kernel type
   */
	{
		const auto kernel = kernelFactory<kernel_type>(p, radii);
		// Dummy condition that always returns true, so we can use the same function for all cases
		// The compiler should optimize this away
		constexpr auto dummyCondition = [](const Lpoint&) { return true; };

		return neighbors(kernel, dummyCondition);
	}

	template<Kernel_t kernel_type = Kernel_t::square, class Function>
	[[nodiscard]] inline std::vector<Lpoint*> searchNeighbors(const Point& p, double radius, Function&& condition) const
	/**
   * @brief Search neighbors function. Given a point and a radius, return the points inside a given kernel type
   * @param p Center of the kernel to be used
   * @param radius Radius of the kernel to be used
   * @param condition function that takes a candidate neighbor point and imposes an additional condition (should return a boolean).
   * The signature of the function should be equivalent to `bool cnd(const Lpoint &p);`
   * @return Points inside the given kernel type
   */
	{
		const auto kernel = kernelFactory<kernel_type>(p, radius);
		return neighbors(kernel, std::forward<Function&&>(condition));
	}

	template<Kernel_t kernel_type = Kernel_t::square, class Function>
	[[nodiscard]] inline std::vector<Lpoint*> searchNeighbors(const Point& p, const Vector& radii,
	                                                          Function&& condition) const
	/**
   * @brief Search neighbors function. Given a point and a radius, return the points inside a given kernel type
   * @param p Center of the kernel to be used
   * @param radii Radii of the kernel to be used
   * @param condition function that takes a candidate neighbor point and imposes an additional condition (should return a boolean).
   * The signature of the function should be equivalent to `bool cnd(const Lpoint &p);`
   * @return Points inside the given kernel type
   */
	{
		const auto kernel = kernelFactory<kernel_type>(p, radii);
		return neighbors(kernel, std::forward<Function&&>(condition));
	}

	[[nodiscard]] std::vector<Lpoint*> KNN(const Point& p, size_t k, size_t maxNeighs = DEFAULT_KNN) const;

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
		std::stack<morton_t> toVisit;

        if(!isNode(root)) // Checks if the root is an actual node in the tree
            return ptsInside;
		toVisit.push(root); // Root of the tree

		while (!toVisit.empty()) {
            const morton_t code = toVisit.top();
			toVisit.pop();

			if (isLeaf(code)) {
                auto node = nodes.find(code)->second;
				for (Lpoint* point_ptr : node->points) {
                    // Check the point
					if (k.isInside(*point_ptr) && k.center().id() != point_ptr->id() && condition(*point_ptr))
						ptsInside.emplace_back(point_ptr); // add the point to the result list
				}
			} else {
                // If we are in an inner node, add all the child octants to the search list
                for(int index = 0; index<8; index++) {
                    morton_t childCode = getChildrenCode(code, index);
                    if(isNode(childCode))
                        toVisit.push(childCode);
                }
			}
		}
		return ptsInside;
	}

    // Search neighbors overloads
    [[nodiscard]] inline std::vector<Lpoint*> searchNeighbors2D(const Point& p, const double radius) const
	{
		return searchNeighbors<Kernel_t::square>(p, radius);
	}

	[[nodiscard]] inline std::vector<Lpoint*> searchCylinderNeighbors(const Lpoint& p, const double radius,
	                                                                  const double zMin, const double zMax) const
	{
		return searchNeighbors<Kernel_t::circle>(p, radius,
		                                         [&](const Lpoint& p) { return p.getZ() >= zMin && p.getZ() <= zMax; });
	}

	[[nodiscard]] inline std::vector<Lpoint*> searchCircleNeighbors(const Lpoint& p, const double radius) const
	{
		return searchNeighbors<Kernel_t::circle>(p, radius);
	}

	[[nodiscard]] inline std::vector<Lpoint*> searchCircleNeighbors(const Lpoint* p, const double radius) const
	{
		return searchCircleNeighbors(*p, radius);
	}

	[[nodiscard]] inline std::vector<Lpoint*> searchNeighbors3D(const Point& p, const Vector& radii) const
	{
		return searchNeighbors<Kernel_t::cube>(p, radii);
	}

	[[nodiscard]] inline std::vector<Lpoint*> searchNeighbors3D(const Point& p, double radius) const
	{
		return searchNeighbors<Kernel_t::cube>(p, radius);
	}

	[[nodiscard]] inline std::vector<Lpoint*> searchNeighbors3D(const Point& p, const double radius,
	                                                            const std::vector<bool>& flags) const
	/**
     * Searching neighbors in 3D using a different radius for each direction
     * @param p Point around the neighbors will be search
     * @param radius Vector of radiuses: one per spatial direction
     * @param flags Vector of flags: return only points which flags[pointId] == false
     * @return Points inside the given kernel
     */
	{
		const auto condition = [&](const Point& point) { return !flags[point.id()]; };
		return searchNeighbors<Kernel_t::cube>(p, radius, condition);
	}

    // Simpler functions for counting number of neighbors
	template<Kernel_t kernel_type = Kernel_t::square>
	[[nodiscard]] inline size_t numNeighbors(const Point& p, const double radius) const
	/**
   * @brief Search neighbors function. Given a point and a radius, return the number of points inside a given kernel type
   * @param p Center of the kernel to be used
   * @param radius Radius of the kernel to be used
   * @param condition function that takes a candidate neighbor point and imposes an additional condition (should return a boolean).
   * The signature of the function should be equivalent to `bool cnd(const Lpoint &p);`
   * @return Points inside the given kernel
   */
	{
		const auto kernel = kernelFactory<kernel_type>(p, radius);
		return numNeighbors(kernel);
	}
    
    template<typename Kernel, typename Function>
	[[nodiscard]] size_t numNeighbors(const Kernel& k, morton_t root = 0) const
	{
		size_t ptsInside = 0;
		std::stack<morton_t> toVisit;

        if(!isNode(root)) // Checks if the root is an actual node in the tree
            return ptsInside;
		toVisit.push(root); // Root of the tree

		while (!toVisit.empty()) {
            const morton_t code = toVisit.top();
			toVisit.pop();

			if (isLeaf(code)) {
                auto node = nodes.find(code)->second;
				for (Lpoint* point_ptr : node->points) {
                    // Check the point
					if (k.isInside(*point_ptr) && k.center().id() != point_ptr->id())
						ptsInside++;
				}
			} else {
                // If we are in an inner node, add all the child octants to the search list
                for(int index = 0; index<8; index++) {
                    morton_t childCode = getChildrenCode(code, index);
                    if(isNode(childCode))
                        toVisit.push(childCode);
                }
			}
		}
		return ptsInside;
	}

    template<typename Kernel, typename Function>
	[[nodiscard]] size_t numNeighbors(const Kernel& k, Function&& condition, morton_t root = 0) const
	{
		size_t ptsInside = 0;
		std::stack<morton_t> toVisit;

        if(!isNode(root)) // Checks if the root is an actual node in the tree
            return ptsInside;
		toVisit.push(root); // Root of the tree

		while (!toVisit.empty()) {
            const morton_t code = toVisit.top();
			toVisit.pop();

			if (isLeaf(code)) {
                auto node = nodes.find(code)->second;
				for (Lpoint* point_ptr : node->points) {
                    // Check the point
					if (k.isInside(*point_ptr) && k.center().id() != point_ptr->id() && condition(*point_ptr))
						ptsInside++;
				}
			} else {
                // If we are in an inner node, add all the child octants to the search list
                for(int index = 0; index<8; index++) {
                    morton_t childCode = getChildrenCode(code, index);
                    if(isNode(childCode))
                        toVisit.push(childCode);
                }
			}
		}
		return ptsInside;
	}

    // Other neighbourhood search methods
    [[nodiscard]] inline std::vector<Lpoint*> searchSphereNeighbors(const Point& point, const float radius) const
	{
		return searchNeighbors<Kernel_t::sphere>(point, radius);
	}

	[[nodiscard]] std::vector<Lpoint*> searchNeighborsRing(const Lpoint& p, const Vector& innerRingRadii,
	                                                       const Vector& outerRingRadii) const
	/**
	 * A point is considered to be inside a Ring around a point if its outside the innerRing and inside the outerRing
	 * @param p Center of the kernel to be used
	 * @param innerRingRadii Radii of the inner part of the ring. Points within this part will be excluded
	 * @param outerRingRadii Radii of the outer part of the ring
	 * @return The points located between the inner ring and the outer ring
	 */
	{
		// Search points within "outerRingRadii"
		const auto outerKernel = kernelFactory<Kernel_t::cube>(p, outerRingRadii);
		// But not too close (within "innerRingRadii")
		const auto innerKernel = kernelFactory<Kernel_t::cube>(p, innerRingRadii);
		const auto condition   = [&](const Point& point) { return !innerKernel.isInside(point); };

		return neighbors(outerKernel, condition);
	}

	void writeOctree(std::ofstream& f, size_t index) const;

	void    extractPoint(const Lpoint* p);
	Lpoint* extractPoint();
	void    extractPoints(std::vector<Lpoint>& points);
	void    extractPoints(std::vector<Lpoint*>& points);

	std::vector<Lpoint*> searchEraseCircleNeighbors(const std::vector<Lpoint*>& points, double radius);

	/** Inside a sphere */
	std::vector<Lpoint*> searchEraseSphereNeighbors(const std::vector<Lpoint*>& points, float radius);

	/** Connected inside a spherical shell*/
	[[nodiscard]] std::vector<Lpoint*> searchConnectedShellNeighbors(const Point& point, float nextDoorDistance,
	                                                                 float minRadius, float maxRadius) const;

	/** Connected circle neighbors*/
	std::vector<Lpoint*> searchEraseConnectedCircleNeighbors(float nextDoorDistance);

	static std::vector<Lpoint*> connectedNeighbors(const Point* point, std::vector<Lpoint*>& neighbors,
	                                               float nextDoorDistance);

	static std::vector<Lpoint*> extractCloseNeighbors(const Point* p, std::vector<Lpoint*>& neighbors, float radius);

	std::vector<Lpoint*> kClosestCircleNeighbors(const Lpoint* p, size_t k) const;
	std::vector<Lpoint*> nCircleNeighbors(const Lpoint* p, size_t n, float& radius, float minRadius, float maxRadius,
	                                      float maxIncrement = 0.25, float maxDecrement = 0.25) const;

	std::vector<Lpoint*> nSphereNeighbors(const Lpoint& p, size_t n, float& radius, float minRadius, float maxRadius,
	                                      float maxStep = 0.25) const;

    /**
     * Debug function for testing linear octree functionality
     */
    void testOctree(std::vector<Lpoint>& points) {
        // Check all points were inserted correctly
        int total = 0;
        for (auto& [code, node] : nodes) {
            total += node->points.size();
        }
        std::cout << "Total inserted " << total << " out of " << points.size() << " points\n";

/*         // Little test for seeing morton codes 
        for(int i = 0; i<2; i++){
            for(int j = 0; j<2; j++) {
                for(int k = 0; k<2; k++) {
                    morton_t code = (libmorton::morton3D_64_encode(i, j, k) << 7) | 1;
                    std::cout << "(" << i << ", " << j << ", " << k << ") code: " 
                        << std::bitset<64>(code) << "\n";
                }
            }
        }

        // Get center code at depth 3
        coords_t x, y, z;
        getAnchorCoords(center, 3, x, y, z);
        std::cout << "anchor coods: " << x << " " << y << " " << z << "\n";
        morton_t code = encodeMortonPoint(center, 3);
        printMortonCode(center, code, true);
        
        // Get their children
        for(int index = 0; index<8; index++) {
            morton_t childCode = getChildrenCode(code, index);
            std::cout << "child #" << index << "\n";
            printMortonCode(childCode, true);
            Point coords = getNodeCenter(childCode);
            std::cout << "center: " << coords.getX() << " " << coords.getY() << " " << coords.getZ() << " " << std::endl;
        }

        // Get their siblings
        for(int index = 0; index<8; index++) {
            morton_t siblingCode = getSiblingCode(code, index);
            std::cout << "sibling #" << index << "\n";
            printMortonCode(siblingCode, true);
            Point coords = getNodeCenter(siblingCode);
            std::cout << "center: " << coords.getX() << " " << coords.getY() << " " << coords.getZ() << " " << std::endl;
        }

        // Get its parent
        morton_t parentCode = getParentCode(code);
        std::cout << "parent ";
        printMortonCode(parentCode, true);
        Point coords = getNodeCenter(parentCode);
        std::cout << "center: " << coords.getX() << " " << coords.getY() << " " << coords.getZ() << " " << std::endl; */

        // Search for all points in the octree
        TimeWatcher tw;
        int not_found = 0;
        std::cout << "Searching all the points in the octree... " << std::endl;
        tw.start();
        for(int i = 0; i<points.size(); i++) {
            Point p = points[i];
            morton_t code = 0;
            bool not_found_flag;
            while(!isLeaf(code) && getDepth(code) <= MAX_DEPTH) {
                not_found_flag = true;
                for(uint8_t index = 0; index < 8; index++) {
                    morton_t childCode = getChildrenCode(code, index);
                    // If the node point is inside, go into the leaf
                    if(isInside(p, childCode)) {
                        code = childCode;
                        not_found_flag = false;
                    }
                }
                if(not_found_flag)
                    break;
            }
            
            if(not_found_flag || !isInside(p, code)) {    
                not_found++;
                //std::cout << "Could not find the point " << p << std::endl;
                //printNodeGeometry(code);
            } 
        }
        tw.stop();
        std::cout   << "Found " << (points.size() - not_found) << "/" << points.size() 
                    << " points in the octree in " << tw.getElapsedDecimalSeconds() << " seconds (" 
                    << ((float) tw.getElapsedMicros() / (float) points.size()) << " microseconds per point)" << std::endl; 
    }
};