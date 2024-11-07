/**
 * A linear (map-based) implementation of the Octree using Morton codes for quick access with good spacial locality
 * 
 * Pablo Díaz Viñambres 22/10/24
 * 
 */


#pragma once

#include "Lpoint.hpp"
#include "octree_linear_node.hpp"
#include "Box.hpp"
#include <stack>
#include <bitset>
#include <unordered_map>
#include "libmorton/morton.h"

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
        assert(depth < MAX_DEPTH && index >= 0b000 && index <= 0b111);
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
     * Method for checking whether node is a leaf or an inner node
     * 
     * (We have both methods because of the case where code is not on the map, both return false)
     */
    [[nodiscard]] bool isLeaf(morton_t code) const { 
        auto it = nodes.find(code);
        if(it == nodes.end()) {
            std::cout << code << " not found" << std::endl;
            return false;
        }
        return it->second->isLeaf();
    }

    [[nodiscard]] bool isInner(morton_t code) const {
        auto it = nodes.find(code);
        if(it == nodes.end()) {
            return false;
        }
        return !(it->second->isLeaf());
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

    inline bool isNode(morton_t code) const {
        return nodes.find(code) != nodes.end();
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
    LinearOctree();

    explicit LinearOctree(std::vector<Lpoint>& points) {
        center = mbb(points, radii);
        std::vector<Lpoint*> points_p;
        points_p.reserve(points.size());
        for (auto& point : points) {
            points_p.push_back(&point);
        }
        buildOctree(points_p);
    }

	explicit LinearOctree(std::vector<Lpoint*>& points);

	LinearOctree(const Point& center, float radius);
	LinearOctree(Point center, float radius, std::vector<Lpoint*>& points);
	LinearOctree(Point center, float radius, std::vector<Lpoint>& points);

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