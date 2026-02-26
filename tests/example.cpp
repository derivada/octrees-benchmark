#include <iostream>
#include <vector>
#include <optional>
#include <fstream>

#include "geometry/point.hpp"
#include "geometry/point_containers.hpp"
#include "geometry/point_metadata.hpp"
#include "encoding/morton_encoder_3d.hpp"
#include "structures/linear_octree.hpp"

int main() {
    std::cout << "--- Octrees Benchmark Library Test ---" << std::endl;
    
    // Create some points using Array of Structs container
    PointsAoS points;
    points.push_back(Point(1.2, 3.4, 5.6));
    points.push_back(Point(7.8, 9.0, 1.2));
    points.push_back(Point(3.4, 5.6, 7.8));
    points.push_back(Point(9.0, 1.2, 3.4));
    
    std::cout << "Created " << points.size() << " points." << std::endl;
    
    // Setup metadata
    std::optional<std::vector<PointMetadata>> metadata = std::nullopt;
    
    // Get encoder (Morton 3D)
    PointEncoding::MortonEncoder3D enc;
    
    // Sort points
    std::cout << "Sorting points..." << std::endl;
    auto [codes, box] = enc.sortPoints(points, metadata);
    std::cout << "Points sorted and encoded." << std::endl;
    
    // Build an octree
    std::cout << "Building LinearOctree..." << std::endl;
    LinearOctree octree(points, codes, box, enc);
    std::cout << "Octree built successfully!" << std::endl;
    
    // Perform a basic kNN search
    size_t k = 2;
    std::vector<size_t> indexes(k);
    std::vector<double> distances(k);
    
    Point searchPoint(5.0, 5.0, 5.0);
    std::cout << "Performing " << k << "-NN search for point (5.0, 5.0, 5.0)..." << std::endl;
    octree.knn(searchPoint, k, indexes, distances);
    
    for (size_t i = 0; i < k; ++i) {
        std::cout << "  Neighbor " << i + 1 << " - Index: " << indexes[i] 
                  << " Distance: " << distances[i] << std::endl;
    }
    
    std::cout << "Test completed successfully." << std::endl;
    return 0;
}
