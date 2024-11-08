#include "benchmarking.hpp"
#include "octree.hpp"
#include "octree_linear.hpp"
#include <random>

#pragma once 

class OctreeBenchmark {
    private:
        constexpr static size_t search_size = 1;
        constexpr static float min_radius = 0.01;
        constexpr static float max_radius = 100.0;

        std::vector<Lpoint> points;
        Octree* pOctree = nullptr;
        LinearOctree* lOctree = nullptr;
        std::mt19937 rng;
        
        void octreePointerBuild() {
            // Pointer based octree
            if(pOctree != nullptr)
                delete pOctree;
            pOctree = new Octree(points);
        }
        void octreeLinearBuild() {
            // Pointer based octree
            if(lOctree != nullptr)
                delete lOctree;
            lOctree = new LinearOctree(points);
        }

        void octreePointerSearchNeighSphere() {
            std::uniform_int_distribution<size_t> dist_indexes(0, search_size- 1);
            std::uniform_real_distribution<float> dist_radius(min_radius, max_radius);

            for(int i = 0; i<search_size; i++) {
                auto neigh_points = pOctree->searchSphereNeighbors(points[dist_indexes(rng)], dist_radius(rng));
            }
        }

        void octreeLinearSearchNeighSphere() {
            std::uniform_int_distribution<size_t> dist_indexes(0, search_size- 1);
            std::uniform_real_distribution<float> dist_radius(min_radius, max_radius);

            for(int i = 0; i<search_size; i++) {
                auto neigh_points = lOctree->searchSphereNeighbors(points[dist_indexes(rng)], dist_radius(rng));
            }
        }
        
    public:
        OctreeBenchmark(std::vector<Lpoint> points) : points(points) {
            rng.seed(0);
            octreePointerBuild();
            octreeLinearBuild();
        }

        void benchmarkbuild(size_t repeats) {
            benchmarking::benchmark("Pointer octree build", repeats, [this]() { octreePointerBuild(); });
            benchmarking::benchmark("Linear octree build", repeats, [this]() { octreeLinearBuild(); });
        }   

        void benchmarkSearchNeighSphere(size_t repeats) {
            benchmarking::benchmark("Pointer octree neighbor search with spheres", repeats, [this]() { octreePointerSearchNeighSphere(); });
            benchmarking::benchmark("Linear octree neighbor search with spheres", repeats, [this]() { octreeLinearSearchNeighSphere(); });
        }


};
