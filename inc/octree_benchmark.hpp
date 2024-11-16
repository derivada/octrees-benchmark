#include "benchmarking.hpp"
#include "octree.hpp"
#include "octree_linear.hpp"
#include "octree_linear_old.hpp"
#include "morton_encoder.hpp"
#include <random>
#include "point.hpp"
#include <omp.h>

#pragma once 

class OctreeBenchmark {
    private:
        constexpr static size_t search_size = 10000;
        constexpr static float min_radius = 0.01;
        constexpr static float max_radius = 100.0;
        
        // Copy points for the linear octree because it reorders them and also to be more fair on comparisons
        // i.e. neither tree has points already in-cache when executing after the other
        std::vector<Lpoint> &points, &lOctreePoints;
        Octree* pOctree = nullptr;
        LinearOctreeOld* oldLOctree = nullptr;
        LinearOctree *lOctree = nullptr;
        std::mt19937 rng;
        
        size_t searchPointIndexes[search_size];
        float searchRadii[search_size];
        std::vector<Lpoint*> searchResultsPointer[search_size];
        std::vector<Lpoint*> searchResultsOldLinear[search_size];
        std::vector<Lpoint*> searchResultsLinear[search_size];

        void generateSearchSet() {
            rng.seed(0);
            std::uniform_int_distribution<size_t> indexDist(0, points.size()-1);
            std::uniform_real_distribution<float> radiusDist(min_radius, max_radius);
#pragma omp parallel for
            for(int i = 0; i<search_size; i++) {
                searchPointIndexes[i] = indexDist(rng);
                searchRadii[i] = radiusDist(rng);
                searchResultsPointer[i].clear();
                searchResultsOldLinear[i].clear();
                searchResultsLinear[i].clear();
            }
        }

        bool checkSearchResults() {
            bool correct = true;
            for(int i = 0; i<search_size; i++) {
                auto point = searchResultsPointer[i];
                auto lin = searchResultsLinear[i];
                auto oldLin = searchResultsOldLinear[i];
                /**
                if(point.size() != oldLin.size()) {
                    std::cout << "Wrong search result size in set " << i << 
                    "\n Pointer = " << point.size() << " Old linear = " << oldLin.size() << std::endl;
                    correct = false;
                } else {
                    sort(point.begin(), point.end(), [&] (Lpoint *p, Lpoint* q) -> bool {
                        return p->id() < q->id();
                    });
                    sort(oldLin.begin(), oldLin.end(), [&] (Lpoint *p, Lpoint* q) -> bool {
                        return p->id() < q->id();
                    });
                    for(int j = 0; j<point.size(); j++){
                        if(point[j]->id() != oldLin[j]->id()) {
                            std::cout << "Wrong search result point in set " << i << " at index " << j <<
                            "\n Pointer = " << point[j]->id() << " Old linear = " << oldLin[j]->id() << std::endl;
                            correct = false;
                        }
                    }
                } */

                if(point.size() != lin.size()) {
                    std::cout << "Wrong search result size in set " << i << 
                    "\n Pointer = " << point.size() << " Linear = " << lin.size() << std::endl << 
                    "Search center: " << points[searchPointIndexes[i]] << " with radii " << searchRadii[i] << std::endl;
                    correct = false;
                } else {
                    sort(point.begin(), point.end(), [&] (Lpoint *p, Lpoint* q) -> bool {
                        return p->id() < q->id();
                    });
                    sort(lin.begin(), lin.end(), [&] (Lpoint *p, Lpoint* q) -> bool {
                        return p->id() < q->id();
                    });
                    for(int j = 0; j<point.size(); j++){
                        if(point[j]->id() != lin[j]->id()) {
                            std::cout << "Wrong search result point in set " << i << " at index " << j <<
                            "\n Pointer = " << point[j]->id() << " Linear =  " << lin[j]->id() << std::endl;
                            correct = false;
                        }
                    }
                }
            }
            return correct;
        }

        void octreePointerBuild() {
            // Pointer based octree
            if(pOctree != nullptr)
                delete pOctree;
            pOctree = new Octree(points);
        }
        void octreeOldLinearBuild() {
            // Old linear octree impl.
            if(oldLOctree != nullptr)
                delete oldLOctree;
            oldLOctree = new LinearOctreeOld(points);
        }
        void octreeLinearBuild() {
            // Linear octree impl.
            if(lOctree != nullptr)
                delete lOctree;
            lOctree = new LinearOctree(lOctreePoints);
        }

        void octreePointerSearchNeighSphere() {
#pragma omp parallel for schedule(static)
            for(int i = 0; i<search_size; i++) {
                searchResultsPointer[i] = pOctree->searchSphereNeighbors(points[searchPointIndexes[i]], searchRadii[i]);
            }
        }

        void octreeOldLinearSearchNeighSphere() {
#pragma omp parallel for schedule(static)
            for(int i = 0; i<search_size; i++) {
                searchResultsOldLinear[i] = oldLOctree->searchSphereNeighbors(points[searchPointIndexes[i]], searchRadii[i]);
            }
        }
        void octreeLinearSearchNeighSphere() {
#pragma omp parallel for schedule(static)
            for(int i = 0; i<search_size; i++) {
                searchResultsLinear[i] = lOctree->searchSphereNeighbors(points[searchPointIndexes[i]], searchRadii[i]);
            }
        }
    public:
        OctreeBenchmark(std::vector<Lpoint> &points, std::vector<Lpoint> &lsOctreePoints) : points(points), lOctreePoints(lsOctreePoints) {
            rng.seed(0);
            octreePointerBuild();
            // octreeOldLinearBuild();
            octreeLinearBuild();
        }

        void benchmarkbuild(size_t repeats) {
            benchmarking::benchmark("Pointer octree build", repeats, [this]() { octreePointerBuild(); });
            // benchmarking::benchmark("Old linear octree build", repeats, [this]() { octreeOldLinearBuild(); });
            benchmarking::benchmark("Linear octree build", repeats, [this]() { octreeLinearBuild(); });
        }   

        void benchmarkSearchNeighSphere(size_t repeats, bool checkResults = false) {
            generateSearchSet();
            std::cout << "Generated search set containing " << search_size << " operations" << std::endl;
            benchmarking::benchmark("Pointer octree neighbor search with spheres", repeats, [this]() { octreePointerSearchNeighSphere(); });
            benchmarking::benchmark("Linear octree neighbor search with spheres", repeats, [this]() { octreeLinearSearchNeighSphere(); });
            if(checkResults && checkSearchResults())
                std::cout << "All search methods yield the same results!" << std::endl;
            
        }
};
