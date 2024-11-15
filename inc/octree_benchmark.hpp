#include "benchmarking.hpp"
#include "octree.hpp"
#include "octree_linear.hpp"
#include "octree_linear_sort.hpp"
#include "morton_encoder.hpp"
#include <random>
#include "point.hpp"
#include <omp.h>

#pragma once 

class OctreeBenchmark {
    private:
        constexpr static size_t search_size = 10000;
        constexpr static float min_radius = 0.01;
        constexpr static float max_radius = 10.0;
        
        // Copy points for the linear octree because it reorders them
        std::vector<Lpoint> &points, &lsOctreePoints;
        Octree* pOctree = nullptr;
        LinearOctree* lOctree = nullptr;
        LinearOctreeSort *lsOctree = nullptr;
        std::mt19937 rng;
        
        size_t searchPointIndexes[search_size];
        float searchRadii[search_size];
        std::vector<Lpoint*> searchResultsPointer[search_size];
        std::vector<Lpoint*> searchResultsLinear[search_size];
        std::vector<Lpoint*> searchResultsLinearSort[search_size];

        void generateSearchSet() {
            rng.seed(0);
            std::uniform_int_distribution<size_t> indexDist(0, points.size()-1);
            std::uniform_real_distribution<float> radiusDist(min_radius, max_radius);
#pragma omp parallel for
            for(int i = 0; i<search_size; i++) {
                searchPointIndexes[i] = indexDist(rng);
                searchRadii[i] = radiusDist(rng);
                searchResultsPointer[i].clear();
                searchResultsLinear[i].clear();
                searchResultsLinearSort[i].clear();
            }
        }

        bool checkSearchResults() {
            bool correct = true;
            for(int i = 0; i<search_size; i++) {
                auto point = searchResultsPointer[i];
                auto linSort = searchResultsLinearSort[i];
                auto lin = searchResultsLinear[i];
                if(point.size() != lin.size()) {
                    std::cout << "Wrong search result size in set " << i << 
                    "\n Pointer = " << point.size() << " Linear: " << lin.size() << std::endl;
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
                            "\n Pointer = " << point[j]->id() << " Linear: " << lin[j]->id() << std::endl;
                            correct = false;
                        }
                    }
                }
                if(point.size() != linSort.size()) {
                    std::cout << "Wrong search result size in set " << i << 
                    "\n Pointer = " << point.size() << " Linear sort: " << linSort.size() << std::endl << 
                    "Search center: " << points[searchPointIndexes[i]] << " with radii " << searchRadii[i] << std::endl;
                    correct = false;
                } else {
                    sort(point.begin(), point.end(), [&] (Lpoint *p, Lpoint* q) -> bool {
                        return p->id() < q->id();
                    });
                    sort(linSort.begin(), linSort.end(), [&] (Lpoint *p, Lpoint* q) -> bool {
                        return p->id() < q->id();
                    });
                    for(int j = 0; j<point.size(); j++){
                        if(point[j]->id() != linSort[j]->id()) {
                            std::cout << "Wrong search result point in set " << i << " at index " << j <<
                            "\n Pointer = " << point[j]->id() << " Linear sort: " << linSort[j]->id() << std::endl;
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
        void octreeLinearBuild() {
            // Pointer based octree
            if(lOctree != nullptr)
                delete lOctree;
            lOctree = new LinearOctree(points);
        }
        void octreeLinearSortBuild() {
            // Pointer based octree
            if(lsOctree != nullptr)
                delete lsOctree;
            lsOctree = new LinearOctreeSort(lsOctreePoints);
        }

        void octreePointerSearchNeighSphere() {
#pragma omp parallel for schedule(static)
            for(int i = 0; i<search_size; i++) {
                searchResultsPointer[i] = pOctree->searchSphereNeighbors(points[searchPointIndexes[i]], searchRadii[i]);
            }
        }

        void octreeLinearSearchNeighSphere() {
#pragma omp parallel for schedule(static)
            for(int i = 0; i<search_size; i++) {
                searchResultsLinear[i] = lOctree->searchSphereNeighbors(points[searchPointIndexes[i]], searchRadii[i]);
            }
        }
        void octreeLinearSortSearchNeighSphere() {
#pragma omp parallel for schedule(static)
            for(int i = 0; i<search_size; i++) {
                searchResultsLinearSort[i] = lsOctree->searchSphereNeighbors(points[searchPointIndexes[i]], searchRadii[i]);
            }
        }
    public:
        OctreeBenchmark(std::vector<Lpoint> &points, std::vector<Lpoint> &lsOctreePoints) : points(points), lsOctreePoints(lsOctreePoints) {
            rng.seed(0);
            octreePointerBuild();
            octreeLinearBuild();
            octreeLinearSortBuild();
        }

        void benchmarkbuild(size_t repeats) {
            benchmarking::benchmark("Pointer octree build", repeats, [this]() { octreePointerBuild(); });
            benchmarking::benchmark("Linear octree build", repeats, [this]() { octreeLinearBuild(); });
            benchmarking::benchmark("Linear sort octree build", repeats, [this]() { octreeLinearSortBuild(); });
        }   

        void benchmarkSearchNeighSphere(size_t repeats, bool checkResults = false) {
            generateSearchSet();
            std::cout << "Generated search set containing " << search_size << " operations" << std::endl;
            benchmarking::benchmark("Pointer octree neighbor search with spheres", repeats, [this]() { octreePointerSearchNeighSphere(); });
            benchmarking::benchmark("Linear octree neighbor search with spheres", repeats, [this]() { octreeLinearSearchNeighSphere(); });
            benchmarking::benchmark("Linear sort octree neighbor search with spheres", repeats, [this]() { octreeLinearSortSearchNeighSphere(); });
            if(checkResults && checkSearchResults())
                std::cout << "All search methods yield the same results!" << std::endl;
            
        }
};
