#pragma once
#include <vector>
#include <iterator>

template <typename Point_t>
class NeighSearchResult {
    public:
        std::vector<Point_t> emptyPoints;
        std::vector<Point_t>& points;
        std::vector<std::pair<size_t, size_t>> ranges;
        size_t numberOfPoints = 0;

        // Empty constructor
        NeighSearchResult() : points(emptyPoints) {}

        // Regular constructor
        NeighSearchResult(std::vector<Point_t>& points)
            : points(points), ranges() {}

        // Copy constructor
        NeighSearchResult(const NeighSearchResult& other) = default;
    
        // Move constructor
        NeighSearchResult(NeighSearchResult&& other) noexcept = default;
            
        // Copy assignment operator
        NeighSearchResult& operator=(const NeighSearchResult& other) {
            if (this != &other) {
                points = other.points;
                ranges = other.ranges;
                numberOfPoints = other.numberOfPoints;
            }
            return *this;
        }

        // Move assignment operator
        NeighSearchResult& operator=(NeighSearchResult&& other) noexcept = default;

        // Adds a new range of point cloud indexes and updates numberOfPoints
        inline void addRange(size_t first, size_t last) {
            if (first <= last) {
                ranges.emplace_back(first, last);
                numberOfPoints += (last - first + 1);
            }
        }

        class Iterator {
            public:
                using iterator_category = std::forward_iterator_tag;
                using value_type = Point_t;
                using difference_type = std::ptrdiff_t;
                using pointer = const Point_t*;
                using reference = const Point_t&;
        
                Iterator(const NeighSearchResult& result, size_t currentRange)
                        : result(result), currentRange(currentRange), 
                        currentIndex((currentRange < result.ranges.size()) ? result.ranges[currentRange].first : SIZE_MAX) {
                    updateCurrentPoint();
                }
        
                reference operator*() const { return *currentPoint; }
                pointer operator->() const { return currentPoint; }
        
                Iterator& operator++() {
                    ++currentIndex;
                    updateCurrentPoint();
                    return *this;
                }
        
                Iterator operator++(int) {
                    Iterator temp = *this;
                    ++(*this);
                    return temp;
                }
        
                bool operator==(const Iterator& other) const {
                    return currentIndex == other.currentIndex;
                }
        
                bool operator!=(const Iterator& other) const { return !(*this == other); }
        
            private:
                const NeighSearchResult& result;
                pointer currentPoint = nullptr;
                size_t currentRange;
                size_t currentIndex;
                
                void updateCurrentPoint() {
                    // Skip empty ranges by advancing currentRange until a valid one is found
                    while ( currentRange < result.ranges.size() && 
                            currentIndex >= result.ranges[currentRange].second) {
                        ++currentRange;
                        if (currentRange < result.ranges.size()) {
                            currentIndex = result.ranges[currentRange].first;
                        }
                    }
                
                    // If we exited the loop because we ran out of ranges, stop iteration
                    if (currentRange >= result.ranges.size()) {
                        currentPoint = nullptr;
                        currentIndex = SIZE_MAX; // reset to invalid index
                        return;
                    }
                
                    // Update current point
                    currentPoint = &result.points[currentIndex];
                }
            };
        
        Iterator begin() const {
            return Iterator(*this, 0);
        }

        Iterator end() const {
            return Iterator(*this, ranges.size());
        }

        size_t size() const {
            return numberOfPoints;
        }
};
