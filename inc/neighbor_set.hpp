#pragma once
#include <vector>
#include <iterator>

/**
 * @brief A structure that stores a set of neighboring points within a linear octree.
 * 
 * This class provides a range-based structure for efficiently accessing points in the neighborhood
 * of a given point. The NeighborSet maintains references to an external point cloud and only
 * stores minimal information (ranges of indices) to minimize copying while allowing fast iteration.
 *
 * @tparam Point_t The type representing a point in the dataset.
 */
template <typename Point_t>
class NeighborSet {
    public:
        /// @brief Reference to the external point cloud.
        std::vector<Point_t>* points = nullptr;

        /// @brief Ranges of indices defining neighborhoods.
        std::vector<std::pair<size_t, size_t>> ranges;

        /// @brief Total number of points in the neighborhood.
        size_t numberOfPoints = 0;

        /// @brief Empty constructor
        NeighborSet() = default;

        /**
         * @brief Constructs a NeighborSet with a reference to an existing point cloud.
         * @param points Reference to the point cloud.
         */        
        NeighborSet(std::vector<Point_t>* points)
            : points(points), ranges() {}

        /// @brief Copy constructor
        NeighborSet(const NeighborSet& other)
            : points(other.points),  // Same external reference
            ranges(other.ranges),
            numberOfPoints(other.numberOfPoints) {}

        /// @brief Copy assignment operator
        NeighborSet& operator=(const NeighborSet& other) {
            if (this != &other) {
                points = other.points;  // Same external reference
                ranges = other.ranges;
                numberOfPoints = other.numberOfPoints;
            }
            return *this;
        }

        /// @brief Move constructor
        NeighborSet(NeighborSet&& other) noexcept
            : points(other.points),  // Move ownership of reference
            ranges(std::move(other.ranges)),
            numberOfPoints(other.numberOfPoints) {
            other.points = nullptr;  // Invalidate moved-from object
            other.numberOfPoints = 0;
        }

        /// @brief Move assignment operator
        NeighborSet& operator=(NeighborSet&& other) noexcept {
            if (this != &other) {
                points = other.points;
                ranges = std::move(other.ranges);
                numberOfPoints = other.numberOfPoints;

                other.points = nullptr;  // Invalidate moved-from object
                other.numberOfPoints = 0;
            }
            return *this;
        }

        /**
         * @brief Adds a new range of indices inside the neighborhood.
         * 
         * The range represents a contiguous block of points within the point cloud. The range
         * is considered to be left-exclusive i.e. [first, last)
         * 
         * @param first Index of the first point in the range (included).
         * @param last Index of the last point in the range (not included).
         */
        inline void addRange(size_t first, size_t last) {
            if (first <= last) {
                ranges.emplace_back(first, last);
                numberOfPoints += (last - first);
            }
        }

        /// @brief Iterator for traversing the NeighborSet efficiently.
        class Iterator {
            public:
                using iterator_category = std::forward_iterator_tag;
                using value_type = Point_t;
                using difference_type = std::ptrdiff_t;
                using pointer = const Point_t*;
                using reference = const Point_t&;

                /**
                 * @brief Constructs an iterator for a given NeighborSet.
                 * @param result Reference to the NeighborSet.
                 * @param currentRange Initial range index.
                 */
                Iterator(const NeighborSet& result, size_t currentRange)
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
                const NeighborSet& result;
                pointer currentPoint = nullptr;

                /// @brief Index of the current range in ranges.
                size_t currentRange;
                /// @brief Current index within the range.
                size_t currentIndex;
                
                /// @brief Updates the iterator to point to the next valid position.
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
                    currentPoint = &(*result.points)[currentIndex];
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

        [[nodiscard]] bool empty() const noexcept {
            return begin() == end();
        }
};
