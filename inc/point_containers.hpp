#pragma once

#include <cassert>
#include <concepts>
#include <cstddef>
#include <iostream>
#include <memory>
#include <variant>
#include <vector>
#include "Geometry/point.hpp"


template <typename T>
concept PointContainer = 
    std::copyable<T> && std::movable<T> &&
    requires(T c, size_t i, const Point& p) {
        { c.size() } -> std::convertible_to<size_t>;
        { c.resize(i) };
        { c.clear() };
        { c.set(i, p) };
        { c[i] } -> std::convertible_to<Point>;
        { c.begin() } -> std::input_iterator;
        { c.end() } -> std::input_iterator;
        { c.push_back(p) };
    };



// -----------------------------------------------------
// AoS container: wrapper around std::vector<Point>
// -----------------------------------------------------
class PointsAoS {
    std::vector<Point> data;

public:
    PointsAoS(size_t n = 0) : data(n) {}

    PointsAoS(const PointsAoS&) = default;
    PointsAoS& operator=(const PointsAoS&) = default;
    PointsAoS(PointsAoS&&) noexcept = default;
    PointsAoS& operator=(PointsAoS&&) noexcept = default;

    // Access (read-only or read-write)
    Point& operator[](size_t i) { return data[i]; }
    const Point& operator[](size_t i) const { return data[i]; }

    // Set function for consistency with PointsSoA
    void set(size_t i, const Point& p) {
        data[i] = p;
    }

    void resize(size_t n) { data.resize(n); }
    size_t size() const { return data.size(); }
    void clear() { data.clear(); }

    void push_back(const Point& p) { data.push_back(p); }

    // Iterators (same pattern as PointsSoA)
    using Iterator = std::vector<Point>::const_iterator;

    inline Iterator begin() const { return data.begin(); }
    inline Iterator end() const { return data.end(); }
    inline Iterator cbegin() const { return data.cbegin(); }
    inline Iterator cend() const { return data.cend(); }
};


#pragma once

// -----------------------------------------------------
// SoA container (SIMD)
// -----------------------------------------------------
class PointsSoA {
    std::vector<double> xs, ys, zs;
    std::vector<unsigned int> ids;
public:
    PointsSoA(size_t n = 0) : xs(n), ys(n), zs(n), ids(n) {}

    PointsSoA(const PointsSoA&) = default;
    PointsSoA& operator=(const PointsSoA&) = default;
    PointsSoA(PointsSoA&&) noexcept = default;
    PointsSoA& operator=(PointsSoA&&) noexcept = default;

    Point operator[](size_t i) const {
        return Point{ids[i], xs[i], ys[i], zs[i]};
    }

    void set(size_t i, const Point& p) {
        xs[i] = p.getX();
        ys[i] = p.getY();
        zs[i] = p.getZ();
        ids[i] = p.id();
    }

    size_t size() const { return xs.size(); }
    void resize(size_t n) {
        xs.resize(n); ys.resize(n); zs.resize(n); ids.resize(n);
    }
    void clear() {
        xs.clear(); ys.clear(); zs.clear(); ids.clear();
    }

    void push_back(const Point& p) {
        xs.push_back(p.getX());
        ys.push_back(p.getY());
        zs.push_back(p.getZ());
        ids.push_back(p.id());
    }


    double* dataX() { return xs.data(); }
    double* dataY() { return ys.data(); }
    double* dataZ() { return zs.data(); }
    unsigned int* dataIds() { return ids.data(); }

    const double* dataX() const { return xs.data(); }
    const double* dataY() const { return ys.data(); }
    const double* dataZ() const { return zs.data(); }
    const unsigned int* dataIds() const { return ids.data(); }

    // -------------------------------
    // SoA Iterator (yields Point copies)
    // -------------------------------
    class Iterator {
        const PointsSoA* soa;
        size_t index;

    public:
        using value_type = Point;
        using reference = Point;
        using pointer = void;
        using difference_type = std::ptrdiff_t;
        using iterator_category = std::forward_iterator_tag;

        Iterator(const PointsSoA* soa_, size_t index_)
            : soa(soa_), index(index_) {}

        Point operator*() const { return (*soa)[index]; }

        Iterator& operator++() {
            ++index;
            return *this;
        }

        Iterator operator++(int) {
            Iterator tmp = *this;
            ++(*this);
            return tmp;
        }

        bool operator==(const Iterator& other) const {
            return index == other.index && soa == other.soa;
        }

        bool operator!=(const Iterator& other) const {
            return !(*this == other);
        }
    };

    inline Iterator begin() const { return Iterator(this, 0); }
    inline Iterator end() const { return Iterator(this, size()); }
    inline Iterator cbegin() const { return begin(); }
    inline Iterator cend() const { return end(); }
};