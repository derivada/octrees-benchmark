#pragma once

#include "Geometry/point.hpp"
#include "Geometry/Box.hpp"
#include <cstdint>
#include <optional>
#include "Geometry/PointMetadata.hpp"
#include "encoding_octree_log.hpp"
#include "TimeWatcher.hpp"
#include <bitset>

// Base class for all Encoders
namespace PointEncoding {



using coords_t = uint_fast32_t;
using key_t = uint_fast64_t;

class PointEncoder {
public:
    virtual ~PointEncoder() = default;

    virtual key_t encode(coords_t x, coords_t y, coords_t z) const = 0;
    virtual void decode(key_t code, coords_t &x, coords_t &y, coords_t &z) const = 0;

    virtual uint32_t maxDepth() const = 0;
    virtual double eps() const = 0;
    virtual key_t upperBound() const = 0;
    virtual uint32_t unusedBits() const = 0;
    virtual std::string getEncoderName() const = 0;
    virtual std::string getShortEncoderName() const = 0;

    inline void getAnchorCoords(const Point& p, const Box &bbox, 
        coords_t &x, coords_t &y, coords_t &z) const  {
        // Put physical coords into the unit cube
        double x_transf = ((p.getX() - bbox.center().getX())  + bbox.radii().getX()) / (2 * bbox.radii().getX());
        double y_transf = ((p.getY() - bbox.center().getY())  + bbox.radii().getY()) / (2 * bbox.radii().getY());
        double z_transf = ((p.getZ() - bbox.center().getZ())  + bbox.radii().getZ()) / (2 * bbox.radii().getZ());
        
        // Scale to [0,2^L)^3 for morton encoding, handle edge case where coordinate could be 2^L if _transf is exactly 1.0
        coords_t maxCoord = (1u << maxDepth()) - 1u;
        x = std::min((coords_t) (x_transf * (1 << (maxDepth()))), maxCoord);
        y = std::min((coords_t) (y_transf * (1 << (maxDepth()))), maxCoord);
        z = std::min((coords_t) (z_transf * (1 << (maxDepth()))), maxCoord);
    }
    
    /// @brief A wrapper for doing PointEncoding::getAnchorCoords + encode in one single step     
    virtual key_t encodeFromPoint(const Point& p, const Box& bbox) const = 0;

    template <typename Point_t>
    std::vector<key_t> encodePoints(const std::vector<Point_t> &points, const Box &bbox) const {
        std::vector<key_t> codes(points.size());
        #pragma omp parallel for schedule(static)
            for(size_t i = 0; i < points.size(); i++) {
                codes[i] = encodeFromPoint(points[i], bbox);
            }
        return codes;
    }

    /**
     * @brief This function computes the encodings of the points and sorts them in
     * the given order. The points array is altered in-place here!
     * 
     * @details This is the main reordering function that we use to achieve spatial locality during neighbourhood
     * searches. Encodings are first computed using encodeFromPoint, put into a pairs vector and then reordered.
     * 
     * @param points The input/output array of 3D points (i.e. the point cloud)
     * @param codes The output array of computed codes (allocated here, only pass unallocated reference)
     * @param bbox The precomputed global bounding box of points
     */
    template <typename Point_t>
    std::vector<key_t> sortPoints(std::vector<Point_t> &points, const Box &bbox, std::shared_ptr<EncodingOctreeLog> log = nullptr) const {
        // Temporal vector of pairs
        std::cout << "sorting.." << std::endl;
        std::vector<std::pair<key_t, Point_t>> encoded_points(points.size());
        TimeWatcher tw;
        tw.start();
        // Compute encodings in parallel
        #pragma omp parallel for schedule(static)
            for(size_t i = 0; i < points.size(); i++) {
                encoded_points[i] = std::make_pair(encodeFromPoint(points[i], bbox), points[i]);
            }
        tw.stop();
        if(log)
            log->encodingTime = tw.getElapsedDecimalSeconds();
        // Sort the points
        tw.start();
        std::sort(encoded_points.begin(), encoded_points.end(),
            [](const auto& a, const auto& b) {
                return a.first < b.first;  // Compare only the morton codes
        });
        
        // Copy back sorted codes and points in parallel
        std::vector<key_t> codes(points.size());
        #pragma omp parallel for schedule(static)
            for(size_t i = 0; i < points.size(); i++) {
                codes[i] = encoded_points[i].first;
                points[i] = encoded_points[i].second;
            }
        tw.stop();
        if(log)
            log->sortingTime = tw.getElapsedDecimalSeconds();
        return codes;
    }

    template <typename Point_t>
    std::vector<key_t> sortPointsRadix(std::vector<Point_t>& points, const Box& bbox, std::shared_ptr<EncodingOctreeLog> log = nullptr) const {
        size_t n = points.size();
        // Configure to 4, 8 or 16 (higher -> extra memory, but potentially faster if cache is big)
        constexpr int BITS_PER_PASS = 16; 
        constexpr int NUM_BUCKETS = 1 << BITS_PER_PASS;
        constexpr size_t BUCKET_MASK = NUM_BUCKETS - 1;
        constexpr int NUM_PASSES = sizeof(key_t) * 8 / BITS_PER_PASS;
        TimeWatcher tw;
        tw.start();
        std::vector<Point_t> buffer(n);
        for (int pass = 0; pass < NUM_PASSES; ++pass) {
            int shift = pass * BITS_PER_PASS;
            // Step 1: Build histograms
            const int nThreads = omp_get_max_threads();
            std::vector<std::vector<size_t>> localHist(nThreads, std::vector<size_t>(NUM_BUCKETS, 0));
            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                auto& hist = localHist[tid];
                #pragma omp for nowait
                for (size_t i = 0; i < n; ++i) {
                    size_t encoded_key = encodeFromPoint(points[i], bbox);
                    size_t bucket = (encoded_key >> shift) & BUCKET_MASK;
                    hist[bucket]++;
                }
            }
            
            // Step 2: Convert local histogram to local offsets by running a 2D scan
            size_t offset = 0;
            for (int b = 0; b < NUM_BUCKETS; ++b) {
                for (int t = 0; t < nThreads; ++t) {
                    size_t val = localHist[t][b];
                    localHist[t][b] = offset;
                    offset += val;
                }
            }
    
            // Step 3: Redistribute into buffer using per-thread local offsets
            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                auto& localOffset = localHist[tid];
    
                #pragma omp for
                for (size_t i = 0; i < n; ++i) {
                    size_t encoded_key = encodeFromPoint(points[i], bbox);
                    size_t bucket = (encoded_key >> shift) & BUCKET_MASK;
                    size_t pos = localOffset[bucket]++;
                    buffer[pos] = points[i];
                }
            }
            std::swap(points, buffer);
        }

        tw.stop();
        if (log) log->sortingTime = tw.getElapsedDecimalSeconds();
        tw.start();    
        // Compute final encodings in parallel
        std::vector<key_t> keys(n);
        #pragma omp parallel for
        for (size_t i = 0; i < n; ++i) {
            keys[i] = { encodeFromPoint(points[i], bbox) };
        }
        tw.stop();
        if (log) log->encodingTime = tw.getElapsedDecimalSeconds();
        return keys;
    }
   
    template <typename Point_t>
    std::pair<std::vector<key_t>, Box> sortPoints(std::vector<Point_t> &points, std::shared_ptr<EncodingOctreeLog> log = nullptr) const {
        // Find the bbox
        TimeWatcher tw;
        tw.start();
        Vector radii;
        Point center = mbb(points, radii);
        Box bbox = Box(center, radii);
        tw.stop();
        std::cout << "NON-PARALLEL: center " << center << " , radii " << radii << " time: " << tw.getElapsedDecimalSeconds() << std::endl;
        tw.start();
        Vector radii_parallel;
        Point center_parallel = mbb_parallel(points, radii_parallel);
        Box bbox_parallel = Box(center_parallel, radii_parallel);
        tw.stop();
        std::cout << "PARALLEL: center " << center_parallel << " , radii " << radii_parallel << " time: " << tw.getElapsedDecimalSeconds()  << std::endl;
        if(log) {
            log->boundingBoxTime = tw.getElapsedDecimalSeconds();
            log->encoderType = getShortEncoderName();
            log->cloudSize = points.size();
        }
        // Call the regular sortPoints
        std::cout << "calling sortPoints" << std::endl;
        return std::make_pair(sortPoints<Point_t>(points, bbox, log), bbox);
    }

    /**
     * @brief A similar function to @see sortPoints(), but here the LiDAR metadata of the points is passed separately, and a tuple of 3 elements is sorted instead
     * of a pair. 
     */
    template <typename Point_t>
    std::vector<key_t> sortPoints(std::vector<Point_t> &points, 
        std::optional<std::vector<PointMetadata>> &meta_opt, const Box &bbox, std::shared_ptr<EncodingOctreeLog> log = nullptr) const {
        if(!meta_opt.has_value()) {
            // regular sort without metadata
            std::vector<Point_t> points_copy = points;
            TimeWatcher tw;
            tw.start();
            auto keys = sortPoints<Point_t>(points, bbox, log);
            tw.stop();
            std::cout << "normal: " << tw.getElapsedDecimalSeconds() << std::endl;
            tw.start();
            auto keys_copy = sortPointsRadix<Point_t>(points_copy, bbox, log);
            tw.stop();
            std::cout << "radix: " << tw.getElapsedDecimalSeconds() << std::endl;
            for(int i = 0; i<points.size(); i++) {
                if(points[i] != points_copy[i]) {
                    std::cout << "different points at index " << i << std::endl;
                    std::cout << "   normal: " << points[i].getX() << "," << points[i].getY() << "," << points[i].getZ() << std::endl;
                    std::cout << "   radix: " << points_copy[i].getX() << "," << points_copy[i].getY() << "," << points_copy[i].getZ() << std::endl;
                }
                if(keys[i] != keys_copy[i]) {
                    std::cout << "different keys at index " << i << std::endl;
                    std::cout << "   normal: " << keys[i] << std::endl;
                    std::cout << "   radix: " << keys_copy[i] << std::endl;
                }            
            }
            return keys;
        }
        // Temporal vector of 3-tuples
        // TODO: find a better way, since this consumes a lot of extra memory 
        std::vector<PointMetadata>& metadata = meta_opt.value();
        std::vector<std::tuple<key_t, Point_t, PointMetadata>> encoded_points(points.size());

        // Compute encodings in parallel
        TimeWatcher tw;
        tw.start();
        #pragma omp parallel for schedule(static)
            for(size_t i = 0; i < points.size(); i++) {
                encoded_points[i] = std::make_tuple(encodeFromPoint(points[i], bbox), points[i], metadata[i]);
            }
        tw.stop();
        if(log)
            log->encodingTime = tw.getElapsedDecimalSeconds();
        tw.start();
        // TODO: implement parallel radix sort and stop using vector of tuples if possible (only sort the encodings)
        std::sort(encoded_points.begin(), encoded_points.end(),
            [](const auto& a, const auto& b) {
                return std::get<0>(a) < std::get<0>(b);  // Compare only the codes
        });
        
        // Copy back sorted codes, points, and metadata in parallel
        std::vector<key_t> codes(points.size());
        #pragma omp parallel for schedule(static)
            for(size_t i = 0; i < points.size(); i++) {
                std::tie(codes[i], points[i], metadata[i]) = encoded_points[i];
            }
        tw.stop();
        if(log)
            log->sortingTime = tw.getElapsedDecimalSeconds();
        return codes;
    }

    template <typename Point_t>
    std::pair<std::vector<key_t>, Box> sortPoints(std::vector<Point_t> &points, 
        std::optional<std::vector<PointMetadata>> &meta_opt, std::shared_ptr<EncodingOctreeLog> log = nullptr) const {
        // Find the bbox
        TimeWatcher tw;
        tw.start();
        Vector radii;
        Point center = mbb(points, radii);
        Box bbox = Box(center, radii);
        tw.stop();
        if(log) {
            log->boundingBoxTime = tw.getElapsedDecimalSeconds();
            log->encoderType = getShortEncoderName();
            log->cloudSize = points.size();
        }
        // Call the regular sortPoints with metadata
        return std::make_pair(sortPoints<Point_t>(points, meta_opt, bbox, log), bbox);
    }


    /**
     * @brief Get the center and radii of an octant at a given octree level.
     * 
     * @tparam Encoder The encoder type.
     * @param code The encoded key.
     * @param level The level in the octree.
     * @param bbox The bounding box.
     * @param halfLengths The half lengths of the bounding box.
     * @param precomputedRadii The precomputed radii corresponding to that level.
     * @return A pair containing the center point and the radii vector.
     */
    inline std::pair<Point, Vector> getCenterAndRadii(key_t code, uint32_t level, const Box &bbox, 
            const double* halfLengths, const std::vector<Vector> precomputedRadii) const {
        // Decode the points back into their integer coordinates
        coords_t min_x, min_y, min_z;
        decode(code, min_x, min_y, min_z);

        // Now adjust the coordinates so they indicate the lowest code in the current level
        // In Morton curves this is not needed, but in Hilbert curves it is, since it can return any corner instead of lower one we need
        // Fun fact: finding this mistake when adding Hilbert curves took 6 hours of debugging
        coords_t mask = ((1u << maxDepth()) - 1) ^ ((1u << (maxDepth() - level)) - 1);
        min_x &= mask, min_y &= mask, min_z &= mask;

        // Find the physical center by multiplying the encoding with the halfLength
        // to get to the low corner of the cell, and then adding the radii of the cell
        Point center = Point(
            bbox.minX() + min_x * halfLengths[0] * 2, 
            bbox.minY() + min_y * halfLengths[1] * 2, 
            bbox.minZ() + min_z * halfLengths[2] * 2
        ) + precomputedRadii[level];
        
        return {center, precomputedRadii[level]};
    }

    /// @brief Count the leading zeros in a binary key.
    constexpr uint32_t countLeadingZeros(key_t x) 
    {
        #if defined(__GNUC__) || defined(__clang__)
            if (x == 0) return 8 * sizeof(key_t);
            // 64-bit keys
            if constexpr (sizeof(key_t) == 8) {
                return __builtin_clzll(x);
            }
            // 32-bit keys
            else {
                return __builtin_clz(x);
            }
        #else
            uint32_t depth = 0;
            for (; x != 1; x >>= 3, depth++);
            return depth;
        #endif
    }

    /// @brief Check if a number is a power of 8.
    constexpr bool isPowerOf8(key_t n) {
        key_t lz = countLeadingZeros(n - 1) - unusedBits();
        return lz % 3 == 0 && !(n & (n - 1));
    }

    /// @brief Get the level in the octree from a given morton code
    inline uint32_t getLevel(key_t range) {
        assert(isPowerOf8(range));
        if(range == upperBound())
            return key_t(0);
        return (countLeadingZeros(range - key_t(1)) - unusedBits()) / key_t(3);
    }

    /// @brief Get the sibling ID of the code at a given level
    constexpr uint32_t getSiblingId(key_t code, uint32_t level) {
        // Shift 3*(21-level) to get the 3 bits corresponding to the level
        return (code >> (key_t(3) * (maxDepth() - level))) & key_t(7);
    }   

    /**
     * @brief Get the maximum range allowed in a level of the tree.
     * 
     * @example At level 0 the range is the entire 63 bit span, at level 10 the range is 11*3 bit span
     * at level 20 (last to minimum), the range will just be 8 between each node, i.e. the 8 siblings that
     * can be on max level 21 between two nodes at level 20
     * 
     * @param treeLevel The level in the tree.
     * @return The maximum range allowed in the level.
     */
    constexpr key_t nodeRange(uint32_t treeLevel)
    {
        assert(treeLevel < maxDepth());
        uint32_t shifts = maxDepth() - treeLevel;

        return 1ul << (key_t(3) * shifts);
    }

    /// @brief Returns the amount of bits before the placeholder bit.
    constexpr uint32_t decodePrefixLength(key_t code) {
        return 8 * sizeof(key_t) - 1 - countLeadingZeros(code);
    }

    /**
     * @brief Transforms the SFC (leaf) format into the placeholder bit format, by putting non empty octants into the 
     * beginning of the key and then adding a placeholder bit after them.
     * @example 0 100 000 000 000 ... 000 --> 0 000 000 ... 000 001 100
     * @tparam Encoder The encoder type.
     * @param code The encoded key.
     * @param level The level of the octree on which we are, i.e. the number of octants considered in the key
     * @return The encoded key with the placeholder bit.
     */
    constexpr key_t encodePlaceholderBit(key_t code, int level) {
        key_t ret = code >> 3 * (maxDepth() - level);
        key_t placeHolderMask = key_t(1) << (3*level);

        return placeHolderMask | ret;
    }

    /// @brief Inverse operation to encodePlaceholderBit
    constexpr key_t decodePlaceholderBit(key_t code) {
        int prefixLength        = decodePrefixLength(code);
        key_t placeHolderMask = key_t(1) << prefixLength;
        key_t ret             = code ^ placeHolderMask;

        return ret << (key_t(3) * maxDepth() - prefixLength);
    }

    /// @brief Find the common prefix between two keys in placeholder format.
    constexpr int32_t commonPrefix(key_t key1, key_t key2) {
        return int32_t(countLeadingZeros(key1 ^ key2)) - unusedBits();
    }

    /// @brief Extract the octal digit at a given position in a code in the SFC format.
    constexpr unsigned octalDigit(key_t code, uint32_t position) {
        return (code >> (key_t(3) * (maxDepth() - position))) & key_t(7);
    }
    
    /// @brief Get the ceiling of the logarithm base 8 of a number.
    constexpr uint32_t log8ceil(key_t n) {
        if (n == key_t(0)) { return 0; }

        uint32_t lz = countLeadingZeros(n - key_t(1));
        return maxDepth() - (lz - unusedBits()) / 3;
    }
};
} // namespace PointEncoding
