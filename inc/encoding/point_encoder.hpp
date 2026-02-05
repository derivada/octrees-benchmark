#pragma once

#include <bitset>
#include <cstdint>
#include <immintrin.h>
#include <optional>

#include "benchmarking/encoding_log.hpp"
#include "benchmarking/time_watcher.hpp"
#include "geometry/box.hpp"
#include "geometry/point.hpp"
#include "geometry/point_metadata.hpp"

// Base class for all Encoders
namespace PointEncoding {
    
using coords_t = uint_fast32_t;
using key_t = uint_fast64_t;

class PointEncoder {
public:
    virtual ~PointEncoder() = default;

    virtual key_t encode(coords_t x, coords_t y, coords_t z) const = 0;
    virtual void encodeVectorized(const uint32_t *x, const uint32_t *y, const uint32_t *z, std::vector<key_t> &keys, size_t i) const=0;
    virtual void decode(key_t code, coords_t &x, coords_t &y, coords_t &z) const = 0;
    
    virtual uint32_t maxDepth() const = 0;
    virtual double eps() const = 0;
    virtual key_t upperBound() const = 0;
    virtual uint32_t unusedBits() const = 0;
    virtual EncoderType getEncoder() const = 0; 
    virtual std::string getEncoderName() const = 0;
    virtual std::string getShortEncoderName() const = 0;
    virtual bool is3D() const = 0;
    
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

    template <PointContainer Container>
    std::vector<key_t> encodePoints(const Container &points, const Box &bbox) const {
        size_t n = points.size();
        std::vector<key_t> codes(n);
        #pragma omp parallel for schedule(static)
            for(size_t i = 0; i < n; i++) {
                codes[i] = encodeFromPoint(points[i], bbox);
            }
        return codes;
    }

#if defined(__AVX512F__) && defined(__AVX512DQ__)
    template <PointContainer Container>
    void encodePointsVectorized(const Container &points, const Box &bbox, std::vector<key_t> &keys) const
    {
        size_t n = points.size();

        if constexpr (std::is_same_v<Container, PointsSoA>) // Only to avoid compilation errors
        {
            const auto *soa = dynamic_cast<const PointsSoA *>(&points);
            if (!soa)
            {
                std::cerr << "  [Error] Could not cast to PointsSoA\n";
                return;
            }

            // Constants
            __m512d bboxCenterX = _mm512_set1_pd(bbox.center().getX());
            __m512d bboxCenterY = _mm512_set1_pd(bbox.center().getY());
            __m512d bboxCenterZ = _mm512_set1_pd(bbox.center().getZ());
            __m512d bboxRadiiX = _mm512_set1_pd(bbox.radii().getX());
            __m512d bboxRadiiY = _mm512_set1_pd(bbox.radii().getY());
            __m512d bboxRadiiZ = _mm512_set1_pd(bbox.radii().getZ());

            __m512d two = _mm512_set1_pd(2.0);
            __m512d scale = _mm512_set1_pd(1 << maxDepth());
            __m512d maxCoord = _mm512_set1_pd((1u << maxDepth()) - 1u);

            // 2 Iterations of 8 elements, and acumulates 16 elements at a time
            #pragma omp parallel
            {
                #pragma omp for schedule(static)
                for (size_t i = 0; i < n - 15; i += 16)
                {
                    alignas(64) double x_vals[8], y_vals[8], z_vals[8];
                    alignas(64) uint32_t x[16], y[16], z[16];

                    // Load 8 elements
                    __m512d pointsX = _mm512_loadu_pd(soa->dataX() + i);
                    __m512d pointsY = _mm512_loadu_pd(soa->dataY() + i);
                    __m512d pointsZ = _mm512_loadu_pd(soa->dataZ() + i);

                    // Put physical coords into the unit cube: ((p - center) + radii) / (2 * radii)
                    __m512d x_transf = _mm512_div_pd(
                        _mm512_add_pd(_mm512_sub_pd(pointsX, bboxCenterX), bboxRadiiX),
                        _mm512_mul_pd(two, bboxRadiiX));
                    __m512d y_transf = _mm512_div_pd(
                        _mm512_add_pd(_mm512_sub_pd(pointsY, bboxCenterY), bboxRadiiY),
                        _mm512_mul_pd(two, bboxRadiiY));
                    __m512d z_transf = _mm512_div_pd(
                        _mm512_add_pd(_mm512_sub_pd(pointsZ, bboxCenterZ), bboxRadiiZ),
                        _mm512_mul_pd(two, bboxRadiiZ));

                    // Scale to [0,2^L)^3 for morton encoding, handle edge case where coordinate could be 2^L if _transf is exactly 1.0
                    __m512d x_scaled = _mm512_min_pd(_mm512_mul_pd(x_transf, scale), maxCoord);
                    __m512d y_scaled = _mm512_min_pd(_mm512_mul_pd(y_transf, scale), maxCoord);
                    __m512d z_scaled = _mm512_min_pd(_mm512_mul_pd(z_transf, scale), maxCoord);

                    // Cast double to uint32_t
                    __m256i x_uint = _mm512_cvttpd_epu32(x_scaled);
                    __m256i y_uint = _mm512_cvttpd_epu32(y_scaled);
                    __m256i z_uint = _mm512_cvttpd_epu32(z_scaled);

                    // Store
                    _mm256_store_si256((__m256i *)x, x_uint);
                    _mm256_store_si256((__m256i *)y, y_uint);
                    _mm256_store_si256((__m256i *)z, z_uint);

                    // Second Iteration
                    pointsX = _mm512_loadu_pd(soa->dataX() + i + 8);
                    pointsY = _mm512_loadu_pd(soa->dataY() + i + 8);
                    pointsZ = _mm512_loadu_pd(soa->dataZ() + i + 8);

                    // Put physical coords into the unit cube: ((p - center) + radii) / (2 * radii)
                    x_transf = _mm512_div_pd(
                        _mm512_add_pd(_mm512_sub_pd(pointsX, bboxCenterX), bboxRadiiX),
                        _mm512_mul_pd(two, bboxRadiiX));
                    y_transf = _mm512_div_pd(
                        _mm512_add_pd(_mm512_sub_pd(pointsY, bboxCenterY), bboxRadiiY),
                        _mm512_mul_pd(two, bboxRadiiY));
                    z_transf = _mm512_div_pd(
                        _mm512_add_pd(_mm512_sub_pd(pointsZ, bboxCenterZ), bboxRadiiZ),
                        _mm512_mul_pd(two, bboxRadiiZ));

                    // Scale to [0,2^L)^3 for morton encoding, handle edge case where coordinate could be 2^L if _transf is exactly 1.0
                    x_scaled = _mm512_min_pd(_mm512_mul_pd(x_transf, scale), maxCoord);
                    y_scaled = _mm512_min_pd(_mm512_mul_pd(y_transf, scale), maxCoord);
                    z_scaled = _mm512_min_pd(_mm512_mul_pd(z_transf, scale), maxCoord);

                    // Cast double to uint32_t
                    x_uint = _mm512_cvttpd_epu32(x_scaled);
                    y_uint = _mm512_cvttpd_epu32(y_scaled);
                    z_uint = _mm512_cvttpd_epu32(z_scaled);

                    // Store
                    _mm256_store_si256((__m256i *)&x[8], x_uint);
                    _mm256_store_si256((__m256i *)&y[8], y_uint);
                    _mm256_store_si256((__m256i *)&z[8], z_uint);

                    encodeVectorized(x, y, z, keys, i);
                }

                    // Process remaining items
                    #pragma omp single nowait
                    for (size_t i = n - (n % 16); i < n; ++i)
                    {
                        keys[i] = encodeFromPoint(points[i], bbox);
                    }
                }
            }
        }
#else
    template <PointContainer Container>
    void encodePointsVectorized(const Container &points, const Box &bbox, std::vector<key_t> &keys) const
    {

        size_t n = points.size();

        if constexpr (std::is_same_v<Container, PointsSoA>) // Only to avoid compilation errors
        {
            const auto *soa = dynamic_cast<const PointsSoA *>(&points);
            if (!soa)
            {
                std::cerr << "Error: Could not cast to PointsSoA\n";
                return;
            }

            // Constants
            __m256d bboxCenterX = _mm256_set1_pd(bbox.center().getX());
            __m256d bboxCenterY = _mm256_set1_pd(bbox.center().getY());
            __m256d bboxCenterZ = _mm256_set1_pd(bbox.center().getZ());
            __m256d bboxRadiiX = _mm256_set1_pd(bbox.radii().getX());
            __m256d bboxRadiiY = _mm256_set1_pd(bbox.radii().getY());
            __m256d bboxRadiiZ = _mm256_set1_pd(bbox.radii().getZ());

            __m256d two = _mm256_set1_pd(2.0);
            __m256d scale = _mm256_set1_pd(1 << maxDepth());
            __m256d maxCoord = _mm256_set1_pd((1u << maxDepth()) - 1u);

                // 2 Iterations of 4 elements, and acumulates 8 elements at a time
                #pragma omp parallel
                {
                    #pragma omp for schedule(static)
                    for (size_t i = 0; i < n - 7; i += 8)
                    {
                        alignas(32) uint32_t x[8], y[8], z[8];

                        // Load 4 elements
                        __m256d pointsX = _mm256_load_pd(soa->dataX() + i);
                        __m256d pointsY = _mm256_load_pd(soa->dataY() + i);
                        __m256d pointsZ = _mm256_load_pd(soa->dataZ() + i);

                        // Put physical coords into the unit cube: ((p - center) + radii) / (2 * radii)
                        __m256d x_transf = _mm256_div_pd(_mm256_add_pd(_mm256_sub_pd(pointsX, bboxCenterX), bboxRadiiX), _mm256_mul_pd(two, bboxRadiiX));
                        __m256d y_transf = _mm256_div_pd(_mm256_add_pd(_mm256_sub_pd(pointsY, bboxCenterY), bboxRadiiY), _mm256_mul_pd(two, bboxRadiiY));
                        __m256d z_transf = _mm256_div_pd(_mm256_add_pd(_mm256_sub_pd(pointsZ, bboxCenterZ), bboxRadiiZ), _mm256_mul_pd(two, bboxRadiiZ));

                        // Scale to [0,2^L)^3 for morton encoding, handle edge case where coordinate could be 2^L if _transf is exactly 1.0
                        __m256d x_scaled = _mm256_min_pd(_mm256_mul_pd(x_transf, scale), maxCoord);
                        __m256d y_scaled = _mm256_min_pd(_mm256_mul_pd(y_transf, scale), maxCoord);
                        __m256d z_scaled = _mm256_min_pd(_mm256_mul_pd(z_transf, scale), maxCoord);

                        // Store
                        alignas(32) double x_vals[4], y_vals[4], z_vals[4];
                        _mm256_store_pd(x_vals, x_scaled);
                        _mm256_store_pd(y_vals, y_scaled);
                        _mm256_store_pd(z_vals, z_scaled);

                        for (int j = 0; j < 4; ++j)
                        {
                            x[j] = static_cast<uint32_t>(x_vals[j]);
                            y[j] = static_cast<uint32_t>(y_vals[j]);
                            z[j] = static_cast<uint32_t>(z_vals[j]);
                        }

                        // Second iteration
                        pointsX = _mm256_load_pd(soa->dataX() + i + 4);
                        pointsY = _mm256_load_pd(soa->dataY() + i + 4);
                        pointsZ = _mm256_load_pd(soa->dataZ() + i + 4);

                        // Put physical coords into the unit cube: ((p - center) + radii) / (2 * radii)
                        x_transf = _mm256_div_pd(_mm256_add_pd(_mm256_sub_pd(pointsX, bboxCenterX), bboxRadiiX), _mm256_mul_pd(two, bboxRadiiX));
                        y_transf = _mm256_div_pd(_mm256_add_pd(_mm256_sub_pd(pointsY, bboxCenterY), bboxRadiiY), _mm256_mul_pd(two, bboxRadiiY));
                        z_transf = _mm256_div_pd(_mm256_add_pd(_mm256_sub_pd(pointsZ, bboxCenterZ), bboxRadiiZ), _mm256_mul_pd(two, bboxRadiiZ));

                        // Scale to [0,2^L)^3 for morton encoding, handle edge case where coordinate could be 2^L if _transf is exactly 1.0
                        x_scaled = _mm256_min_pd(_mm256_mul_pd(x_transf, scale), maxCoord);
                        y_scaled = _mm256_min_pd(_mm256_mul_pd(y_transf, scale), maxCoord);
                        z_scaled = _mm256_min_pd(_mm256_mul_pd(z_transf, scale), maxCoord);

                        // Store
                        _mm256_store_pd(x_vals, x_scaled);
                        _mm256_store_pd(y_vals, y_scaled);
                        _mm256_store_pd(z_vals, z_scaled);

                        for (int j = 0; j < 4; ++j)
                        {
                            x[j + 4] = static_cast<uint32_t>(x_vals[j]);
                            y[j + 4] = static_cast<uint32_t>(y_vals[j]);
                            z[j + 4] = static_cast<uint32_t>(z_vals[j]);
                        }

                        encodeVectorized(x, y, z, keys, i);
                    }

                    // Process remaining items
                    #pragma omp single nowait
                    for (size_t i = n - (n % 4); i < n; ++i)
                    {
                        keys[i] = encodeFromPoint(points[i], bbox);
                    }
                }
            }
        }
    #endif

        /**
         * @brief Parallel radix sort implementation for the SFC reordering.
         *
         * @returns The final array of encodings of the points
         */
        template <PointContainer Container>
        std::vector<key_t> sortPoints(
            Container &points,
            std::optional<std::vector<PointMetadata>> &meta_opt, 
            const Box &bbox, 
            std::shared_ptr<EncodingLog> log = nullptr) const
        {
            size_t n = points.size();
            constexpr int BITS_PER_PASS = 8;
            constexpr int NUM_BUCKETS = 1 << BITS_PER_PASS;
            constexpr size_t BUCKET_MASK = NUM_BUCKETS - 1;
            constexpr int NUM_PASSES = sizeof(key_t) * 8 / BITS_PER_PASS;

            std::vector<key_t> keys;

            TimeWatcher tw;
            tw.start();

            // Encoding
            if (mainOptions.containerType == ContainerType::SoA && this->getEncoder() == EncoderType::HILBERT_ENCODER_3D)
            {
                keys.resize(n);
                encodePointsVectorized(points, bbox, keys);
            }
            else
            {
                keys = encodePoints(points, bbox);
            }

            tw.stop();
            if (log)
                log->encodingTime = tw.getElapsedDecimalSeconds();

            tw.start();

            std::vector<key_t> buffer(n);
            Container bufferDecoded(n);
            std::vector<PointMetadata> metadata_buffer;
            if (meta_opt)
                metadata_buffer.resize(n);

            for (int pass = 0; pass < NUM_PASSES; pass++)
            {
                int shift = pass * BITS_PER_PASS;

                const int nThreads = omp_get_max_threads();
                std::vector<std::vector<size_t>> localHist(nThreads, std::vector<size_t>(NUM_BUCKETS, 0));

                // Step 1: Histogram
                #pragma omp parallel
                {
                    auto &hist = localHist[omp_get_thread_num()];
                    #pragma omp for schedule(static)
                    for (size_t i = 0; i < n; ++i)
                    {
                        size_t bucket = (keys[i] >> shift) & BUCKET_MASK;
                        hist[bucket]++;
                    }

                    // Step 2: Scan histograms to offsets
                    #pragma omp single
                    {
                        size_t offset = 0;
                        for (int b = 0; b < NUM_BUCKETS; b++)
                        {
                            for (int t = 0; t < nThreads; t++)
                            {
                                size_t val = localHist[t][b];
                                localHist[t][b] = offset;
                                offset += val;
                            }
                        }
                    }

                    // Step 3: Scatter to buffer using per-thread offsets
                    auto &localOffset = localHist[omp_get_thread_num()];
                    #pragma omp for schedule(static)
                    for (size_t i = 0; i < n; i++)
                    {
                        size_t bucket = (keys[i] >> shift) & BUCKET_MASK;
                        size_t pos = localOffset[bucket]++;
                        buffer[pos] = keys[i];
                        bufferDecoded.set(pos, points[i]);
                        if (meta_opt)
                            metadata_buffer[pos] = (*meta_opt)[i];
                    }
                }

                std::swap(points, bufferDecoded);
                std::swap(keys, buffer);
                if (meta_opt)
                    std::swap(*meta_opt, metadata_buffer);
            }

            tw.stop();
            if (log)
                log->sortingTime = tw.getElapsedDecimalSeconds();

            return keys;
        }

        /**
         * Original implementation, slower and doesn't use AVX.
         * Kept for reference.
         */
        template <PointContainer Container>
        std::vector<key_t> sortPointsOriginal(Container &points, 
            std::optional<std::vector<PointMetadata>> &meta_opt, const Box &bbox, std::shared_ptr<EncodingLog> log = nullptr) const {
            size_t n = points.size();
            // Choose 4, 8 or 16. 16 is less passes but requires more memory for the histograms and it offers better performance in most cases.
            constexpr int BITS_PER_PASS = 16;
            constexpr int NUM_BUCKETS = 1 << BITS_PER_PASS;
            constexpr size_t BUCKET_MASK = NUM_BUCKETS - 1;
            constexpr int NUM_PASSES = sizeof(key_t) * 8 / BITS_PER_PASS;
        
            TimeWatcher tw;
            tw.start();
        
            Container buffer(n);
            std::vector<PointMetadata> metadata_buffer;
            if (meta_opt) metadata_buffer.resize(n);
        
            for (int pass = 0; pass < NUM_PASSES; pass++) {
                int shift = pass * BITS_PER_PASS;
        
                // Step 1: Histogram
                const int nThreads = omp_get_max_threads();
                std::vector<std::vector<size_t>> localHist(nThreads, std::vector<size_t>(NUM_BUCKETS, 0));
        
                #pragma omp parallel
                {
                    int tid = omp_get_thread_num();
                    auto& hist = localHist[tid];
                    #pragma omp for nowait schedule(static)
                    for (size_t i = 0; i < n; ++i) {
                        size_t encoded_key = encodeFromPoint(points[i], bbox);
                        size_t bucket = (encoded_key >> shift) & BUCKET_MASK;
                        hist[bucket]++;
                    }
                }
        
                // Step 2: Scan histograms to offsets
                size_t offset = 0;
                for (int b = 0; b < NUM_BUCKETS; b++) {
                    for (int t = 0; t < nThreads; t++) {
                        size_t val = localHist[t][b];
                        localHist[t][b] = offset;
                        offset += val;
                    }
                }
        
                // Step 3: Scatter to buffer using per-thread offsets
                #pragma omp parallel
                {
                    int tid = omp_get_thread_num();
                    auto& localOffset = localHist[tid];
        
                    #pragma omp for schedule(static)
                    for (size_t i = 0; i < n; i++) {
                        size_t encoded_key = encodeFromPoint(points[i], bbox);
                        size_t bucket = (encoded_key >> shift) & BUCKET_MASK;
                        size_t pos = localOffset[bucket]++;
                        buffer.set(pos, points[i]);
                        if (meta_opt) metadata_buffer[pos] = (*meta_opt)[i];
                    }
                }
        
                std::swap(points, buffer);
                if (meta_opt) std::swap(*meta_opt, metadata_buffer);
            }
        
            tw.stop();
            if (log) log->sortingTime = tw.getElapsedDecimalSeconds();
            tw.start();
        
            // Final encoding
            std::vector<key_t> keys(n);
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < n; ++i) {
                keys[i] = encodeFromPoint(points[i], bbox);
            }
        
            tw.stop();
            if (log) log->encodingTime = tw.getElapsedDecimalSeconds();
        
            return keys;
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
         * @param meta_opt The optional metadata of the point cloud (if Lpoint is used, it is contained on the struct).
         * meta_opt is a parallel array to points and will be sorted in the same order
         */
        template <PointContainer Container>
        std::pair<std::vector<key_t>, Box> sortPoints(
            Container &points,
            std::optional<std::vector<PointMetadata>> &meta_opt, 
            std::shared_ptr<EncodingLog> log = nullptr) const
        {
            // Find the bbox
            TimeWatcher tw;
            tw.start();
            Vector radii;
            Point center = mbb(points, radii);
            Box bbox = Box(center, radii);
            tw.stop();
            if (log)
            {
                log->boundingBoxTime = tw.getElapsedDecimalSeconds();
            }

            // Call the regular sortPoints with metadata
            return std::make_pair(sortPoints<Container>(points, meta_opt, bbox, log), bbox);
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
        inline Point getCenter(key_t code, uint32_t level, const Box &bbox,
                               const double *halfLengths, const std::vector<Vector> &precomputedRadii) const
        {
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
                               bbox.minZ() + min_z * halfLengths[2] * 2) +
                           precomputedRadii[level];

            return center;
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
        assert(treeLevel <= maxDepth());
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
