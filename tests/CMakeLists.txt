# Include FetchContent for GoogleTest integration
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Enable CTest integration
include(GoogleTest)
enable_testing()

# Point Tests
add_executable(test_points geometry/test_points.cpp)
target_link_libraries(test_points PUBLIC octrees-benchmark_shared gtest_main)
gtest_discover_tests(test_points)

# Encoder Tests
add_executable(test_encoders encoding/test_encoders.cpp)
target_link_libraries(test_encoders PUBLIC octrees-benchmark_shared gtest_main)
gtest_discover_tests(test_encoders)

# Octree Tests
add_executable(test_octree structures/test_octree.cpp)
target_link_libraries(test_octree PUBLIC octrees-benchmark_shared gtest_main)
gtest_discover_tests(test_octree)

add_executable(test_octree_advanced structures/test_octree_advanced.cpp)
target_link_libraries(test_octree_advanced PUBLIC octrees-benchmark_shared gtest_main)
gtest_discover_tests(test_octree_advanced)

# New Geometry Tests
add_executable(test_geometry geometry/test_geometry.cpp)
target_link_libraries(test_geometry PUBLIC octrees-benchmark_shared gtest_main)
gtest_discover_tests(test_geometry)

# New NeighborSet Tests
add_executable(test_neighbor_set structures/test_neighbor_set.cpp)
target_link_libraries(test_neighbor_set PUBLIC octrees-benchmark_shared gtest_main)
gtest_discover_tests(test_neighbor_set)

# Integration/Stress Tests
add_executable(test_complex integration/test_complex.cpp)
target_link_libraries(test_complex PUBLIC octrees-benchmark_shared gtest_main)
gtest_discover_tests(test_complex)
