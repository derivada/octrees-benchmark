# Tests directory
add_executable(test_library example.cpp)
target_link_libraries(test_library PUBLIC octrees-benchmark_shared)
