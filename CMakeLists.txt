cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(octrees-benchmark LANGUAGES C CXX)

# Configure CMake
message("Configuring CMake ...")
include(cmake/CMakeConfig.cmake)
message("CMake configured!\n")

# Handle sources
message("Handling sources ...")
include(cmake/CMakeSources.cmake)
message("Sources handled!\n")

# Load libraries
message("Loading libraries ...")
include(cmake/CMakeLibraries.cmake)
message("Libraries loaded!\n")

# Build executable
message("Building ${PROJECT_NAME} ...")
include(cmake/CMakeBuildExec.cmake)
message("Project built!\n\n")

# Tests
option(BUILD_TESTS "Build unit tests (download GoogleTest)" ON)
if(BUILD_TESTS)
    include(CTest)
    enable_testing()
    add_subdirectory(tests)
endif()

# Examples
add_subdirectory(examples)