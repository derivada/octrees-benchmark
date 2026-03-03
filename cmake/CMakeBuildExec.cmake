# ---  B U I L D I N G  --- #
# ------------------------- #

# Define a macro to link libraries to a target
macro(link_project_dependencies target_name)
    if (TARGET OpenMP::OpenMP_CXX)
        target_link_libraries(${target_name} PUBLIC OpenMP::OpenMP_CXX)
    endif ()

    if (TARGET armadillo::armadillo)
        target_link_libraries(${target_name} PUBLIC armadillo::armadillo)
    else ()
        target_link_libraries(${target_name} PUBLIC ${ARMADILLO_LIBRARIES})
    endif ()

    if (TARGET Eigen3::Eigen)
        target_link_libraries(${target_name} PUBLIC Eigen3::Eigen)
    else ()
        target_link_libraries(${target_name} PUBLIC ${EIGEN_LIBRARIES})
    endif ()

    target_link_libraries(${target_name} PUBLIC ${LASLIB_LIBRARIES})

    if (TARGET PCL::PCL)
        target_compile_definitions(${target_name} PUBLIC HAVE_PCL)

        foreach (lib ${PCL_LIBRARIES})
            target_link_libraries(${target_name} PUBLIC ${lib})
        endforeach ()
    endif ()

    if (TARGET Papi::Papi)
        target_link_libraries(${target_name} PUBLIC Papi::Papi)
    else ()
        target_link_libraries(${target_name} PUBLIC ${PAPI_LIBRARIES})
    endif ()

    # PicoTree
    if (PICOTREE_FOUND)
        target_compile_definitions(${target_name} PUBLIC HAVE_PICOTREE)
    endif()

    if (TARGET pico_tree)
        target_link_libraries(${target_name} PUBLIC pico_tree)
    endif()

    # Google Benchmark
    if (TARGET benchmark::benchmark)
        target_compile_definitions(${target_name} PUBLIC HAVE_GOOGLE_BENCHMARK)
        target_link_libraries(${target_name} PUBLIC benchmark::benchmark)
    endif()

endmacro()

# Static library
add_library(${PROJECT_NAME}_static STATIC ${lib_sources})
link_project_dependencies(${PROJECT_NAME}_static)

# Shared library
add_library(${PROJECT_NAME}_shared SHARED ${lib_sources})
link_project_dependencies(${PROJECT_NAME}_shared)

# Executable
add_executable(${PROJECT_NAME} ${sources})
link_project_dependencies(${PROJECT_NAME})

# Locality benchmarks with Google Benchmark
if (BUILD_BENCHMARKS AND TARGET benchmark::benchmark)
    if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/locality_google_benchmark.cpp")
        add_executable(locality-benchmark benchmarks/locality_google_benchmark.cpp)
        target_link_libraries(locality-benchmark PRIVATE ${PROJECT_NAME}_shared benchmark::benchmark)
    else()
        message(WARNING "Skipping locality-benchmark target: benchmarks/locality_google_benchmark.cpp not found")
    endif()
endif ()

# Set Link Time Optimization (LTO)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flto=auto")