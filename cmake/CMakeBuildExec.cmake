# ---  B U I L D I N G  --- #
# ------------------------- #

# Define a macro to link libraries to a target
macro(link_project_dependencies target_name)
    if (TARGET OpenMP::OpenMP_CXX)
        target_link_libraries(${target_name} PRIVATE OpenMP::OpenMP_CXX)
    endif ()

    if (TARGET armadillo::armadillo)
        target_link_libraries(${target_name} PRIVATE armadillo::armadillo)
    else ()
        target_link_libraries(${target_name} PRIVATE ${ARMADILLO_LIBRARIES})
    endif ()

    if (TARGET Eigen3::Eigen)
        target_link_libraries(${target_name} PRIVATE Eigen3::Eigen)
    else ()
        target_link_libraries(${target_name} PRIVATE ${EIGEN_LIBRARIES})
    endif ()

    target_link_libraries(${target_name} PRIVATE ${LASLIB_LIBRARIES})

    if (TARGET PCL::PCL)
        target_compile_definitions(${target_name} PRIVATE HAVE_PCL)

        foreach (lib ${PCL_LIBRARIES})
            target_link_libraries(${target_name} PRIVATE ${lib})
        endforeach ()
    endif ()

    if (TARGET Papi::Papi)
        target_link_libraries(${target_name} PRIVATE Papi::Papi)
    else ()
        target_link_libraries(${target_name} PRIVATE ${PAPI_LIBRARIES})
    endif ()

    # PicoTree
    if (TARGET pico_tree)
        target_compile_definitions(${target_name} PRIVATE HAVE_PICOTREE)
        target_link_libraries(${target_name} PRIVATE pico_tree)
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

# Set Link Time Optimization (LTO)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flto=auto")