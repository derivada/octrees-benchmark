# - Find PCL
# Find the PCL headers and libraries.
#
# This module defines:
#  PCL_INCLUDE_DIRS - where to find point_cloud.h, etc.
#  PCL_LIBRARIES    - List of libraries when using PCL.
#  PCL_FOUND        - True if PCL found.


find_path(PCL_INCLUDE_DIR
    NAMES pcl/pcl_config.h
    PATHS ${CMAKE_SOURCE_DIR}/lib/pcl/include ${CMAKE_PREFIX_PATH}
    PATH_SUFFIXES pcl-1.15 pcl-1.14 pcl
    DOC "Path to PCL include directory"
)

find_library(PCL_LIBRARY
    NAMES pcl_common
    PATHS ${CMAKE_SOURCE_DIR}/lib/pcl/lib ${CMAKE_PREFIX_PATH}
    PATH_SUFFIXES lib lib64
    DOC "Path to PCL libraries"
)

# add multiple libraries to PCL_LIBRARIES if they are found
set(PCL_LIBRARIES "")
if (PCL_LIBRARY)
    list(APPEND PCL_LIBRARIES ${PCL_LIBRARY})
    # Check for additional PCL libraries (e.g., pcl_common, pcl_io, etc
    foreach(lib io octree kdtree)
        find_library(PCL_${lib}_LIBRARY
            NAMES pcl_${lib}
            PATHS ${CMAKE_SOURCE_DIR}/lib/pcl/lib ${CMAKE_PREFIX_PATH}
            PATH_SUFFIXES lib lib64
        )
        if (PCL_${lib}_LIBRARY)
            list(APPEND PCL_LIBRARIES ${PCL_${lib}_LIBRARY})
        endif()
    endforeach()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(PCL
    REQUIRED_VARS PCL_LIBRARY PCL_INCLUDE_DIR
)   

if(PCL_FOUND)
    set(PCL_INCLUDE_DIRS ${PCL_INCLUDE_DIR})
    # Create an imported target for modern CMake usage
    if(NOT TARGET PCL::PCL)
        add_library(PCL::PCL UNKNOWN IMPORTED)
        set_target_properties(PCL::PCL PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${PCL_INCLUDE_DIRS}"
            IMPORTED_LOCATION "${PCL_LIBRARIES}"
        )
    endif()
endif()

mark_as_advanced(PCL_INCLUDE_DIR PCL_LIBRARY)