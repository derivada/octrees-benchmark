# - Find LASlib
# Find the LASlib headers and libraries.
#
# This module defines:
#  LASLIB_INCLUDE_DIRS - where to find lasreader.h, etc.
#  LASLIB_LIBRARIES    - List of libraries when using LASlib.
#  LASLIB_FOUND        - True if LASlib found.

find_path(LASLIB_INCLUDE_DIR
    NAMES lasreader.hpp
    PATHS ${CMAKE_SOURCE_DIR}/lib/LASlib/include/LASlib ${CMAKE_PREFIX_PATH}
    PATH_SUFFIXES include
    DOC "Path to LASlib include directory"
)

find_library(LASLIB_LIBRARY
    NAMES LASlib
    PATHS ${CMAKE_SOURCE_DIR}/lib/LASlib/lib/LASlib ${CMAKE_PREFIX_PATH}
    PATH_SUFFIXES lib lib64
    DOC "Path to LASlib library"
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LASLIB
    REQUIRED_VARS LASLIB_LIBRARY LASLIB_INCLUDE_DIR
)

if(LASLIB_FOUND)
    set(LASLIB_LIBRARIES ${LASLIB_LIBRARY})
    set(LASLIB_INCLUDE_DIRS ${LASLIB_INCLUDE_DIR})
    
    # Create an imported target for modern CMake usage
    if(NOT TARGET LASLIB::LASLIB)
        add_library(LASLIB::LASLIB UNKNOWN IMPORTED)
        set_target_properties(LASLIB::LASLIB PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${LASlib_INCLUDE_DIRS}"
            IMPORTED_LOCATION "${LASlib_LIBRARIES}"
        )
    endif()
endif()

mark_as_advanced(LASLIB_INCLUDE_DIR LASLIB_LIBRARY)