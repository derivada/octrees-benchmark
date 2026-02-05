# - Find PAPI
# Find the PAPI headers and libraries.
#
# This module defines:
#  PAPI_INCLUDE_DIRS - where to find papi.h, etc.
#  PAPI_LIBRARIES    - List of libraries when using PAPI.
#  PAPI_FOUND        - True if PAPI found.

find_path(PAPI_INCLUDE_DIR
    NAMES papi.h
    PATHS ${CMAKE_SOURCE_DIR}/lib/papi/include ${CMAKE_PREFIX_PATH}
    PATH_SUFFIXES include
    DOC "Path to PAPI include directory"
)

find_library(PAPI_LIBRARY
    NAMES papi
    PATHS ${CMAKE_SOURCE_DIR}/lib/papi/lib ${CMAKE_PREFIX_PATH}
    PATH_SUFFIXES lib lib64
    DOC "Path to PAPI library"
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Papi
    REQUIRED_VARS PAPI_LIBRARY PAPI_INCLUDE_DIR
)

if(PAPI_FOUND)
    set(PAPI_LIBRARIES ${PAPI_LIBRARY})
    set(PAPI_INCLUDE_DIRS ${PAPI_INCLUDE_DIR})
    
    # Create an imported target for modern CMake usage
    if(NOT TARGET Papi::Papi)
        add_library(Papi::Papi UNKNOWN IMPORTED)
        set_target_properties(Papi::Papi PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${PAPI_INCLUDE_DIRS}"
            IMPORTED_LOCATION "${PAPI_LIBRARIES}"
        )
    endif()
endif()

mark_as_advanced(PAPI_INCLUDE_DIR PAPI_LIBRARY)