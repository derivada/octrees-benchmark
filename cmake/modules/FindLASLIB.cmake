# This module defines the following uncached variables:
#  LASLIB_FOUND, if false, do not try to use LASlib.
#  LASLIB_INCLUDE_DIR, where to find lasreader.hpp.
#  LASLIB_LIBRARIES, the libraries to link against to use the LASlib library.
#  LASLIB_LIBRARY_DIRS, the directory where the LASlib library is found.

find_path(LASLIB_INCLUDE_DIR
        lasreader.hpp
        HINTS ${CMAKE_SOURCE_DIR}/lib/LAStools/LASlib/inc)
find_path(LASZIP_INCLUDE_DIR
        mydefs.hpp
        HINTS ${CMAKE_SOURCE_DIR}/lib/LAStools/LASzip/src)

find_path(LASZIP_INCLUDE_DIR_2
        laszip_common.h
        HINTS ${CMAKE_SOURCE_DIR}/lib/LAStools/LASzip/include/laszip)

if (LASLIB_INCLUDE_DIR)
    find_library(LASLIB_LIBRARY
            NAMES LASlib
            HINTS ${CMAKE_SOURCE_DIR}/lib/LAStools/LASlib/lib)
    if (LASLIB_LIBRARY)
        # Set uncached variables as per standard.
        set(LASLIB_FOUND ON)
        set(LASLIB_LIBRARIES ${LASLIB_LIBRARY})
        get_filename_component(LASLIB_LIBRARY_DIRS ${LASLIB_LIBRARY} PATH)
    endif (LASLIB_LIBRARY)
endif (LASLIB_INCLUDE_DIR)

if (LASLIB_FOUND)
    if (NOT LASLIB_FIND_QUIETLY)
        message(STATUS "FindLASLIB: Found LASLIB header directory, ${LASLIB_INCLUDE_DIR}, and library, ${LASLIB_LIBRARIES}.")
    endif (NOT LASLIB_FIND_QUIETLY)
else (LASLIB_FOUND)
    if (LASLIB_FIND_REQUIRED)
        message(FATAL_ERROR "FindLASLIB: Could not find LASLIB header and/or library.")
    endif (LASLIB_FIND_REQUIRED)
endif (LASLIB_FOUND)