# This module defines the following uncached variables:
#  ANN_FOUND, if false, do not try to use ANN.
#  ANN_INCLUDE_DIR, where to find ANN include files.
#  ANN_LIBRARIES, the libraries to link against to use the ANN library.
#  ANN_LIBRARY_DIRS, the directory where the ANN library is found.

find_path(ANN_INCLUDE_DIR
        ANN.h
        HINTS ${CMAKE_SOURCE_DIR}/lib/ann_1.1.2/include/ANN)

if (ANN_INCLUDE_DIR)
    find_library(ANN_LIBRARY
            NAMES ANN
            HINTS ${CMAKE_SOURCE_DIR}/lib/ann_1.1.2/lib)
    if (ANN_LIBRARY)
        # Set uncached variables as per standard.
        set(ANN_FOUND ON)
        set(ANN_LIBRARIES ${ANN_LIBRARY})
        get_filename_component(ANN_LIBRARY_DIRS ${ANN_LIBRARY} PATH)
    endif (ANN_LIBRARY)
endif (ANN_INCLUDE_DIR)

if (ANN_FOUND)
    if (NOT ANN_FIND_QUIETLY)
        message(STATUS "FindANN: Found ANN header directory, ${ANN_INCLUDE_DIR}, and library, ${ANN_LIBRARIES}.")
    endif (NOT ANN_FIND_QUIETLY)
else (ANN_FOUND)
    if (ANN_FIND_REQUIRED)
        message(FATAL_ERROR "FindANN: Could not find ANN header and/or library.")
    endif (ANN_FIND_REQUIRED)
endif (ANN_FOUND)