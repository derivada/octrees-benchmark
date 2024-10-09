# This module defines the following uncached variables:
#  ARMADILLO_FOUND, if false, do not try to use armadillo.
#  ARMADILLO_INCLUDE_DIR, where to find armadillo.h.
#  ARMADILLO_LIBRARIES, the libraries to link against to use the armadillo
#  library.
#  ARMADILLO_LIBRARY_DIRS, the directory where the armadillo library is found.

find_path(ARMADILLO_INCLUDE_DIR
        armadillo
        HINTS ${CMAKE_SOURCE_DIR}/lib/armadillo/include/)

if(ARMADILLO_INCLUDE_DIR)
    find_library(ARMADILLO_LIBRARY
            libarmadillo.so
            HINTS ${CMAKE_SOURCE_DIR}/lib/armadillo/
    )
    if(ARMADILLO_LIBRARY)
        # Set uncached variables as per standard.
        set(ARMADILLO_FOUND ON)
        set(ARMADILLO_LIBRARIES ${ARMADILLO_LIBRARY})
        get_filename_component(ARMADILLO_LIBRARY_DIRS ${ARMADILLO_LIBRARY} PATH)
    endif(ARMADILLO_LIBRARY)
endif(ARMADILLO_INCLUDE_DIR)

if(ARMADILLO_FOUND)
    if(NOT ARMADILLO_FIND_QUIETLY)
        message(STATUS "Findarmadillolib: Found armadillo header directory, ${ARMADILLO_INCLUDE_DIR}, and library, ${ARMADILLO_LIBRARIES}.")
    endif(NOT ARMADILLO_FIND_QUIETLY)
else(ARMADILLO_FOUND)
    if(ARMADILLO_FIND_REQUIRED)
        message(FATAL_ERROR "Findarmadillolib: Could not find armadillo header and/or library.")
    endif(ARMADILLO_FIND_REQUIRED)
endif(ARMADILLO_FOUND)