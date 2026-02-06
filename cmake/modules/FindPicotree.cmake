# This module defines the following uncached variables:
#  Picotree_FOUND, if false, do not try to use PICOTREE.
#  Picotree_INCLUDE_DIR, where to find lasreader.hpp.
#  Picotree_LIBRARIES, the libraries to link against to use the PICOTREE library.
#  Picotree_LIBRARY_DIRS, the directory where the PICOTREE library is found.

find_path(PICOTREE_INCLUDE_DIR
        pico_tree/core.hpp
        HINTS ${CMAKE_SOURCE_DIR}/lib/picotree/include)

if (PICOTREE_INCLUDE_DIR)
    set(Picotree_FOUND ON)
endif (PICOTREE_INCLUDE_DIR)

if (PICOTREE_FOUND)
    if (NOT Picotree_FIND_QUIETLY)
        set(PICOTREE_INCLUDE_DIRS ${PICOTREE_INCLUDE_DIR})
    endif (NOT Picotree_FIND_QUIETLY)
endif (PICOTREE_FOUND)
