# --- CONFIGURATION --- #
# --------------------- #

# Setup compiler flags
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG -w")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")

# CXX Standard
set(CMAKE_CXX_STANDARD 20)

# MISC Flags
#set(CMAKE_CXX_FLAGS "-lstdc++fs")

# Enable LTO (Link Time Optimization)
include(CheckIPOSupported)

# Optional IPO. Do not use IPO if it's not supported by compiler.
check_ipo_supported(RESULT supported OUTPUT error)
if (supported)
    message(STATUS "IPO is supported: ${supported}")
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
else ()
    message(WARNING "IPO is not supported: <${error}>")
endif ()