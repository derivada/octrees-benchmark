# ---  S O U R C E S  --- #
# ----------------------- #

# Gather all source files
file(GLOB_RECURSE sources CONFIGURE_DEPENDS src/*.cpp)

set(lib_sources ${sources})
list(REMOVE_ITEM lib_sources 
    "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/main_options.cpp"
)

# Include directories
include_directories(
        "inc/"
        "src/readers"
        )