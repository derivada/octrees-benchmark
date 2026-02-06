# ---  L I B R A R I E S  --- #
# --------------------------- #

# Add module directory to the include path.
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${PROJECT_SOURCE_DIR}/cmake/modules")

# Add lib/ folder to the list of folder where CMake looks for packages
set(LIB_FOLDER "${CMAKE_SOURCE_DIR}/lib")
set(LOCAL_MODULE_PATH "$ENV{HOME}/local")

list(APPEND CMAKE_MODULE_PATH ${LIB_FOLDER})
list(APPEND CMAKE_MODULE_PATH ${LOCAL_MODULE_PATH})



# OpenMP
find_package(OpenMP REQUIRED)
if (OPENMP_CXX_FOUND)
    message(STATUS "OpenMP found and to be linked")
else ()
    message(SEND_ERROR "Could not find OpenMP")
endif ()

# Eigen3
find_package(Eigen3 REQUIRED)
if (TARGET Eigen3::Eigen)
    message(STATUS "Dependency Eigen3::Eigen found")
elseif (${EIGEN3_FOUND})
    include_directories(${EIGEN3_INCLUDE_DIR})
    message(STATUS "Eigen include: ${EIGEN3_INCLUDE_DIR}")
else ()
    message(SEND_ERROR "Could find Eigen3")
endif ()

# LASlib
find_package(LASLIB REQUIRED)
if (${LASLIB_FOUND})
  include_directories(${LASLIB_INCLUDE_DIR} ${LASZIP_INCLUDE_DIR})
    message(STATUS "LASlib include: ${LASLIB_INCLUDE_DIR} ${LASZIP_INCLUDE_DIR}")
else ()
    message(SEND_ERROR "Could not find LASLIB")
endif ()


# Hint Boost so PCL's own config can find the locally built Boost.
set(BOOST_ROOT "${PROJECT_SOURCE_DIR}/lib/boost")
set(Boost_ROOT "${PROJECT_SOURCE_DIR}/lib/boost")
set(Boost_NO_SYSTEM_PATHS ON)

# Include Boost
set(BOOST_INCLUDE_DIRS "${BOOST_ROOT}/include")
include_directories(${BOOST_INCLUDE_DIRS})

# PCL (PointCloudLibrary)
set(PCL_DIR "${CMAKE_SOURCE_DIR}/lib/pcl")
message("PCL directory: ${PCL_DIR}")
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
if(EXISTS ${PCL_DIR})
    # Include from from lib directory
    message("Loading PCL from \"${PCL_DIR}\".")
    set(PCL_INCLUDE_DIRS "${PCL_DIR}/include/pcl-1.15/")
    file(GLOB PCL_LIBRARIES "${PCL_DIR}/lib/*.so")
    add_definitions(-DHAVE_PCL)
else()
    message("Loading PCL from system.")
    find_package(PCL 1.15 REQUIRED)
endif()
message("PCL include: ${PCL_INCLUDE_DIRS}")
message("PCL libraries: ${PCL_LIBRARIES}")
include_directories(${PCL_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})


# PAPI
find_package(Papi REQUIRED)
if (${PAPI_FOUND})
    include_directories(${PAPI_INCLUDE_DIRS})
    message(STATUS "Papi include: ${PAPI_INCLUDE_DIRS}")
    message(STATUS "Papi libraries: ${PAPI_LIBRARIES}")
else ()
    message(SEND_ERROR "Could not find Papi")
endif ()

# Picotree
find_package(Picotree)
if (${Picotree_FOUND})
    include_directories(${PICOTREE_INCLUDE_DIRS})
    message(STATUS "Picotree include: ${PICOTREE_INCLUDE_DIRS}")
else ()
    message(WARNING "Could not find Picotree. Building without Picotree support.")
endif ()