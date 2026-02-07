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

# Picotree
find_package(Picotree OPTIONAL)
if (${PICOTREE_FOUND})
    include_directories(${PICOTREE_INCLUDE_DIRS})
    message(STATUS "Picotree include: ${PICOTREE_INCLUDE_DIRS}")
else ()
    message(STATUS "Could not find Picotree, building without Picotree support")
endif ()


# Hint Boost so PCL's own config can find the locally built Boost.
set(BOOST_ROOT "${PROJECT_SOURCE_DIR}/lib/boost")
set(Boost_ROOT "${PROJECT_SOURCE_DIR}/lib/boost")
set(Boost_NO_SYSTEM_PATHS ON)

# Include Boost
set(BOOST_INCLUDE_DIRS "${BOOST_ROOT}/include")
include_directories(${BOOST_INCLUDE_DIRS})

# PCL (PointCloudLibrary)
find_package(PCL OPTIONAL)
if (${PCL_FOUND})
    include_directories(${PCL_INCLUDE_DIRS})
    message(STATUS "PCL include: ${PCL_INCLUDE_DIRS}")
    message(STATUS "PCL libraries: ${PCL_LIBRARIES}")
else ()
    message(STATUS "Could not find PCL, building without PCL support")
endif ()

# PAPI
find_package(Papi REQUIRED)
if (${PAPI_FOUND})
    include_directories(${PAPI_INCLUDE_DIRS})
    message(STATUS "Papi include: ${PAPI_INCLUDE_DIRS}")
    message(STATUS "Papi libraries: ${PAPI_LIBRARIES}")
else ()
    message(SEND_ERROR "Could not find Papi")
endif ()