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

# # Armadillo
# find_package(Armadillo REQUIRED)
# if (TARGET armadillo::armadillo)
#     message(STATUS "Dependency armadillo::armadillo found")
# elseif (${ARMADILLO_FOUND})
#     include_directories(${ARMADILLO_INCLUDE_DIR})
#     message(STATUS "Armadillo include: " ${ARMADILLO_INCLUDE_DIR})
#     message(STATUS "Armadillo libraries: " ${ARMADILLO_LIBRARIES})
# else ()
#     message(SEND_ERROR "Could find armadillo::armadillo")
# endif ()

# # Eigen3
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

# PCL
set(PCL_DIR "$ENV{HOME}/local/pcl/share/pcl-1.15" CACHE PATH "Path to PCL config")
find_package(PCL 1.3 QUIET)

if(PCL_FOUND)
    message(STATUS "PCL found: ${PCL_VERSION}")
    include_directories(${PCL_INCLUDE_DIRS})
    link_directories(${PCL_LIBRARY_DIRS})
    add_definitions(${PCL_DEFINITIONS})
    add_definitions(-DHAVE_PCL)
else()
    message(WARNING "PCL not found. Building without PCL support.")
endif()


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