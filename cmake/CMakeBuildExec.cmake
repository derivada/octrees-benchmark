# ---  B U I L D I N G  --- #
# ------------------------- #

# Executable
add_executable(${PROJECT_NAME} ${sources})

# Linking libraries
if (TARGET OpenMP::OpenMP_CXX)
    target_link_libraries(${PROJECT_NAME}
            PRIVATE
            OpenMP::OpenMP_CXX)
endif ()

if (TARGET armadillo::armadillo)
    target_link_libraries(${PROJECT_NAME}
            PRIVATE
            armadillo::armadillo)
else ()
    target_link_libraries(${PROJECT_NAME}
            PRIVATE
            ${ARMADILLO_LIBRARIES})
endif ()

if (TARGET Eigen3::Eigen)
    target_link_libraries(${PROJECT_NAME}
            PRIVATE
            Eigen3::Eigen)
else ()
    target_link_libraries(${PROJECT_NAME}
            PRIVATE
            ${EIGEN_LIBRARIES})
endif ()

target_link_libraries(${PROJECT_NAME}
        PRIVATE
        ${LASLIB_LIBRARIES})

target_link_libraries(${PROJECT_NAME}
        PRIVATE
        ${PCL_LIBRARIES})

target_link_libraries(${PROJECT_NAME} PRIVATE ${PAPI_LIBRARY})
target_include_directories(${PROJECT_NAME} PRIVATE ${PAPI_INCLUDE_DIR})

# Añadimos linkeo de pthread y atomic para resolver errores
find_package(Threads REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE Threads::Threads atomic)

# Set Link Time Optimization (LTO)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flto=auto")

