# --- CONFIGURATION --- #
# --------------------- #

include(CheckCXXCompilerFlag)

CHECK_CXX_COMPILER_FLAG("-mavx2" COMPILER_SUPPORTS_AVX2)

if(COMPILER_SUPPORTS_AVX2)
    set(AVX2_FLAGS "-mavx2")
else()
    message(WARNING "AVX2 not supported by the compiler")
endif()


CHECK_CXX_COMPILER_FLAG("-mbmi2" COMPILER_SUPPORTS_BMI2)

if(COMPILER_SUPPORTS_BMI2)
    set(BMI2_FLAGS "-mbmi2")
else()
    message(WARNING "BMI2 not supported by the compiler")
endif()


CHECK_CXX_COMPILER_FLAG("-mavx512f" COMPILER_SUPPORTS_AVX512F)

if(COMPILER_SUPPORTS_AVX512F)
    set(AVX512_FLAGS "-mavx512f")
else()
    message(WARNING "AVX-512 not supported by the compiler")
endif()

CHECK_CXX_COMPILER_FLAG("-mavx512dq" COMPILER_SUPPORTS_AVX512DQ)

if(COMPILER_SUPPORTS_AVX512DQ)
    set(AVX512DQ_FLAGS "-mavx512dq")
else()
    message(WARNING "AVX-512DQ not supported by the compiler")
endif()

# Unir flags compatibles
set(CMAKE_CXX_FLAGS "${AVX2_FLAGS} ${BMI2_FLAGS}") # Add AVX512_FLAGS and AVX512DQ_FLAGS if supported

# Setup compiler flags
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG -w ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g ${CMAKE_CXX_FLAGS}")

# new Profile build type for tool perf
set(CMAKE_CXX_FLAGS_PROFILE "-O2 -g -fno-omit-frame-pointer ${CMAKE_CXX_FLAGS}")

# Set profile type
set(CMAKE_BUILD_TYPE "Profile" CACHE STRING "Build type")

# Asignar flags si el tipo de build es "Profile"
if(CMAKE_BUILD_TYPE STREQUAL "Profile")
    message(STATUS "Usando configuración de build: Profile")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_PROFILE}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS_PROFILE}")
endif()

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
