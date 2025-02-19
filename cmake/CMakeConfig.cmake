# --- CONFIGURATION --- #
# --------------------- #

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-mavx2" COMPILER_SUPPORTS_AVX2)

if(COMPILER_SUPPORTS_AVX2)
    set(AVX2_FLAGS "-mavx2")
else()
    message(WARNING "AVX2 not supported by the compiler")
endif()

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-mbmi2" COMPILER_SUPPORTS_BMI2)

if(COMPILER_SUPPORTS_BMI2)
    set(BMI2_FLAGS "-mbmi2")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${BMI2_FLAGS}")
else()
    message(WARNING "BMI2 not supported by the compiler")
endif()


# Setup compiler flags
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG -w ${AVX2_FLAGS} ${BMI2_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g ${AVX2_FLAGS} ${BMI2_FLAGS}")

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