# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

# Exports target SEAL::seal
#
# Creates variables:
#   SEAL_BUILD_TYPE : The build configuration used
#   SEAL_DEBUG : Set to non-zero value if library is compiled with extra debugging code (very slow!)
#   SEAL_LIB_BUILD_TYPE : Set to either "Static", "Static_PIC", or "Shared" depending on library build type
#   SEAL_USE_CXX17 : Set to non-zero value if library is compiled as C++17 instead of C++14
#   SEAL_ENFORCE_HE_STD_SECURITY : Set to non-zero value if library is compiled to enforce at least
#       a 128-bit security level based on HomomorphicEncryption.org security estimates
#   SEAL_USE_MSGSL : Set to non-zero value if library is compiled with Microsoft GSL support
#   MSGSL_INCLUDE_DIR : Holds the path to Microsoft GSL if library is compiled with Microsoft GSL support

include(CMakeFindDependencyMacro)

set(SEAL_BUILD_TYPE Release)
set(SEAL_DEBUG OFF)
set(SEAL_LIB_BUILD_TYPE Static_PIC)
set(SEAL_USE_CXX17 ON)
set(SEAL_ENFORCE_HE_STD_SECURITY )
set(SEAL_USE_MSGSL OFF)
if(SEAL_USE_MSGSL)
    set(MSGSL_INCLUDE_DIR MSGSL_INCLUDE_DIR-NOTFOUND)
endif()

set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_dependency(Threads REQUIRED)

include(${CMAKE_CURRENT_LIST_DIR}/SEALTargets.cmake)

if(NOT SEAL_FIND_QUIETLY)
    message(STATUS "Microsoft SEAL -> Version ${SEAL_VERSION} detected")
    if(SEAL_DEBUG)
        message(STATUS "Performance warning: Microsoft SEAL compiled in debug mode")
    endif()
message(STATUS "Microsoft SEAL -> Library build type: ${SEAL_LIB_BUILD_TYPE}")
endif()
