cmake_minimum_required (VERSION 3.0)

if(APPLE)
    set(CMAKE_C_COMPILER "/usr/local/bin/gcc-9")
    set(CMAKE_CXX_COMPILER "/usr/local/bin/g++-9")
endif()

project (SCI)
set(NAME "SCI")

option(BUILD_TESTS "Build tests" OFF)
message(STATUS "Option: BUILD_TESTS = ${BUILD_TESTS}")

option(BUILD_NETWORKS "Build networks" OFF)
message(STATUS "Option: BUILD_NETWORKS = ${BUILD_NETWORKS}")

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

set(CMAKE_PREFIX_PATH
    ${CMAKE_CURRENT_SOURCE_DIR}/extern/SEAL/native/src
    ${CMAKE_CURRENT_SOURCE_DIR}/extern/eigen/build)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/extern/SEAL/native/src
    ${CMAKE_CURRENT_SOURCE_DIR}/extern/eigen)

if(NOT CMAKE_BUILD_TYPE)
   set(CMAKE_BUILD_TYPE Release)
endif()
message(STATUS "Build type (CMAKE_BUILD_TYPE): ${CMAKE_BUILD_TYPE}")

set(CMAKE_CXX_FLAGS "-pthread -Wall -march=native -maes -mrdseed -faligned-new -std=c++17 -w")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -ggdb")

add_subdirectory(src)

if (BUILD_TESTS)
    add_subdirectory(tests)
endif (BUILD_TESTS)

if (BUILD_NETWORKS)
    add_subdirectory(networks)
endif (BUILD_NETWORKS)
