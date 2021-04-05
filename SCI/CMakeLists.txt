cmake_minimum_required (VERSION 3.13)

project (SCI)
set(NAME "SCI")

option(BUILD_TESTS "Build tests" OFF)
message(STATUS "Option: BUILD_TESTS = ${BUILD_TESTS}")

option(BUILD_NETWORKS "Build networks" OFF)
message(STATUS "Option: BUILD_NETWORKS = ${BUILD_NETWORKS}")

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(src)

if (BUILD_TESTS)
    add_subdirectory(tests)
endif()

if (BUILD_NETWORKS)
    add_subdirectory(networks)
endif()

set(INSTALL_DIR "${CMAKE_INSTALL_PREFIX}")
include(CMakePackageConfigHelpers)
configure_package_config_file(
	"${CMAKE_CURRENT_SOURCE_DIR}/cmake/SCIConfig.cmake.in"
	"${CMAKE_CURRENT_SOURCE_DIR}/cmake/SCIConfig.cmake"
	INSTALL_DESTINATION lib/cmake/SCI
	PATH_VARS INSTALL_DIR)
install(FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/SCIConfig.cmake"
        DESTINATION lib/cmake/SCI)
