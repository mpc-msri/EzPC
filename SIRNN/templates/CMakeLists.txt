cmake_minimum_required (VERSION 3.0)

# Set paths to GNU gcc and g++ on Mac OS X
set(MAC_GCC "/usr/local/bin/gcc-9")
set(MAC_GPP "/usr/local/bin/g++-9")

if(APPLE)
    set(CMAKE_C_COMPILER ${MAC_GCC})
    set(CMAKE_CXX_COMPILER ${MAC_GPP})
endif()

project (FastGRNN-Inference)
set(NAME "FastGRNN-Inference")

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(CMAKE_CXX_FLAGS "-O3")

find_package(SCI REQUIRED PATHS SCI_BUILD_LOCATION)
