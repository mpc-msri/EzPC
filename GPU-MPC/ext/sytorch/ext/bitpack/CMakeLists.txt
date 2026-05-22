cmake_minimum_required(VERSION 3.16)
project(bitpack)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -maes -Wno-ignored-attributes -march=native")

add_library(${PROJECT_NAME} STATIC
    src/bitpack/bitpack.cpp
)

target_include_directories(${PROJECT_NAME}
PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

add_executable(
    bitpack_test
    tests/test.cpp
)

target_link_libraries(bitpack_test ${PROJECT_NAME})
