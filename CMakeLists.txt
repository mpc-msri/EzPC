cmake_minimum_required(VERSION 3.16)
project(fptraining)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-write-strings -Wno-unused-result")
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
add_executable(fptraining main.cpp)
target_link_libraries (fptraining Eigen3::Eigen)

