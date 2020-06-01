cmake_minimum_required (VERSION 3.0)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if(APPLE)
    set(CMAKE_C_COMPILER "/usr/local/bin/gcc-9")
    set(CMAKE_CXX_COMPILER "/usr/local/bin/g++-9")
endif()

project(Porthos)
set(NAME "Porthos")

find_package(OpenMP REQUIRED)
find_package(OpenSSL REQUIRED)

FILE(GLOB CXXSources *.cpp)
list(REMOVE_ITEM CXXSources ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
add_library(Porthos-Protocols ${CXXSources})

target_include_directories(Porthos-Protocols 
	PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../lib_eigen/
	${OPENSSL_INCLUDE_DIR})

target_link_libraries(Porthos-Protocols
	PUBLIC OpenMP::OpenMP_CXX
	${OPENSSL_LIBRARIES})

add_executable(Porthos3PC ${CMAKE_CURRENT_SOURCE_DIR}/example_neural_nets/mainResNet50.cpp)
target_link_libraries(Porthos3PC Porthos-Protocols)

target_compile_options(Porthos-Protocols
	PUBLIC "-pthread;-Wall;-march=native;-msse4.1;-maes;-mpclmul;-mrdseed;-fpermissive;-fpic;-std=c++17;-w;-O3")

target_compile_options(Porthos3PC
	PRIVATE "-pthread;-Wall;-march=native;-msse4.1;-maes;-mpclmul;-mrdseed;-fpermissive;-fpic;-std=c++17;-w;-O3")


