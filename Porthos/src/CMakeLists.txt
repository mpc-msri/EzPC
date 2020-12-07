cmake_minimum_required (VERSION 3.0)

if(APPLE)
		set(CMAKE_C_COMPILER "/usr/local/bin/gcc-9")
		set(CMAKE_CXX_COMPILER "/usr/local/bin/g++-9")
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

project(Porthos)
set(NAME "Porthos")

find_package(OpenMP REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(Boost 1.65 REQUIRED)

FILE(GLOB CXXSources *.cpp)
list(REMOVE_ITEM CXXSources ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
add_library(Porthos-Protocols ${CXXSources})

target_include_directories(Porthos-Protocols 
	PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../lib_eigen/
	${OPENSSL_INCLUDE_DIR})

target_link_libraries(Porthos-Protocols
	PUBLIC OpenMP::OpenMP_CXX
	${OPENSSL_LIBRARIES})

macro (add_network _name)
	add_executable(${_name} ${CMAKE_CURRENT_SOURCE_DIR}/example_neural_nets/main${_name}.cpp)
	target_link_libraries(${_name} Porthos-Protocols -lrt -lboost_system)
endmacro()

target_compile_options(Porthos-Protocols
	PUBLIC "-pthread;-Wall;-march=native;-msse4.1;-maes;-mpclmul;-mrdseed;-fpermissive;-fpic;-std=c++17;-w;-O3;-fopenmp;")

add_network(DenseNet121)
add_network(ResNet50)
add_network(SqNetImgNet)
