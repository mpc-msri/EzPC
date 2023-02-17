cmake_minimum_required(VERSION 3.16)
project(sytorch)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-write-strings -Wno-unused-result -maes -Wno-ignored-attributes -march=native -Wno-deprecated-declarations -fopenmp")
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Threads REQUIRED)

add_subdirectory(backend/sci)
add_subdirectory(backend/minillama)

add_library(${PROJECT_NAME} STATIC)
target_sources(${PROJECT_NAME} 
PRIVATE
    src/sytorch/datasets/mnist.cpp
    src/sytorch/random.cpp
    src/sytorch/train.cpp
    src/sytorch/networks.cpp
)

target_include_directories(${PROJECT_NAME}
PRIVATE
    backend/minillama/cryptoTools 
PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries (${PROJECT_NAME} Eigen3::Eigen Threads::Threads SCI-FloatML LLAMA)

add_executable(main 
    main.cpp 
    backend/cleartext.cpp
)
target_link_libraries (main Eigen3::Eigen Threads::Threads SCI-FloatML LLAMA ${PROJECT_NAME})

target_include_directories(main 
PRIVATE 
    backend/minillama/cryptoTools 
    backend/sci/src
)
