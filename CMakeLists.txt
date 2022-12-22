cmake_minimum_required(VERSION 3.16)
project(fptraining)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-write-strings -Wno-unused-result -maes -Wno-ignored-attributes -march=native -Wno-deprecated-declarations")
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Threads REQUIRED)

add_subdirectory(backend/sci)
add_subdirectory(backend/minillama)

add_executable(fptraining 
    main.cpp 
    networks.cpp
    mnist.cpp
    random.cpp
    train.cpp
    backend/cleartext.cpp
)
target_link_libraries (fptraining Eigen3::Eigen Threads::Threads SCI-FloatML LLAMA)
target_include_directories(fptraining PRIVATE backend/minillama/cryptoTools backend/sci/src)

add_executable(tests 
    tests/main.cpp
    tests/cleartext.cpp
    tests/llama.cpp
    networks.cpp
    mnist.cpp
    random.cpp
    train.cpp
    backend/cleartext.cpp
)
target_link_libraries (tests Eigen3::Eigen Threads::Threads SCI-FloatML LLAMA)
target_include_directories(tests PRIVATE backend/minillama/cryptoTools backend/sci/src)
