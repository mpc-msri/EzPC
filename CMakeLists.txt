cmake_minimum_required(VERSION 3.16)
project(fptraining)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-write-strings -Wno-unused-result -maes -Wno-ignored-attributes -march=native -Wno-deprecated-declarations -fopenmp")
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

add_executable(gpu 
    # gpu_main.cpp 
    # chemistry.cpp
    # backend/minillama/gpu/fc_test.cpp
    # generated/fc_inference.cpp
    # generated/fc_iteration.cpp
    # generated/chemistry.cpp
    # gpu_reference_code/iteration.cpp
    generated/3layer.cpp
    networks.cpp
    mnist.cpp
    random.cpp
    train.cpp
    backend/cleartext.cpp
)
target_link_libraries (gpu Eigen3::Eigen Threads::Threads SCI-FloatML LLAMA LLAMA-GPU cudart)
target_include_directories(gpu PRIVATE backend/minillama/cryptoTools backend/sci/src)

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

add_custom_target(execs)
add_dependencies(execs fptraining gpu)