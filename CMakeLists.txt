cmake_minimum_required(VERSION 3.16)
project(sytorch)

if(APPLE)

    if(CMAKE_C_COMPILER_ID MATCHES "Clang\$")
        set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp")
        set(OpenMP_C_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY omp)
    endif()

    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang\$")
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY omp)
    endif()

endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-write-strings -Wno-unused-result -maes -Wno-ignored-attributes -march=native -Wno-deprecated-declarations -fopenmp")
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Threads REQUIRED)

add_subdirectory(ext/cryptoTools)
add_subdirectory(ext/sci)
add_subdirectory(ext/llama)

add_library(${PROJECT_NAME} STATIC)
target_sources(${PROJECT_NAME} 
PRIVATE
    src/sytorch/datasets/mnist.cpp
    src/sytorch/random.cpp
    src/sytorch/train.cpp
    src/sytorch/networks.cpp
    src/sytorch/backend/cleartext.cpp
)

target_include_directories(${PROJECT_NAME}
PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries (${PROJECT_NAME} Eigen3::Eigen Threads::Threads SCI-FloatML LLAMA cryptoTools)

add_executable(main 
    main.cpp
)
target_link_libraries (main ${PROJECT_NAME})

add_executable(module 
    module.cpp
)
target_link_libraries (module ${PROJECT_NAME})

add_executable(vgg16 
    vgg16.cpp
)
target_link_libraries (vgg16 ${PROJECT_NAME})

add_executable(chexpert 
    chexpert.cpp
)
target_link_libraries (chexpert ${PROJECT_NAME})

add_executable(resnet18 
    resnet18.cpp
)
target_link_libraries (resnet18 ${PROJECT_NAME})

add_executable(inference 
    inference.cpp
)
target_link_libraries (inference ${PROJECT_NAME})

add_executable(resnet18_acc 
    resnet18_acc.cpp
)
target_link_libraries (resnet18_acc ${PROJECT_NAME})

add_executable(dcftest 
    dcftest.cpp
)
target_link_libraries (dcftest ${PROJECT_NAME})

add_executable(strtest 
    strtest.cpp
)
target_link_libraries (strtest ${PROJECT_NAME})

add_executable(orcav2_exp 
    orcav2_exp.cpp
)
target_link_libraries (orcav2_exp ${PROJECT_NAME})

