set(CMAKE_FIND_DEBUG_MODE 1)
include_directories(${CMAKE_CURRENT_LIST_DIR})

find_package(Eigen3 3.3 QUIET NO_MODULE PATHS "${PROJECT_SOURCE_DIR}/../SCI/extern/eigen/build" NO_DEFAULT_PATH)
if (NOT Eigen3_FOUND)
    message(STATUS "Eigen 3.3 was not found: clone and install Eigen3 locally")
    if (NOT EXISTS "${PROJECT_SOURCE_DIR}/../SCI/extern/eigen/CMakeLists.txt")
        find_package(Git REQUIRED)
        message(STATUS "initialize Git submodule: extern/eigen")
        execute_process(COMMAND git submodule update --init --recursive ../SCI/extern/eigen
                WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
    endif ()
    execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory build
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/../SCI/extern/eigen/")
    execute_process(COMMAND ${CMAKE_COMMAND} -DCMAKE_INSTALL_PREFIX=${PROJECT_SOURCE_DIR}/build ..
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/../SCI/extern/eigen/build")
    find_package(Eigen3 3.3 REQUIRED NO_MODULE PATHS "${PROJECT_SOURCE_DIR}/../SCI/extern/eigen/build" NO_DEFAULT_PATH)
endif()

add_library(fss STATIC 
    api.cpp
    api_varied.cpp
    lib.cpp
    deps/cryptoTools/cryptoTools/Common/Defines.cpp
    deps/cryptoTools/cryptoTools/Common/Log.cpp
    deps/cryptoTools/cryptoTools/Crypto/AES.cpp
    deps/cryptoTools/cryptoTools/Crypto/PRNG.cpp
    group_element.cpp
    comms.cpp
    dcf.cpp
    utils.cpp
    add.cpp
    mult.cpp 
    conv.cpp
    pubdiv.cpp
    spline.cpp
    mini_aes.cpp
    input_prng.cpp
)

find_package(Threads REQUIRED)

target_link_libraries(fss
    PRIVATE
    Threads::Threads
    Eigen3::Eigen
)

target_include_directories(fss
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:./>)

target_include_directories(fss PRIVATE deps/cryptoTools)

target_compile_options(fss PRIVATE -std=c++17 -march=native -Wno-parentheses-equality -Wno-ignored-attributes -maes -msse2 -msse4.1 -mpclmul -fPIC -no-pie)

install(
    TARGETS fss
    EXPORT FSSTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

install(EXPORT FSSTargets
        FILE FSSTargets.cmake
        DESTINATION lib/cmake/FSS)

install(FILES
        "utils.h"
        "fss.h"
        "dcf.h"
        "group_element.h"
        "ArgMapping.h"
        "comms.h"
        "api.h"
        "api_varied.h"
        "array.h"
        "lib.h"
        "keypack.h"
        "conv.h"
        "spline.h"
        "add.h"
        "mult.h"
        "pubdiv.h"
        "mini_aes.h"
        "input_prng.h"
        "config.h"
        DESTINATION include)

install(DIRECTORY
    "deps/cryptoTools/cryptoTools"
    DESTINATION include)
        
        