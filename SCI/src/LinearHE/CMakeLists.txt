find_package(OpenMP REQUIRED)
# set(CMAKE_FIND_DEBUG_MODE 1)

find_package(SEAL 3.3.2 EXACT QUIET)
if (NOT SEAL_FOUND)
    message(STATUS "SEAL 3.3.2 was not found: clone and install SEAL locally")
    if (NOT EXISTS "${PROJECT_SOURCE_DIR}/extern/SEAL/native/src/CMakeLists.txt")
        find_package(Git REQUIRED)
        message(STATUS "initialize Git submodule: extern/SEAL")
        execute_process(COMMAND git submodule update --init --recursive extern/SEAL
                WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
    endif ()
if(APPLE)
    execute_process(COMMAND ${CMAKE_COMMAND} -DCMAKE_INSTALL_PREFIX=${PROJECT_SOURCE_DIR}/build .
        -DCMAKE_C_COMPILER=${MAC_GCC} -DCMAKE_CXX_COMPILER=${MAC_GPP}
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/extern/SEAL/native/src")
else ()
    execute_process(COMMAND ${CMAKE_COMMAND} -DCMAKE_INSTALL_PREFIX=${PROJECT_SOURCE_DIR}/build .
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/extern/SEAL/native/src")
endif()
    execute_process(COMMAND ${CMAKE_COMMAND} --build . --target install
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/extern/SEAL/native/src")
    find_package(SEAL 3.3.2 EXACT REQUIRED PATHS "${PROJECT_SOURCE_DIR}/build/")
endif ()

find_package(Eigen3 3.3 NO_MODULE QUIET)
if (NOT Eigen3_FOUND)
    message(STATUS "Eigen 3.3 was not found: clone and install Eigen3 locally")
    if (NOT EXISTS "${PROJECT_SOURCE_DIR}/extern/eigen/CMakeLists.txt")
        find_package(Git REQUIRED)
        message(STATUS "initialize Git submodule: extern/eigen")
        execute_process(COMMAND git submodule update --init --recursive extern/eigen
                WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
    endif ()
    execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory build
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/extern/eigen/")
    execute_process(COMMAND ${CMAKE_COMMAND} -DCMAKE_INSTALL_PREFIX=${PROJECT_SOURCE_DIR}/build ..
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/extern/eigen/build")
    execute_process(COMMAND ${CMAKE_COMMAND} --build .. --target install
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/extern/eigen/build")
    message(STATUS "${PROJECT_SOURCE_DIR}")
    find_package(Eigen3 3.3 REQUIRED NO_MODULE PATHS "${PROJECT_SOURCE_DIR}/build/")
endif ()

add_library(SCI-LinearHE
    conv-field.cpp
    fc-field.cpp
    elemwise-prod-field.cpp
    utils-HE.cpp
)

target_link_libraries(SCI-LinearHE
    PUBLIC
    SCI-utils
    SEAL::seal
    OpenMP::OpenMP_CXX
    Eigen3::Eigen
)

if(!APPLE)
    target_compile_options(SCI-LinearHE PRIVATE "-fopenmp")
endif()
target_compile_options(SCI-LinearHE PUBLIC "-fpermissive")
