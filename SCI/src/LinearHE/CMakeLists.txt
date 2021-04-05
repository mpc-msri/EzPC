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
    execute_process(COMMAND ${CMAKE_COMMAND} -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX} .
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/extern/SEAL/native/src")
    execute_process(COMMAND ${CMAKE_COMMAND} --build . --target install
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/extern/SEAL/native/src")
    find_package(SEAL 3.3.2 EXACT REQUIRED PATHS "${CMAKE_INSTALL_PREFIX}")
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
