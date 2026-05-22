if (NOT EXISTS "${PROJECT_SOURCE_DIR}/extern/EMP/emp-tool/CMakeLists.txt")
    find_package(Git REQUIRED)
    message(STATUS "initialize Git submodule: extern/EMP/emp-tool")
    execute_process(COMMAND git submodule update --init --recursive extern/EMP/emp-tool
            WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
endif ()
execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory build
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/extern/EMP/emp-tool")
execute_process(COMMAND ${CMAKE_COMMAND} -DTHREADING=ON
    -DCMAKE_INSTALL_PREFIX=${PROJECT_SOURCE_DIR}/build ..
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/extern/EMP/emp-tool/build")
execute_process(COMMAND make install
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/extern/EMP/emp-tool/build")
find_package(emp-tool REQUIRED PATHS "${PROJECT_SOURCE_DIR}/build/" NO_DEFAULT_PATH)
message(STATUS "emp-tool installed: ${emp-tool_FOUND}")

if (NOT EXISTS "${PROJECT_SOURCE_DIR}/extern/EMP/emp-ot/CMakeLists.txt")
    find_package(Git REQUIRED)
    message(STATUS "initialize Git submodule: extern/EMP/emp-ot")
    execute_process(COMMAND git submodule update --init --recursive extern/EMP/emp-ot
            WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
endif ()
execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory build
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/extern/EMP/emp-ot")
execute_process(COMMAND ${CMAKE_COMMAND} -DTHREADING=ON
    -DCMAKE_INSTALL_PREFIX=${PROJECT_SOURCE_DIR}/build ..
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/extern/EMP/emp-ot/build")
execute_process(COMMAND make install
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/extern/EMP/emp-ot/build")
message(STATUS "${PROJECT_SOURCE_DIR}")
find_package(emp-ot REQUIRED PATHS "${PROJECT_SOURCE_DIR}/build/" NO_DEFAULT_PATH)
message(STATUS "emp-ot installed: ${emp-ot_FOUND}")

if (NOT EXISTS "${PROJECT_SOURCE_DIR}/extern/EMP/emp-sh2pc/CMakeLists.txt")
    find_package(Git REQUIRED)
    message(STATUS "initialize Git submodule: extern/EMP/emp-sh2pc")
    execute_process(COMMAND git submodule update --init --recursive extern/EMP/emp-sh2pc
            WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
endif ()
execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory build
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/extern/EMP/emp-sh2pc")
execute_process(COMMAND ${CMAKE_COMMAND} -DTHREADING=ON
    -DCMAKE_INSTALL_PREFIX=${PROJECT_SOURCE_DIR}/build ..
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/extern/EMP/emp-sh2pc/build")
execute_process(COMMAND make install
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/extern/EMP/emp-sh2pc/build")
message(STATUS "${PROJECT_SOURCE_DIR}")
find_package(emp-sh2pc REQUIRED PATHS "${PROJECT_SOURCE_DIR}/build/" NO_DEFAULT_PATH)
message(STATUS "emp-sh2pc installed: ${emp-sh2pc_FOUND}")
