include(CMakeFindDependencyMacro)


####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was SCIConfig.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

####################################################################################

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR})
include("${CMAKE_CURRENT_LIST_DIR}/source_of_randomness.cmake")

find_dependency(OpenSSL REQUIRED)
find_dependency(GMP REQUIRED)
find_dependency(Threads REQUIRED)

set_and_check(INSTALL_DIR "${PACKAGE_PREFIX_DIR}")
find_dependency(Eigen3 REQUIRED PATHS "/Users/deevashwer/EzPC/SCI/build/../extern/eigen/build" NO_DEFAULT_PATH)
find_dependency(SEAL REQUIRED PATHS "/Users/deevashwer/EzPC/SCI/build" NO_DEFAULT_PATH)
find_dependency(OpenMP REQUIRED)

# Add the targets file
include("${CMAKE_CURRENT_LIST_DIR}/SCITargets.cmake")
