# Install script for directory: /home/dagrawal/EzPC/SCI/src

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/dagrawal/EzPC/SCI/build/install")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/home/dagrawal/EzPC/SCI/build/lib/libSCI-FloatML.a")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/home/dagrawal/EzPC/SCI/build/lib/libSCI-OT.a")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/home/dagrawal/EzPC/SCI/build/lib/libSCI-HE.a")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/home/dagrawal/EzPC/SCI/build/lib/libSCI-FloatingPoint.a")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/home/dagrawal/EzPC/SCI/build/lib/libSCI-BuildingBlocks.a")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/home/dagrawal/EzPC/SCI/build/lib/libSCI-LinearOT.a")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/home/dagrawal/EzPC/SCI/build/lib/libSCI-LinearHE.a")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/home/dagrawal/EzPC/SCI/build/lib/libSCI-Math.a")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/home/dagrawal/EzPC/SCI/build/lib/libSCI-GC.a")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/SCI/SCITargets.cmake")
    file(DIFFERENT EXPORT_FILE_CHANGED FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/SCI/SCITargets.cmake"
         "/home/dagrawal/EzPC/SCI/build/src/CMakeFiles/Export/lib/cmake/SCI/SCITargets.cmake")
    if(EXPORT_FILE_CHANGED)
      file(GLOB OLD_CONFIG_FILES "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/SCI/SCITargets-*.cmake")
      if(OLD_CONFIG_FILES)
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/SCI/SCITargets.cmake\" will be replaced.  Removing files [${OLD_CONFIG_FILES}].")
        file(REMOVE ${OLD_CONFIG_FILES})
      endif()
    endif()
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/SCI" TYPE FILE FILES "/home/dagrawal/EzPC/SCI/build/src/CMakeFiles/Export/lib/cmake/SCI/SCITargets.cmake")
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^()$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/SCI" TYPE FILE FILES "/home/dagrawal/EzPC/SCI/build/src/CMakeFiles/Export/lib/cmake/SCI/SCITargets-noconfig.cmake")
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES
    "/home/dagrawal/EzPC/SCI/src/utils"
    "/home/dagrawal/EzPC/SCI/src/OT"
    "/home/dagrawal/EzPC/SCI/src/GC"
    "/home/dagrawal/EzPC/SCI/src/Millionaire"
    "/home/dagrawal/EzPC/SCI/src/NonLinear"
    "/home/dagrawal/EzPC/SCI/src/BuildingBlocks"
    "/home/dagrawal/EzPC/SCI/src/LinearOT"
    "/home/dagrawal/EzPC/SCI/src/LinearHE"
    "/home/dagrawal/EzPC/SCI/src/Math"
    "/home/dagrawal/EzPC/SCI/src/FloatingPoint"
    FILES_MATCHING REGEX "/[^/]*\\.h$" REGEX "/[^/]*\\.hpp$")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES
    "/home/dagrawal/EzPC/SCI/src/defines.h"
    "/home/dagrawal/EzPC/SCI/src/defines_uniform.h"
    "/home/dagrawal/EzPC/SCI/src/defines_float.h"
    "/home/dagrawal/EzPC/SCI/src/globals.h"
    "/home/dagrawal/EzPC/SCI/src/globals_float.h"
    "/home/dagrawal/EzPC/SCI/src/library_fixed.h"
    "/home/dagrawal/EzPC/SCI/src/library_fixed_uniform.h"
    "/home/dagrawal/EzPC/SCI/src/library_float.h"
    "/home/dagrawal/EzPC/SCI/src/cleartext_library_fixed.h"
    "/home/dagrawal/EzPC/SCI/src/cleartext_library_fixed_uniform.h"
    "/home/dagrawal/EzPC/SCI/src/cleartext_library_float.h"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/SCI" TYPE FILE FILES
    "/home/dagrawal/EzPC/SCI/src/utils/cmake/FindGMP.cmake"
    "/home/dagrawal/EzPC/SCI/src/utils/cmake/source_of_randomness.cmake"
    )
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/home/dagrawal/EzPC/SCI/build/src/utils/cmake_install.cmake")
  include("/home/dagrawal/EzPC/SCI/build/src/OT/cmake_install.cmake")
  include("/home/dagrawal/EzPC/SCI/build/src/GC/cmake_install.cmake")
  include("/home/dagrawal/EzPC/SCI/build/src/Millionaire/cmake_install.cmake")
  include("/home/dagrawal/EzPC/SCI/build/src/BuildingBlocks/cmake_install.cmake")
  include("/home/dagrawal/EzPC/SCI/build/src/LinearOT/cmake_install.cmake")
  include("/home/dagrawal/EzPC/SCI/build/src/LinearHE/cmake_install.cmake")
  include("/home/dagrawal/EzPC/SCI/build/src/NonLinear/cmake_install.cmake")
  include("/home/dagrawal/EzPC/SCI/build/src/Math/cmake_install.cmake")
  include("/home/dagrawal/EzPC/SCI/build/src/FloatingPoint/cmake_install.cmake")

endif()

