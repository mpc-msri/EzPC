cmake_minimum_required (VERSION 3.0)
project (MY_PROJ)
find_package(SCI REQUIRED PATHS "/home/t-anweshb/Desktop/EzPC/SCI/build/install")

macro(add name)
 add_executable(${name} "anom_${name}_normal_60_sci_OT0.cpp")
 target_link_libraries(${name} SCI::SCI-OT)
 target_compile_options(${name} PRIVATE -w)
endmacro()

add(dense)
add(dwpw)
add(halfshuffle)
add(fullshuffle)
