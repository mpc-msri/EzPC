cmake_minimum_required (VERSION 3.0)
project (MY_PROJ)
find_package(SCI REQUIRED PATHS "/home/t-anweshb/Desktop/EzPC/SCI/build/install")

macro(add_secure name)
 add_executable(${name}_secure "${name}_60_sci_OT0.cpp")
 target_link_libraries(${name}_secure SCI::SCI-OT)
 target_compile_options(${name}_secure PRIVATE -w)
endmacro()

add_secure(conv3_dense_bench1_normal)
add_secure(conv3_dense_bench2_normal)
add_secure(conv3_dense_bench3_normal)
add_secure(conv3_dense_bench4_normal)

add_secure(conv3_group_bench1_normal)
add_secure(conv3_group_bench2_normal)
add_secure(conv3_group_bench3_normal)
add_secure(conv3_group_bench4_normal)

add_secure(conv5_dense_bench1_normal)
add_secure(conv5_dense_bench2_normal)
add_secure(conv5_dense_bench3_normal)
add_secure(conv5_dense_bench4_normal)

add_secure(conv5_group_bench1_normal)
add_secure(conv5_group_bench2_normal)
add_secure(conv5_group_bench3_normal)
add_secure(conv5_group_bench4_normal)

add_secure(conv3_dense_bench1_winograd)
add_secure(conv3_dense_bench2_winograd)
add_secure(conv3_dense_bench3_winograd)
add_secure(conv3_dense_bench4_winograd)

add_secure(conv3_group_bench1_winograd)
add_secure(conv3_group_bench2_winograd)
add_secure(conv3_group_bench3_winograd)
add_secure(conv3_group_bench4_winograd)

add_secure(conv5_dense_bench1_winograd)
add_secure(conv5_dense_bench2_winograd)
add_secure(conv5_dense_bench3_winograd)
add_secure(conv5_dense_bench4_winograd)

add_secure(conv5_group_bench1_winograd)
add_secure(conv5_group_bench2_winograd)
add_secure(conv5_group_bench3_winograd)
add_secure(conv5_group_bench4_winograd)
