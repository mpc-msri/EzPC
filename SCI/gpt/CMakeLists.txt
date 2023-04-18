project(GPT)
cmake_minimum_required(VERSION 3.0)
find_package(SCI REQUIRED PATHS "/home/t-anweshb/Desktop/EzPC/SCI/build/install")

macro(add_network_beacon name)
	add_executable(${name}_beacon "${name}.cpp")
	target_link_libraries(${name}_beacon SCI::SCI-Beacon)
endmacro()


add_network_beacon(gpt)
add_network_beacon(secure_gpt)