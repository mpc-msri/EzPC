set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

macro (add_network_OT _name)
	add_executable(${_name}-OT "main_${_name}.cpp")
    target_link_libraries(${_name}-OT SCI-OT) 
    target_compile_options(${_name}-OT PUBLIC "-Wno-parentheses-equality")
endmacro()

macro (add_network_HE _name)
    add_executable(${_name}-HE "main_${_name}.cpp")
    target_link_libraries(${_name}-HE SCI-HE) 
    target_compile_options(${_name}-HE PUBLIC "-Wno-parentheses-equality")
endmacro()

# Networks
add_network_HE(sqnet)
add_network_OT(sqnet)
add_network_HE(resnet50)
add_network_OT(resnet50)
add_network_HE(densenet121)
add_network_OT(densenet121)
add_network_HE(minionn)
add_network_OT(minionn)
add_network_HE(resnet32_cifar)
add_network_OT(resnet32_cifar)

# Beacon benchmarks

macro(add_network_secfloat name)
	add_executable(${name}_secfloat "main_${name}.cpp")
	target_link_libraries(${name}_secfloat SCI-SecfloatML)
endmacro()

macro(add_network_beacon name)
	add_executable(${name}_beacon "main_${name}.cpp")
	target_link_libraries(${name}_beacon SCI-Beacon)
endmacro()

macro(add_network name)
	add_network_secfloat(${name})
	add_network_beacon(${name})
endmacro()

add_network(logistic128)
add_network(ffnn128)
add_network(relevance32)
add_network(lenet128)
add_network(hinet4)