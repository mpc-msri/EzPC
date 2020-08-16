set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

macro (add_network _name)
	add_executable(${_name} "main_${_name}.cpp")
    target_link_libraries(${_name} SCI-Protocols) 
endmacro()

# Networks
add_network(densenet121_field64)
add_network(densenet121_ring32)
add_network(resnet50_field64)
add_network(resnet50_ring32)
add_network(sqnet_ring32)
add_network(sqnet_field64)
add_network(minionn_ring32)
add_network(minionn_field64)
add_network(resnet32_cifar_field64)
