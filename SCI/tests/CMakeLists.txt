set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

macro (add_test _name)
	add_executable(${_name} "test_${_name}.cpp")
    target_link_libraries(${_name} SCI-Protocols) 
endmacro()

# Tests
add_test(HE_conv)
add_test(HE_fc)
add_test(HE_elemwise_prod)
add_test(ring_relu)
add_test(argmax)
add_test(ring_argmax)
add_test(field_maxpool)
add_test(ring_maxpool)
add_test(linear)
add_test(functionalities)
