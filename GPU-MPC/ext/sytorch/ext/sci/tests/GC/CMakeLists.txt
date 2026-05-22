macro (add_GC_test _name)
	add_executable(${_name} "test_${_name}.cpp")
    target_link_libraries(${_name} SCI-OT) 
endmacro()

add_GC_test(bit)
add_GC_test(int)
add_GC_test(msnzb)
add_GC_test(and)
