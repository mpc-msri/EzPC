add_library(SCI-GC
    emp-tool.cpp
    halfgate_gen.cpp
    halfgate_eva.cpp)
target_link_libraries(SCI-GC
    PUBLIC SCI-utils SCI-OT
)
