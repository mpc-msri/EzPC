add_subdirectory(utils)
add_subdirectory(OT)
add_subdirectory(Millionaire)
add_subdirectory(NonLinear)
add_subdirectory(LinearHE)

add_library(SCI-Protocols INTERFACE)

target_link_libraries(SCI-Protocols
    INTERFACE
    SCI-OT
    SCI-NonLinear
    SCI-LinearHE
)
