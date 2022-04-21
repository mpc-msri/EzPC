find_package(OpenMP REQUIRED)

add_library(SCI-BuildingBlocks
    value-extension.cpp
    aux-protocols.cpp
    truncation.cpp)

target_link_libraries(SCI-BuildingBlocks
    PUBLIC SCI-OTPrimitive SCI-GC SCI-Millionaire SCI-LinearOT OpenMP::OpenMP_CXX
)
