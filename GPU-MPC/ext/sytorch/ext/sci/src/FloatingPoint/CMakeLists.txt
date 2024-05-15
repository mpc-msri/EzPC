add_library(SCI-FloatingPoint fp-math.cpp floating-point.cpp fixed-point.cpp bool-data.cpp)
target_link_libraries(SCI-FloatingPoint
    PUBLIC SCI-BuildingBlocks SCI-Math
)
