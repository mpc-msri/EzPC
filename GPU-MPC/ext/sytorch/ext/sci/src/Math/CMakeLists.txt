add_library(SCI-Math math-functions.cpp)
target_link_libraries(SCI-Math
    PUBLIC SCI-LinearOT SCI-BuildingBlocks
)
