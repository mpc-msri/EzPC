add_library(SCI-LinearOT linear-ot.cpp)
target_link_libraries(SCI-LinearOT
    PUBLIC
    SCI-BuildingBlocks
    Eigen3::Eigen
)
