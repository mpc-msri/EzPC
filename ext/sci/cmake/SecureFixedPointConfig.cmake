include(CMakeFindDependencyMacro)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR})
include("${CMAKE_CURRENT_LIST_DIR}/source_of_randomness.cmake")

find_dependency(OpenSSL REQUIRED)
find_dependency(GMP REQUIRED)
find_dependency(Threads REQUIRED)
find_dependency(Eigen3 REQUIRED)

# Add the targets file
include("${CMAKE_CURRENT_LIST_DIR}/SecureFixedPointTargets.cmake")

if(USE_RANDOM_DEVICE)
    target_compile_definitions(SCI-utils INTERFACE EMP_USE_RANDOM_DEVICE=1)
endif(USE_RANDOM_DEVICE)
