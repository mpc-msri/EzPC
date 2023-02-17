set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(cmake/source_of_randomness.cmake)

set(OPENSSL_ROOT_DIR "/usr/local/opt/openssl")
find_package(OpenSSL REQUIRED)
find_package(GMP REQUIRED)

add_library(SCI-utils INTERFACE)

target_compile_options(SCI-utils
    INTERFACE
    "-pthread;-maes;-msse4.1;-mavx;-mavx2;-faligned-new;-std=c++17;-O3")
#     "-pthread;-Wall;-maes;-msse4.1;-mavx;-mavx2;-faligned-new;-std=c++17;-w;-g;-ggdb;-O0")

if(USE_RANDOM_DEVICE)
    target_compile_definitions(SCI-utils INTERFACE EMP_USE_RANDOM_DEVICE=1)
else(USE_RANDOM_DEVICE)
    target_compile_options(SCI-utils INTERFACE "-mrdseed")
endif(USE_RANDOM_DEVICE)

target_include_directories(SCI-utils
    INTERFACE ${OPENSSL_INCLUDE_DIR} ${GMP_INCLUDE_DIR}
)

target_link_libraries(SCI-utils
    INTERFACE ${OPENSSL_LIBRARIES} ${GMP_LIBRARIES}
)
