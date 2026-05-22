set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(cmake/source_of_randomness.cmake)

set(OPENSSL_ROOT_DIR "/usr/local/opt/openssl")
find_package(OpenSSL REQUIRED)
find_package(GMP REQUIRED)

add_library(SCI-utils INTERFACE)

target_compile_options(SCI-utils
    INTERFACE
    "-pthread;-faligned-new;-std=c++17;-O3;-Wno-parentheses-equality;-Wno-deprecated-declarations;-Wno-macro-redefined")
#     "-pthread;-Wall;-maes;-msse4.1;-mavx;-mavx2;-faligned-new;-std=c++17;-w;-g;-ggdb;-O0")

if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|i[3-6]86")
    target_compile_options(SCI-utils INTERFACE "-maes;-msse4.1;-mavx;-mavx2")
endif()

if(USE_RANDOM_DEVICE)
    target_compile_definitions(SCI-utils INTERFACE EMP_USE_RANDOM_DEVICE=1)
else(USE_RANDOM_DEVICE)
    target_compile_options(SCI-utils INTERFACE "-mrdseed")
endif(USE_RANDOM_DEVICE)

if(APPLE)
    find_path(LIBOMP_INCLUDE_DIR omp.h
        PATHS /opt/homebrew/opt/libomp/include /usr/local/opt/libomp/include)
endif()

target_include_directories(SCI-utils
    INTERFACE ${OPENSSL_INCLUDE_DIR} ${GMP_INCLUDE_DIR} ${LIBOMP_INCLUDE_DIR}
)

target_link_libraries(SCI-utils
    INTERFACE ${OPENSSL_LIBRARIES} ${GMP_LIBRARIES}
)
