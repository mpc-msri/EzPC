cmake_minimum_required(VERSION 3.17)
project(sytorch)

if(APPLE)

    if(CMAKE_C_COMPILER_ID MATCHES "Clang\$")
        set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp")
        set(OpenMP_C_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY omp)
    endif()

    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang\$")
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY omp)
    endif()

endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-write-strings -Wno-unused-result -maes -Wno-ignored-attributes -march=native -Wno-deprecated-declarations -fopenmp")
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Threads REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_subdirectory(ext/cryptoTools)
add_subdirectory(ext/bitpack)
add_subdirectory(ext/llama)
add_subdirectory(ext/sci)
# add_subdirectory(../SCI ext/sci)

add_library(${PROJECT_NAME} STATIC)
target_sources(${PROJECT_NAME} 
PRIVATE
    src/sytorch/random.cpp
    src/sytorch/backend/cleartext.cpp
    src/sytorch/backend/baseline_cleartext.cpp
    src/sytorch/backend/float.cpp
    src/sytorch/softmax.cpp
)

target_include_directories(${PROJECT_NAME}
PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries (${PROJECT_NAME} Eigen3::Eigen Threads::Threads CUDA::cudart LLAMA cryptoTools SCI-FloatML)

add_executable(
    gpt2
    examples/gpt2.cpp
)

target_link_libraries(gpt2 ${PROJECT_NAME})

add_executable(
    resnet50
    examples/resnet50.cpp
)

target_link_libraries(resnet50 ${PROJECT_NAME})

add_executable(
    resnet18
    examples/resnet18.cpp
)

target_link_libraries(resnet18 ${PROJECT_NAME})

add_executable(
    vgg16
    examples/vgg16.cpp
)

target_link_libraries(vgg16 ${PROJECT_NAME})

add_executable(
    dpftest
    tests/dpf.cpp
)

target_link_libraries(dpftest ${PROJECT_NAME})

add_executable(
    dcftest
    tests/dcf.cpp
)

target_link_libraries(dcftest ${PROJECT_NAME})


add_executable(
    pubcmptest
    tests/pubcmp.cpp
)

target_link_libraries(pubcmptest ${PROJECT_NAME})

add_executable(
    cliptest
    tests/clip.cpp
)

target_link_libraries(cliptest ${PROJECT_NAME})

add_executable(
    mp_cliptest
    tests/multi_party/clip.cpp
)

target_link_libraries(mp_cliptest ${PROJECT_NAME})

add_executable(
    mp_luttest
    tests/multi_party/lut.cpp
)

target_link_libraries(mp_luttest ${PROJECT_NAME})

add_executable(
    mp_exptest
    tests/multi_party/exp.cpp
)

target_link_libraries(mp_exptest ${PROJECT_NAME})

add_executable(
    mp_tanhtest
    tests/multi_party/tanh.cpp
)

target_link_libraries(mp_tanhtest ${PROJECT_NAME})

add_executable(
    mp_gelutest
    tests/multi_party/gelu.cpp
)

target_link_libraries(mp_gelutest ${PROJECT_NAME})

add_executable(
    mp_softmax_test
    tests/multi_party/softmax.cpp
)

target_link_libraries(mp_softmax_test ${PROJECT_NAME})

add_executable(
    bf16test
    tests/bf16.cpp
)

target_link_libraries(bf16test ${PROJECT_NAME})

add_executable(
    mp_bf16_test
    tests/multi_party/bf16.cpp
)

target_link_libraries(mp_bf16_test ${PROJECT_NAME})

add_executable(
    mp_rsqrt_test
    tests/multi_party/rsqrt.cpp
)

target_link_libraries(mp_rsqrt_test ${PROJECT_NAME})

add_executable(
    mp_layernorm_test
    tests/multi_party/layernorm.cpp
)

target_link_libraries(mp_layernorm_test ${PROJECT_NAME})

add_executable(
    truncatereducetest
    tests/truncatereduce.cpp
)

target_link_libraries(truncatereducetest ${PROJECT_NAME})

add_executable(
    mp_truncatereduce_test
    tests/multi_party/truncatereduce.cpp
)

target_link_libraries(mp_truncatereduce_test ${PROJECT_NAME})

add_executable(
    bert
    examples/bert.cpp
)

target_link_libraries(bert ${PROJECT_NAME})

add_executable(
    lutsstest
    tests/lutss.cpp
)

target_link_libraries(lutsstest ${PROJECT_NAME})

add_executable(
    dpfet
    tests/dpfet.cpp
)

target_link_libraries(dpfet ${PROJECT_NAME})

add_executable(
    dcf_dpf_et
    tests/dcf_dpf_et.cpp
)

target_link_libraries(dcf_dpf_et ${PROJECT_NAME})


add_executable(
    sloth_drelu
    tests/sloth_drelu.cpp
)

target_link_libraries(sloth_drelu ${PROJECT_NAME})

add_executable(
    mp_sloth_relu
    tests/multi_party/sloth_relu.cpp
)

target_link_libraries(mp_sloth_relu ${PROJECT_NAME})

add_executable(
    mp_sloth_clip
    tests/multi_party/sloth_clip.cpp
)

target_link_libraries(mp_sloth_clip ${PROJECT_NAME})

add_executable(
    mp_sloth_maxpool
    tests/multi_party/sloth_maxpool.cpp
)

target_link_libraries(mp_sloth_maxpool ${PROJECT_NAME})

add_executable(
    mp_prtrunc
    tests/multi_party/prtrunc.cpp
)

target_link_libraries(mp_prtrunc ${PROJECT_NAME})

add_executable(
    gelu_ulp
    tests/gelu_ulp.cpp
)

target_link_libraries(gelu_ulp ${PROJECT_NAME})

add_executable(
    mp_gemm
    tests/multi_party/gemm.cpp
)

target_link_libraries(mp_gemm ${PROJECT_NAME})

add_executable(
    mp_sloth_drelu
    tests/multi_party/sloth_drelu.cpp
)

target_link_libraries(mp_sloth_drelu ${PROJECT_NAME})

add_executable(
    mp_sloth_maxpool_tri
    tests/multi_party/sloth_maxpool_tri.cpp
)

target_link_libraries(mp_sloth_maxpool_tri ${PROJECT_NAME})

add_executable(
    eigenbenchmark
    tests/eigenbenchmark.cpp
)

target_link_libraries(eigenbenchmark ${PROJECT_NAME})

add_executable(
    llama7b
    examples/llama7b.cpp
)

target_link_libraries(llama7b ${PROJECT_NAME})

add_executable(
    wrap
    tests/wrap.cpp
)

target_link_libraries(wrap ${PROJECT_NAME})

add_executable(
    mp_sloth_lrs
    tests/multi_party/sloth_lrs.cpp
)

target_link_libraries(mp_sloth_lrs ${PROJECT_NAME})

add_executable(
    mp_sloth_ars
    tests/multi_party/sloth_ars.cpp
)

target_link_libraries(mp_sloth_ars ${PROJECT_NAME})

add_executable(
    gpt2dummy
    examples/gpt2dummy.cpp
)

target_link_libraries(gpt2dummy ${PROJECT_NAME})

add_executable(
    gpt2correctness
    examples/gpt2correctness.cpp
)

target_link_libraries(gpt2correctness ${PROJECT_NAME})

add_executable(
    gptneo
    examples/gptneo.cpp
)

target_link_libraries(gptneo ${PROJECT_NAME})

add_executable(
    gpt-neo_nexttoken
    examples/gpt-neo_nexttoken.cpp
)

target_link_libraries(gpt-neo_nexttoken ${PROJECT_NAME})

add_executable(
    mp_sloth_ars_faithful
    tests/multi_party/sloth_ars_faithful.cpp
)

target_link_libraries(mp_sloth_ars_faithful ${PROJECT_NAME})

add_executable(
    bertbenchmark
    examples/bertbenchmark.cpp
)

target_link_libraries(bertbenchmark ${PROJECT_NAME})

add_executable(
    gpt2benchmark
    examples/gpt2benchmark.cpp
)

target_link_libraries(gpt2benchmark ${PROJECT_NAME})

add_executable(
    gptneobenchmark
    examples/gptneobenchmark.cpp
)

target_link_libraries(gptneobenchmark ${PROJECT_NAME})
