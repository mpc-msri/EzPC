cmake_minimum_required(VERSION 3.16)
project(fptraining)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-write-strings -Wno-unused-result -maes -Wno-ignored-attributes -march=native")
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Threads REQUIRED)

add_executable(fptraining 
    main.cpp 
    random.cpp
    backend/minillama/cryptoTools/cryptoTools/Common/Defines.cpp
    backend/minillama/cryptoTools/cryptoTools/Common/Log.cpp
    backend/minillama/cryptoTools/cryptoTools/Crypto/AES.cpp
    backend/minillama/cryptoTools/cryptoTools/Crypto/PRNG.cpp
    backend/minillama/api.cpp
    backend/minillama/comms.cpp
    backend/minillama/config.cpp
    backend/minillama/conv.cpp
    backend/minillama/dcf.cpp
    backend/minillama/input_prng.cpp
    backend/minillama/mult.cpp
    backend/minillama/prng.cpp
    backend/minillama/pubdiv.cpp
    backend/minillama/relutruncate.cpp
    backend/minillama/stats.cpp
    backend/minillama/utils.cpp)
target_link_libraries (fptraining Eigen3::Eigen Threads::Threads)
target_include_directories(fptraining PRIVATE backend/minillama/cryptoTools)
