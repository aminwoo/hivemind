cmake_minimum_required(VERSION 3.16)
project(hivemind)

# Set C++ version and optimization level
set(CMAKE_CXX_STANDARD 23)

# Performance flags:
#   -Ofast: Maximum optimization (includes -O3 + fast-math)
#   -march=native: CPU-specific optimizations (AVX2, FMA, etc.)
#   -flto: Link-time optimization for cross-file inlining
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Ofast -DNDEBUG -DNNUE_EMBEDDING_OFF -march=native -flto")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -flto")

# Add source files
set(LIB_SOURCES
    src/engine.cc
    src/board.cc
    src/node.cc
    src/planes.cc
    src/searchthread.cc
    src/uci.cc
    src/agent.cc
    src/onnx_utils.cc
    src/benchmark.cc
    src/zobrist.cc
    src/globals.cc
    src/rl/gamepgn.cc
    src/rl/selfplay.cc
    src/rl/training_data_writer.cc
    src/rl/model_eval.cc
)

# For finding FindTensorRT.cmake
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

set(TensorRT_DIR /usr/src/tensorrt/)
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)

find_package(Threads REQUIRED)
find_package(TensorRT REQUIRED)
find_package(CUDA REQUIRED)

add_subdirectory(src/Fairy-Stockfish)

add_library(${PROJECT_NAME}_lib STATIC ${LIB_SOURCES})
target_include_directories(${PROJECT_NAME}_lib PUBLIC ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(${PROJECT_NAME}_lib PUBLIC ${CUDA_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${TensorRT_LIBRARIES} Fairy-Stockfish Threads::Threads)

add_executable(${PROJECT_NAME} src/main.cc)
target_link_libraries(${PROJECT_NAME} PRIVATE ${PROJECT_NAME}_lib)

include(CTest)
enable_testing()
add_subdirectory(tests)