cmake_minimum_required(VERSION 3.16)
project(hivemind)

# Set C++ version and optimization level
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Ofast -DNDEBUG -DNNUE_EMBEDDING_OFF")

# Add source files
set(SOURCES
    src/main.cc
    src/engine.cc
    src/board.cc
    src/node.cc
    src/planes.cc
    src/searchthread.cc
    src/uci.cc
    src/agent.cc
    src/onnx_utils.cc
    src/benchmark.cc
)

# For finding FindTensorRT.cmake
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

set(TensorRT_DIR /usr/src/tensorrt/)
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)

find_package(Threads REQUIRED)
find_package(TensorRT REQUIRED)
find_package(CUDA REQUIRED)

add_executable(${PROJECT_NAME} ${SOURCES})
add_subdirectory(src/Fairy-Stockfish)

target_include_directories(${PROJECT_NAME} PUBLIC ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PUBLIC ${CUDA_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${TensorRT_LIBRARIES} Fairy-Stockfish Threads::Threads)