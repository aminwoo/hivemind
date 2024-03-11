cmake_minimum_required(VERSION 3.16)
project(hivemind)

# Set C++ version and optimization level
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Ofast -DNDEBUG -DNNUE_EMBEDDING_OFF")

# For finding FindTensorRT.cmake
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

set(TensorRT_DIR /usr/src/tensorrt/)
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)

find_package(TensorRT REQUIRED)
find_package(CUDA REQUIRED)

add_executable(${PROJECT_NAME} main.cpp engine.cpp planes.cpp bugboard.cpp network.cpp node.cpp clock.cpp searchthread.cpp agent.cpp)
add_subdirectory(Fairy-Stockfish)

target_include_directories(${PROJECT_NAME} PUBLIC 
                        ${CUDA_INCLUDE_DIRS} 
                        ${TensorRT_INCLUDE_DIRS}
                        ${TensorRT_DIR}/samples/common)

target_link_libraries(${PROJECT_NAME} PUBLIC 
                        ${CUDA_LIBRARIES} 
                        ${CMAKE_THREAD_LIBS_INIT} 
                        ${TensorRT_LIBRARIES} Fairy-Stockfish)



