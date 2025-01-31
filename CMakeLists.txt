cmake_minimum_required(VERSION 3.16)
project(hivemind)

# Set C++ version and optimization level
set(CMAKE_CXX_STANDARD 20)
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

# ===========================
# Added Fetch Command Section
# ===========================

# Variables for fetching the model
set(MODEL_FILE model-1.36699-0.564-0078-v3.0.onnx)
set(REPO_URL https://github.com/aminwoo/hivemind-networks/raw/master/weights)
set(NETWORKS_DIR ${CMAKE_SOURCE_DIR}/networks)
set(MODEL_PATH ${NETWORKS_DIR}/model.onnx)

# Add a custom target to fetch the model
add_custom_target(fetch_model
    COMMAND ${CMAKE_COMMAND} -E make_directory ${NETWORKS_DIR}
    COMMAND ${CMAKE_COMMAND} -E echo "Downloading ${MODEL_FILE} from ${REPO_URL}"
    COMMAND curl -sL ${REPO_URL}/${MODEL_FILE} -o ${MODEL_PATH}
    COMMENT "Downloading ${MODEL_FILE}"
    VERBATIM
)

# Ensure that the model is downloaded before building the executable
add_dependencies(${PROJECT_NAME} fetch_model)

# Optionally, you can make fetch_model part of the default build
# by adding it to the ALL target. Uncomment the following line if desired:
# add_dependencies(ALL ${PROJECT_NAME} fetch_model)
