cmake_minimum_required(VERSION 3.16)
project(hivemind)

option(HIVEMIND_FAST_BUILD "Favor faster local builds over maximum runtime optimization" ON)
option(HIVEMIND_USE_UNITY_BUILD "Enable unity builds for project targets" ON)
option(HIVEMIND_USE_CCACHE "Use ccache when available" ON)

# Set C++ version and optimization level
set(CMAKE_CXX_STANDARD 23)

if(HIVEMIND_USE_CCACHE)
    find_program(CCACHE_PROGRAM ccache)
    if(CCACHE_PROGRAM)
        set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
        set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    endif()
endif()

# Fast local builds trade some runtime optimization for much shorter compile/link times.
set(HIVEMIND_COMMON_FLAGS "-Wall -DNDEBUG -DNNUE_EMBEDDING_OFF -march=native")
if(HIVEMIND_FAST_BUILD)
    string(APPEND HIVEMIND_COMMON_FLAGS " -O2")
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION FALSE)
else()
    string(APPEND HIVEMIND_COMMON_FLAGS " -Ofast -flto")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -flto")
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${HIVEMIND_COMMON_FLAGS}")

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
    src/selfplay.cc
    src/tournament.cc
    src/zobrist.cc
    src/globals.cc
)

# For finding FindTensorRT.cmake
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

if(NOT TensorRT_DIR)
    set(TensorRT_DIR /usr/src/tensorrt/)
endif()
if(NOT CUDA_TOOLKIT_ROOT_DIR)
    set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
endif()

find_package(Threads REQUIRED)
find_package(TensorRT REQUIRED)
find_package(CUDA REQUIRED)

set(CMAKE_BUILD_RPATH "${TensorRT_DIR}/lib;${CUDA_TOOLKIT_ROOT_DIR}/lib64;${CUDA_TOOLKIT_ROOT_DIR}/lib")
set(CMAKE_INSTALL_RPATH "${TensorRT_DIR}/lib;${CUDA_TOOLKIT_ROOT_DIR}/lib64;${CUDA_TOOLKIT_ROOT_DIR}/lib")

add_subdirectory(src/Fairy-Stockfish)

if(HIVEMIND_USE_UNITY_BUILD)
    set_target_properties(Fairy-Stockfish PROPERTIES UNITY_BUILD ON UNITY_BUILD_BATCH_SIZE 8)
endif()

add_library(${PROJECT_NAME}_lib STATIC ${LIB_SOURCES})
target_include_directories(${PROJECT_NAME}_lib PUBLIC ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_definitions(${PROJECT_NAME}_lib PUBLIC
    TENSORRT_LIBRARY_DIR="${TensorRT_DIR}/lib"
    HIVEMIND_WORKSPACE_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/.."
    HIVEMIND_FP16_CONVERTER_SCRIPT="${CMAKE_CURRENT_SOURCE_DIR}/scripts/convert_onnx_fp16.py"
)
if(HIVEMIND_USE_UNITY_BUILD)
    set_target_properties(${PROJECT_NAME}_lib PROPERTIES UNITY_BUILD ON UNITY_BUILD_BATCH_SIZE 8)
endif()
target_link_libraries(${PROJECT_NAME}_lib PUBLIC ${CUDA_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${TensorRT_LIBRARIES} Fairy-Stockfish Threads::Threads)
target_link_libraries(${PROJECT_NAME}_lib PUBLIC ${CMAKE_DL_LIBS})

add_executable(${PROJECT_NAME} src/main.cc)
set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME hivemind.bin)
if(HIVEMIND_USE_UNITY_BUILD)
    set_target_properties(${PROJECT_NAME} PROPERTIES UNITY_BUILD ON UNITY_BUILD_BATCH_SIZE 8)
endif()
target_link_libraries(${PROJECT_NAME} PRIVATE ${PROJECT_NAME}_lib)

file(WRITE "${CMAKE_BINARY_DIR}/hivemind"
"#!/bin/sh\n"
"set -e\n"
"SCRIPT_DIR=\"$(cd \"$(dirname \"$0\")\" && pwd)\"\n"
"export LD_LIBRARY_PATH=\"${TensorRT_DIR}/lib:${CUDA_TOOLKIT_ROOT_DIR}/lib64:${CUDA_TOOLKIT_ROOT_DIR}/lib\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}\"\n"
"exec \"$SCRIPT_DIR/hivemind.bin\" \"$@\"\n")
execute_process(COMMAND /bin/chmod 755 "${CMAKE_BINARY_DIR}/hivemind")

include(CTest)
enable_testing()
add_subdirectory(tests)