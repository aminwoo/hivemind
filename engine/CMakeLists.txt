cmake_minimum_required(VERSION 3.16)

if(POLICY CMP0146)
    cmake_policy(SET CMP0146 OLD)
endif()
if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

project(hivemind VERSION 2.2.0 LANGUAGES CXX)

# Defaults to tensorrt so existing build commands are unaffected; the
# onnxruntime backend is opt-in and needs no CUDA.
set(HIVEMIND_BACKEND "tensorrt" CACHE STRING
    "Inference backend: tensorrt (NVIDIA, FP16) or onnxruntime (portable, FP32)")
set_property(CACHE HIVEMIND_BACKEND PROPERTY STRINGS onnxruntime tensorrt)
option(HIVEMIND_FAST_BUILD "Favor faster local builds over maximum runtime optimization" ON)
option(HIVEMIND_USE_UNITY_BUILD "Enable unity builds for project targets" ON)
option(HIVEMIND_USE_CCACHE "Use ccache when available" ON)
option(HIVEMIND_NATIVE_ARCH "Optimize for the build machine's CPU" ON)
option(HIVEMIND_USE_PEXT "Use BMI2 PEXT for sliding attacks (benchmark first; slower on most CPUs tested)" OFF)
option(HIVEMIND_ORT_FP16 "Use native FP16 tensors with the ONNX Runtime backend" OFF)
option(HIVEMIND_PORTABLE_BUNDLE "Use relocatable runtime search paths for release packaging" OFF)

# Set C++ version and optimization level. Keep compiler flags portable: GNU
# switches passed unconditionally here used to make the ONNX Runtime backend
# fail during CMake's first compiler check on Visual Studio.
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(HIVEMIND_USE_CCACHE)
    find_program(CCACHE_PROGRAM ccache)
    if(CCACHE_PROGRAM)
        set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
        set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    endif()
endif()

# Fast local builds trade some runtime optimization for much shorter compile/link times.
set(HIVEMIND_COMMON_DEFINITIONS NDEBUG NNUE_EMBEDDING_OFF)
if(MSVC)
    set(HIVEMIND_COMMON_FLAGS /W4 /O2)
else()
    set(HIVEMIND_COMMON_FLAGS -Wall)
endif()
if(HIVEMIND_USE_PEXT)
    list(APPEND HIVEMIND_COMMON_DEFINITIONS HIVEMIND_ENABLE_PEXT)
endif()
if(HIVEMIND_NATIVE_ARCH)
    if(MSVC)
        list(APPEND HIVEMIND_COMMON_FLAGS /arch:AVX2)
    else()
        list(APPEND HIVEMIND_COMMON_FLAGS -march=native)
    endif()
endif()
if(HIVEMIND_FAST_BUILD OR MSVC)
    if(NOT MSVC)
        list(APPEND HIVEMIND_COMMON_FLAGS -O2)
    endif()
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION FALSE)
else()
    list(APPEND HIVEMIND_COMMON_FLAGS -Ofast -flto)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -flto")
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()
add_compile_options(${HIVEMIND_COMMON_FLAGS})
add_compile_definitions(${HIVEMIND_COMMON_DEFINITIONS})
if(WIN32)
    add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

# Add source files
set(LIB_SOURCES
    src/common/globals.cc
    src/environment/board.cc
    src/environment/planes.cc
    src/environment/zobrist.cc
    # src/nn/engine.cc or engine_ort.cc is appended below per backend
    src/nn/onnx_utils.cc
    src/search/agent.cc
    src/search/mate_probe.cc
    src/search/node.cc
    src/search/searchthread.cc
    src/interface/uci.cc
    src/tools/benchmark.cc
    src/tools/selfplay.cc
    src/tools/tournament.cc
)

if(HIVEMIND_BACKEND STREQUAL "tensorrt")
    list(APPEND LIB_SOURCES src/nn/engine.cc)

    # For finding FindTensorRT.cmake
    set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

    if(NOT TensorRT_DIR)
        if(DEFINED ENV{TensorRT_DIR})
            set(TensorRT_DIR $ENV{TensorRT_DIR})
        else()
            set(TensorRT_DIR /usr/src/tensorrt/)
        endif()
    endif()
    if(NOT CUDA_TOOLKIT_ROOT_DIR)
        set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
    endif()

    find_package(TensorRT REQUIRED)
    find_package(CUDA REQUIRED)

    set(CMAKE_BUILD_RPATH "${TensorRT_DIR}/lib;${CUDA_TOOLKIT_ROOT_DIR}/lib64;${CUDA_TOOLKIT_ROOT_DIR}/lib")
    set(CMAKE_INSTALL_RPATH "${TensorRT_DIR}/lib;${CUDA_TOOLKIT_ROOT_DIR}/lib64;${CUDA_TOOLKIT_ROOT_DIR}/lib")
elseif(HIVEMIND_BACKEND STREQUAL "onnxruntime")
    list(APPEND LIB_SOURCES src/nn/engine_ort.cc)

    # Prefer a vendored runtime (third_party/onnxruntime), fall back to the system.
    if(NOT ONNXRuntime_ROOT)
        if(DEFINED ENV{ONNXRuntime_ROOT})
            set(ONNXRuntime_ROOT $ENV{ONNXRuntime_ROOT})
        elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/onnxruntime")
            set(ONNXRuntime_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/onnxruntime")
        endif()
    endif()

    find_path(ONNXRuntime_INCLUDE_DIR onnxruntime_cxx_api.h
        HINTS ${ONNXRuntime_ROOT}/include
        PATH_SUFFIXES onnxruntime onnxruntime/core/session)
    find_library(ONNXRuntime_LIBRARY NAMES onnxruntime
        HINTS ${ONNXRuntime_ROOT}/lib)

    if(NOT ONNXRuntime_INCLUDE_DIR OR NOT ONNXRuntime_LIBRARY)
        message(FATAL_ERROR
            "ONNX Runtime not found. Install onnxruntime-devel, or download a "
            "release into third_party/onnxruntime, or pass -DONNXRuntime_ROOT=<dir>.")
    endif()
    message(STATUS "ONNX Runtime: ${ONNXRuntime_LIBRARY}")

    get_filename_component(ONNXRuntime_LIBDIR "${ONNXRuntime_LIBRARY}" DIRECTORY)
    if(UNIX)
        set(CMAKE_BUILD_RPATH "${ONNXRuntime_LIBDIR}")
        set(CMAKE_INSTALL_RPATH "$ORIGIN;$ORIGIN/lib;$ORIGIN/../lib;${ONNXRuntime_LIBDIR}")
        if(HIVEMIND_PORTABLE_BUNDLE)
            set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
            set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib")
        endif()
    elseif(WIN32)
        find_file(ONNXRuntime_DLL NAMES onnxruntime.dll
            HINTS ${ONNXRuntime_ROOT}/lib ${ONNXRuntime_ROOT}/bin)
        if(NOT ONNXRuntime_DLL)
            message(FATAL_ERROR
                "onnxruntime.dll not found below ONNXRuntime_ROOT (${ONNXRuntime_ROOT}).")
        endif()
    endif()
else()
    message(FATAL_ERROR "Unknown HIVEMIND_BACKEND '${HIVEMIND_BACKEND}'")
endif()

find_package(Threads REQUIRED)

add_subdirectory(src/Fairy-Stockfish)

if(HIVEMIND_USE_UNITY_BUILD)
    set_target_properties(Fairy-Stockfish PROPERTIES UNITY_BUILD ON UNITY_BUILD_BATCH_SIZE 8)
endif()

add_library(${PROJECT_NAME}_lib STATIC ${LIB_SOURCES})
if(HIVEMIND_BACKEND STREQUAL "tensorrt")
    target_include_directories(${PROJECT_NAME}_lib SYSTEM PUBLIC ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS})
    target_compile_definitions(${PROJECT_NAME}_lib PUBLIC HIVEMIND_BACKEND_TENSORRT)
else()
    target_include_directories(${PROJECT_NAME}_lib SYSTEM PUBLIC ${ONNXRuntime_INCLUDE_DIR})
    target_compile_definitions(${PROJECT_NAME}_lib PUBLIC HIVEMIND_BACKEND_ONNXRUNTIME)
    if(HIVEMIND_ORT_FP16)
        target_compile_definitions(${PROJECT_NAME}_lib PUBLIC HIVEMIND_ORT_FP16)
    endif()
endif()
target_include_directories(${PROJECT_NAME}_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
if(HIVEMIND_BACKEND STREQUAL "tensorrt")
    target_compile_definitions(${PROJECT_NAME}_lib PUBLIC
        TENSORRT_LIBRARY_DIR="${TensorRT_DIR}/lib"
        HIVEMIND_WORKSPACE_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/.."
        HIVEMIND_FP16_CONVERTER_SCRIPT="${CMAKE_CURRENT_SOURCE_DIR}/scripts/convert_onnx_fp16.py"
    )
endif()
if(HIVEMIND_USE_UNITY_BUILD)
    set_target_properties(${PROJECT_NAME}_lib PROPERTIES UNITY_BUILD ON UNITY_BUILD_BATCH_SIZE 8)
endif()
if(HIVEMIND_BACKEND STREQUAL "tensorrt")
    target_link_libraries(${PROJECT_NAME}_lib PUBLIC ${CUDA_LIBRARIES} ${TensorRT_LIBRARIES})
else()
    target_link_libraries(${PROJECT_NAME}_lib PUBLIC ${ONNXRuntime_LIBRARY})
endif()
target_link_libraries(${PROJECT_NAME}_lib PUBLIC ${CMAKE_THREAD_LIBS_INIT} Fairy-Stockfish Threads::Threads)
target_link_libraries(${PROJECT_NAME}_lib PUBLIC ${CMAKE_DL_LIBS})

add_executable(${PROJECT_NAME} src/main.cc)
if(WIN32)
    set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME hivemind)
else()
    set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME hivemind.bin)
endif()
if(HIVEMIND_USE_UNITY_BUILD)
    set_target_properties(${PROJECT_NAME} PROPERTIES UNITY_BUILD ON UNITY_BUILD_BATCH_SIZE 8)
endif()
target_link_libraries(${PROJECT_NAME} PRIVATE ${PROJECT_NAME}_lib)
target_compile_definitions(${PROJECT_NAME} PRIVATE
    HIVEMIND_VERSION="${PROJECT_VERSION}")

if(WIN32 AND HIVEMIND_BACKEND STREQUAL "onnxruntime")
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNXRuntime_DLL}" "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
        COMMENT "Copying ONNX Runtime beside hivemind.exe")
elseif(HIVEMIND_BACKEND STREQUAL "tensorrt")
    file(WRITE "${CMAKE_BINARY_DIR}/hivemind"
    "#!/bin/sh\n"
    "set -e\n"
    "SCRIPT_DIR=\"$(cd \"$(dirname \"$0\")\" && pwd)\"\n"
    "export LD_LIBRARY_PATH=\"${TensorRT_DIR}/lib:${CUDA_TOOLKIT_ROOT_DIR}/lib64:${CUDA_TOOLKIT_ROOT_DIR}/lib\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}\"\n"
    "exec \"$SCRIPT_DIR/hivemind.bin\" \"$@\"\n")
else()
    # RPATH already points at the runtime; the wrapper just keeps the same entry point.
    file(WRITE "${CMAKE_BINARY_DIR}/hivemind"
    "#!/bin/sh\n"
    "set -e\n"
    "SCRIPT_DIR=\"$(cd \"$(dirname \"$0\")\" && pwd)\"\n"
    "exec \"$SCRIPT_DIR/hivemind.bin\" \"$@\"\n")
endif()
if(UNIX)
    execute_process(COMMAND /bin/chmod 755 "${CMAKE_BINARY_DIR}/hivemind")
endif()

include(CTest)
enable_testing()
if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
