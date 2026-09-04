# source:
# https://github.com/NVIDIA/tensorrt-laboratory/blob/master/cmake/FindTensorRT.cmake

# This module defines the following variables:
#
# ::
#
#   TensorRT_INCLUDE_DIRS
#   TensorRT_LIBRARIES
#   TensorRT_FOUND
#
# ::
#
#   TensorRT_VERSION_STRING - version (x.y.z)
#   TensorRT_VERSION_MAJOR  - major version (x)
#   TensorRT_VERSION_MINOR  - minor version (y)
#   TensorRT_VERSION_PATCH  - patch version (z)
#
# Hints
# ^^^^^
# A user may set ``TensorRT_DIR`` to an installation root to tell this module where to look.
#
if(TensorRT_DIR)
    find_path(TensorRT_INCLUDE_DIR
        NAMES NvInfer.h
        PATHS "${TensorRT_DIR}"
        NO_DEFAULT_PATH
        PATH_SUFFIXES include)

    find_library(TensorRT_LIBRARY
        NAMES nvinfer nvinfer_11
        PATHS "${TensorRT_DIR}"
        NO_DEFAULT_PATH
        PATH_SUFFIXES lib)

    find_library(TensorRT_NVONNXPARSER_LIBRARY
        NAMES nvonnxparser nvonnxparser_11
        PATHS "${TensorRT_DIR}"
        NO_DEFAULT_PATH
        PATH_SUFFIXES lib)
endif()

if(NOT TensorRT_INCLUDE_DIR)
    find_path(TensorRT_INCLUDE_DIR NAMES NvInfer.h PATHS "/usr" PATH_SUFFIXES include)
endif()

if(NOT TensorRT_LIBRARY)
    find_library(TensorRT_LIBRARY NAMES nvinfer nvinfer_11 PATHS "/usr" PATH_SUFFIXES lib)
endif()

if(NOT TensorRT_NVONNXPARSER_LIBRARY)
    find_library(TensorRT_NVONNXPARSER_LIBRARY NAMES nvonnxparser nvonnxparser_11 PATHS "/usr" PATH_SUFFIXES lib)
endif()

mark_as_advanced(TensorRT_INCLUDE_DIR)

if(TensorRT_INCLUDE_DIR AND EXISTS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h")
    # TensorRT 11 defines the public NV_TENSORRT_* macros through numeric
    # TRT_*_ENTERPRISE macros, while older releases used numeric values
    # directly. Match either form from the dedicated version header.
    file(STRINGS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h" TensorRT_MAJOR
        REGEX "^#define (TRT_MAJOR_ENTERPRISE|NV_TENSORRT_MAJOR) [0-9]+.*$")
    file(STRINGS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h" TensorRT_MINOR
        REGEX "^#define (TRT_MINOR_ENTERPRISE|NV_TENSORRT_MINOR) [0-9]+.*$")
    file(STRINGS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h" TensorRT_PATCH
        REGEX "^#define (TRT_PATCH_ENTERPRISE|NV_TENSORRT_PATCH) [0-9]+.*$")

    string(REGEX REPLACE ".* ([0-9]+).*$" "\\1" TensorRT_VERSION_MAJOR "${TensorRT_MAJOR}")
    string(REGEX REPLACE ".* ([0-9]+).*$" "\\1" TensorRT_VERSION_MINOR "${TensorRT_MINOR}")
    string(REGEX REPLACE ".* ([0-9]+).*$" "\\1" TensorRT_VERSION_PATCH "${TensorRT_PATCH}")
    set(TensorRT_VERSION_STRING "${TensorRT_VERSION_MAJOR}.${TensorRT_VERSION_MINOR}.${TensorRT_VERSION_PATCH}")
endif()

include(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(TensorRT REQUIRED_VARS
    TensorRT_LIBRARY TensorRT_NVONNXPARSER_LIBRARY TensorRT_INCLUDE_DIR
    VERSION_VAR TensorRT_VERSION_STRING)

if(TensorRT_FOUND)
    set(TensorRT_INCLUDE_DIRS ${TensorRT_INCLUDE_DIR})

    if(NOT TensorRT_LIBRARIES)
        set(TensorRT_LIBRARIES ${TensorRT_LIBRARY} ${TensorRT_NVONNXPARSER_LIBRARY} ${TensorRT_NVPARSERS_LIBRARY})
    endif()

    if(NOT TARGET TensorRT::TensorRT)
        add_library(TensorRT::TensorRT UNKNOWN IMPORTED)
        set_target_properties(TensorRT::TensorRT PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIRS}")
        set_property(TARGET TensorRT::TensorRT APPEND PROPERTY IMPORTED_LOCATION "${TensorRT_LIBRARY}")
    endif()
endif()
