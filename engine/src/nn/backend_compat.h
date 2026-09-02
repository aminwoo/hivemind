#pragma once

// Backend compatibility shim.
//
// Hivemind was written against TensorRT, so the search pipeline passes network
// tensors around as `__half` and allocates its staging buffers with
// `cudaMallocHost`. Neither is available when the engine is built against a
// portable inference backend, so this header supplies the two things the rest
// of the tree actually needs from CUDA:
//
//   * a half type and its float conversions, and
//   * pinned-host-memory allocation.
//
// On the ONNX Runtime backend the network runs in fp32, so `__half` is a plain
// float and both conversions are identities. Pinned memory has no meaning
// without a device transfer, so it becomes an aligned host allocation.

#include <cstddef>
#include <cstdlib>
#include <new>

#if defined(HIVEMIND_BACKEND_TENSORRT)

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

namespace hm {
inline bool alloc_pinned(void** ptr, std::size_t bytes) {
    return cudaMallocHost(ptr, bytes) == cudaSuccess;
}
inline void free_pinned(void* ptr) { cudaFreeHost(ptr); }
inline const char* alloc_error() { return cudaGetErrorString(cudaGetLastError()); }
}  // namespace hm

#else  // portable backend

// The search tree spells the network scalar type `__half` throughout. Aliasing
// it to float keeps that code untouched and lets the fp32 graph feed directly
// from the plane encoder with no conversion pass.
using __half = float;

inline constexpr float __half2float(float value) { return value; }
inline constexpr float __float2half_rn(float value) { return value; }

namespace hm {
// 64-byte alignment so the AVX2 stores in planes.cc stay on cache lines.
inline bool alloc_pinned(void** ptr, std::size_t bytes) {
    constexpr std::size_t kAlign = 64;
    const std::size_t rounded = ((bytes + kAlign - 1) / kAlign) * kAlign;
    void* p = std::aligned_alloc(kAlign, rounded);
    if (!p) return false;
    *ptr = p;
    return true;
}
inline void free_pinned(void* ptr) { std::free(ptr); }
inline const char* alloc_error() { return "host allocation failed"; }
}  // namespace hm

#endif
