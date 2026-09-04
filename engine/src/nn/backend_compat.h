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
// ONNX Runtime builds can use either FP32 tensors or, with
// HIVEMIND_ORT_FP16, a small layout-compatible IEEE 754 binary16 type. Pinned
// memory has no meaning without a device transfer, so it becomes an aligned
// host allocation.

#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <new>

#if defined(_WIN32)
#include <malloc.h>
#endif

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

#elif defined(HIVEMIND_ORT_FP16)

// The search tree spells the network scalar type `__half` throughout. Keep the
// same two-byte layout as ONNX's FLOAT16 tensors without taking a CUDA header
// dependency on portable builds.
struct __half {
    std::uint16_t bits = 0;
};

static_assert(sizeof(__half) == sizeof(std::uint16_t));

inline float __half2float(__half value) {
    const std::uint32_t sign = static_cast<std::uint32_t>(value.bits & 0x8000u) << 16;
    const std::uint32_t exponent = (value.bits >> 10) & 0x1fu;
    const std::uint32_t mantissa = value.bits & 0x03ffu;

    std::uint32_t result;
    if (exponent == 0) {
        if (mantissa == 0) {
            result = sign;
        } else {
            std::uint32_t normalized = mantissa;
            int shift = 0;
            while ((normalized & 0x0400u) == 0) {
                normalized <<= 1;
                ++shift;
            }
            normalized &= 0x03ffu;
            result = sign
                | static_cast<std::uint32_t>(113 - shift) << 23
                | normalized << 13;
        }
    } else if (exponent == 0x1fu) {
        result = sign | 0x7f800000u | (mantissa << 13);
    } else {
        result = sign | ((exponent + 112u) << 23) | (mantissa << 13);
    }
    return std::bit_cast<float>(result);
}

inline __half __float2half_rn(float value) {
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    const std::uint16_t sign = static_cast<std::uint16_t>((bits >> 16) & 0x8000u);
    const std::uint32_t exponent = (bits >> 23) & 0xffu;
    const std::uint32_t mantissa = bits & 0x007fffffu;

    if (exponent == 0xffu) {
        const std::uint16_t payload = mantissa == 0
            ? 0
            : static_cast<std::uint16_t>((mantissa >> 13) | 0x0200u);
        return {static_cast<std::uint16_t>(sign | 0x7c00u | payload)};
    }

    const int halfExponent = static_cast<int>(exponent) - 127 + 15;
    if (halfExponent >= 31) {
        return {static_cast<std::uint16_t>(sign | 0x7c00u)};
    }
    if (halfExponent <= 0) {
        if (halfExponent < -10) {
            return {sign};
        }
        const std::uint32_t significand = mantissa | 0x00800000u;
        const int shift = 14 - halfExponent;
        std::uint32_t rounded = significand >> shift;
        const std::uint32_t remainder = significand & ((1u << shift) - 1u);
        const std::uint32_t halfway = 1u << (shift - 1);
        if (remainder > halfway || (remainder == halfway && (rounded & 1u))) {
            ++rounded;
        }
        return {static_cast<std::uint16_t>(sign | rounded)};
    }

    std::uint32_t rounded = (static_cast<std::uint32_t>(halfExponent) << 10)
        | (mantissa >> 13);
    const std::uint32_t remainder = mantissa & 0x1fffu;
    if (remainder > 0x1000u || (remainder == 0x1000u && (rounded & 1u))) {
        ++rounded;
    }
    return {static_cast<std::uint16_t>(sign | rounded)};
}

#else  // portable FP32 backend

using __half = float;

inline constexpr float __half2float(float value) { return value; }
inline constexpr float __float2half_rn(float value) { return value; }

#endif

#if !defined(HIVEMIND_BACKEND_TENSORRT)
namespace hm {
// 64-byte alignment so the AVX2 stores in planes.cc stay on cache lines.
inline bool alloc_pinned(void** ptr, std::size_t bytes) {
    constexpr std::size_t kAlign = 64;
    const std::size_t rounded = ((bytes + kAlign - 1) / kAlign) * kAlign;
#if defined(_WIN32)
    void* p = _aligned_malloc(rounded, kAlign);
#else
    void* p = std::aligned_alloc(kAlign, rounded);
#endif
    if (!p) return false;
    *ptr = p;
    return true;
}
inline void free_pinned(void* ptr) {
#if defined(_WIN32)
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif
}
inline const char* alloc_error() { return "host allocation failed"; }
}  // namespace hm
#endif
