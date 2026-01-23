#pragma once

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <fstream>
#include <mutex>
#include <vector>
#include <stdexcept>
#include <chrono>
#include <memory>
#include "constants.h"

/**
 * @brief Simple Logger implementation for TensorRT.
 */
class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        // Only log warnings and errors to reduce console noise during high-speed inference
        if (severity <= Severity::kWARNING) {
            std::cerr << (severity == Severity::kERROR ? "[ERROR] " : "[WARNING] ") << msg << std::endl;
        }
    }
};

/**
 * @brief Optimized TensorRT inference engine wrapper for RTX 4070.
 * * Features: CUDA Graphs, FP16 Precision, and Async Memory Streams.
 */
class Engine {
public:
    explicit Engine(int deviceId);
    ~Engine();

    // Prevent copying to avoid double-free of CUDA resources
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    bool loadNetwork(const std::string& onnxFile, const std::string& engineFile);
    
    /**
     * @brief Performs inference. For max QPS, ensure input/output pointers 
     * are allocated via cudaMallocHost (Pinned Memory).
     */
    bool runInference(float* obs, float* value, float* piA, float* piB);

private:
    // Device ID and Logger
    int m_deviceId;
    Logger m_logger;
    
    // TensorRT Core Objects
    std::unique_ptr<nvinfer1::ICudaEngine> m_engine = nullptr;
    std::unique_ptr<nvinfer1::IExecutionContext> m_context = nullptr;
    
    // CUDA Stream and Graph resources
    cudaStream_t m_cudaStream = nullptr;
    cudaGraph_t m_graph;
    cudaGraphExec_t m_instance;
    bool m_graphCreated = false;
    
    // Mutex for thread-safe access to the single execution context
    std::mutex m_inferenceMutex;

    // GPU DEVICE Buffers (These hold the data on the 4070 VRAM)
    void* m_deviceObsBuffer = nullptr;
    void* m_deviceValueBuffer = nullptr;
    void* m_devicePolicyABuffer = nullptr;
    void* m_devicePolicyBBuffer = nullptr;

    // Internal helper methods
    bool buildEngineFromONNX(const std::string& onnxFile);
    bool loadEngineFromFile(const std::string& engineFile);
    bool saveEngineToFile(const std::string& engineFile);
    bool initializeResources();
};