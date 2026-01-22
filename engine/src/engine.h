#pragma once

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <chrono>
#include "constants.h"
#include <memory>

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity == Severity::kERROR || severity == Severity::kINTERNAL_ERROR) {
            std::cerr << "TensorRT error: " << msg << std::endl;
        }
    }
};

class Engine {
public:
    Engine(int deviceId);
    ~Engine();

    bool loadNetwork(const std::string& onnxFile, const std::string& engineFile);
    bool runInference(float* obs, float* value, float* piA, float* piB);

private:
    // The GPU id that this Engine instance is bound to.
    int m_deviceId;
    
    std::unique_ptr<nvinfer1::ICudaEngine> m_engine = nullptr;
    std::unique_ptr<nvinfer1::IExecutionContext> m_context = nullptr;
    Logger m_logger;

    // CUDA resources
    cudaStream_t m_cudaStream = nullptr; // CUDA stream for asynchronous operations
    void* m_obsBuffer = nullptr;         // GPU buffer for input (obs)
    void* m_valueBuffer = nullptr;       // GPU buffer for value output
    void* m_policyABuffer = nullptr;     // GPU buffer for policy A output
    void* m_policyBBuffer = nullptr;     // GPU buffer for policy B output

    bool buildEngineFromONNX(const std::string& onnxFile);
    bool loadEngineFromFile(const std::string& engineFile);
    bool saveEngineToFile(const std::string& engineFile);
    bool initializeResources();
};
