#pragma once

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <chrono>
#include <memory>

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity == Severity::kERROR || severity == Severity::kINTERNAL_ERROR) {
            std::cerr << "TensorRT error: " << msg << std::endl;
        }
        /*if (severity != Severity::kINFO) {*/
        /*    std::cout << msg << std::endl;*/
        /*}*/
    }
};

class Engine {
public:
    Engine();
    ~Engine();

    bool loadNetwork(const std::string& onnxFile, const std::string& engineFile);
    bool runInference(float* inputPlanes, float* valueOutput, float* policyOutput);

private:
    std::unique_ptr<nvinfer1::ICudaEngine> m_engine = nullptr;
    std::unique_ptr<nvinfer1::IExecutionContext> m_context = nullptr;
    Logger m_logger;

    // CUDA resources
    cudaStream_t m_cudaStream = nullptr; // CUDA stream for asynchronous operations
    void* m_inputBuffer = nullptr;      // GPU buffer for input
    void* m_valueOutputBuffer = nullptr; // GPU buffer for value output
    void* m_policyOutputBuffer = nullptr; // GPU buffer for policy output

    bool buildEngineFromONNX(const std::string& onnxFile);
    bool loadEngineFromFile(const std::string& engineFile);
    bool saveEngineToFile(const std::string& engineFile);
    bool initializeResources();
};
