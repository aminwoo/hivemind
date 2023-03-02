#pragma once

#include "NvInfer.h"
#include "buffers.h"


// Class to extend TensorRT logger
class Logger : public nvinfer1::ILogger {
    void log (Severity severity, const char* msg) noexcept override;
};

class Engine {
public:
    Engine(int batchSize) : batchSize(batchSize) {};
    ~Engine();
    // Build the network
    bool build(std::string onnxModelPath, std::string engineName);
    // Load and prepare the network for inference
    bool loadNetwork();
    void bind_executor_input(int channels);
    void bind_executor_policy(int channels);
    void bind_executor_value(int channels);
    // Run inference.
    void predict(float* inputPlanes, float* valueOutput, float* policyOutput);

private:
    int batchSize;
    int idxInput;
    int idxValueOutput;
    int idxPolicyOutput;

    void* deviceMemory[3];
    size_t memorySizes[3];

    void getGPUUUIDs(std::vector<std::string>& gpuUUIDs);
    bool doesFileExist(const std::string& filepath);

    std::unique_ptr<nvinfer1::ICudaEngine> m_engine = nullptr;
    std::unique_ptr<nvinfer1::IExecutionContext> m_context = nullptr;
    Logger m_logger;
    size_t m_prevBatchSize = 0;
    std::string m_engineName;
    cudaStream_t m_cudaStream = nullptr;
};
