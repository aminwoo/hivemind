#include "engine.h"
#include <iostream>
#include <fstream>
#include <vector>

Engine::Engine(int deviceId) : m_deviceId(deviceId) {
    cudaSetDevice(m_deviceId);
    cudaStreamCreate(&m_cudaStream);
}

Engine::~Engine() {
    cudaSetDevice(m_deviceId);
    if (m_cudaStream) cudaStreamDestroy(m_cudaStream);
    
    // Clean up Device Memory
    if (m_deviceObsBuffer) cudaFree(m_deviceObsBuffer);
    if (m_deviceValueBuffer) cudaFree(m_deviceValueBuffer);
    if (m_devicePolicyABuffer) cudaFree(m_devicePolicyABuffer);
    if (m_devicePolicyBBuffer) cudaFree(m_devicePolicyBBuffer);

    // Clean up Graph
    if (m_graphCreated) {
        cudaGraphExecDestroy(m_instance);
        cudaGraphDestroy(m_graph);
    }
    
    m_context.reset();
    m_engine.reset();
}

bool Engine::loadNetwork(const std::string& onnxFile, const std::string& engineFile) {
    std::ifstream checkFile(engineFile, std::ios::binary);
    if (checkFile.good()) {
        checkFile.close();
        return loadEngineFromFile(engineFile);
    }
    return buildEngineFromONNX(onnxFile) && saveEngineToFile(engineFile);
}

bool Engine::loadEngineFromFile(const std::string& engineFile) {
    std::ifstream file(engineFile, std::ios::binary | std::ios::ate); 
    if (!file) return false;

    std::streamsize size = file.tellg(); 
    file.seekg(0, std::ios::beg);        

    std::vector<char> engineData(size);
    if (!file.read(engineData.data(), size)) return false;
    file.close();

    auto runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(m_logger));
    m_engine.reset(runtime->deserializeCudaEngine(engineData.data(), engineData.size()));
    
    return initializeResources();
}

bool Engine::saveEngineToFile(const std::string& engineFile) {
    auto serializedEngine = std::unique_ptr<nvinfer1::IHostMemory>(m_engine->serialize());
    if (!serializedEngine) return false;

    std::ofstream file(engineFile, std::ios::binary);
    if (!file) return false;

    file.write((char*)serializedEngine->data(), serializedEngine->size());
    return true;
}

bool Engine::buildEngineFromONNX(const std::string& onnxFile) {
    std::cout << "Building TensorRT engine from ONNX: " << onnxFile << std::endl;
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(m_logger));
    // TensorRT 10: explicit batch is now the default, just pass 0
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0));
    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, m_logger));

    if (!parser->parseFromFile(onnxFile.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        return false;
    }

    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    
    config->setBuilderOptimizationLevel(5);
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30); // 1GB
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
    
    auto profile = builder->createOptimizationProfile();
    const char* inputName = network->getInput(0)->getName();
    nvinfer1::Dims4 dims{SearchParams::BATCH_SIZE, NB_INPUT_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH};
    profile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMIN, dims);
    profile->setDimensions(inputName, nvinfer1::OptProfileSelector::kOPT, dims);
    profile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMAX, dims);
    config->addOptimizationProfile(profile);

    // TensorRT 10: use buildSerializedNetwork instead of buildEngineWithConfig
    auto serializedEngine = std::unique_ptr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
    if (!serializedEngine) return false;
    
    auto runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(m_logger));
    m_engine.reset(runtime->deserializeCudaEngine(serializedEngine->data(), serializedEngine->size()));
    return initializeResources();
}

bool Engine::initializeResources() {
    if (!m_engine) return false;
    m_context.reset(m_engine->createExecutionContext());
    
    // Extract batch size from the engine's input dimensions
    const char* inputName = m_engine->getIOTensorName(0);
    nvinfer1::Dims inputDims = m_engine->getTensorShape(inputName);
    m_batchSize = inputDims.d[0];  // First dimension is batch size
    
    // Set Input Shape (Required for V3 Engines/Dynamic Shapes)
    nvinfer1::Dims4 dims{m_batchSize, NB_INPUT_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH};
    m_context->setInputShape(inputName, dims);

    size_t inputSize = m_batchSize * NB_INPUT_CHANNELS * BOARD_HEIGHT * BOARD_WIDTH * sizeof(float);
    size_t valSize = m_batchSize * sizeof(float);
    size_t polSize = m_batchSize * NB_POLICY_CHANNELS * BOARD_HEIGHT * BOARD_WIDTH * sizeof(float);

    // Allocate GPU Device Memory
    cudaMalloc(&m_deviceObsBuffer, inputSize);
    cudaMalloc(&m_deviceValueBuffer, valSize);
    cudaMalloc(&m_devicePolicyABuffer, polSize);
    cudaMalloc(&m_devicePolicyBBuffer, polSize);

    // Bind Tensor Addresses
    m_context->setTensorAddress(inputName, m_deviceObsBuffer);
    m_context->setTensorAddress(m_engine->getIOTensorName(1), m_deviceValueBuffer);
    m_context->setTensorAddress(m_engine->getIOTensorName(2), m_devicePolicyABuffer);
    m_context->setTensorAddress(m_engine->getIOTensorName(3), m_devicePolicyBBuffer);

    return true;
}

bool Engine::runInference(float* obs, float* value, float* piA, float* piB) {
    std::lock_guard<std::mutex> lock(m_inferenceMutex);
    cudaSetDevice(m_deviceId);

    size_t inputSize = m_batchSize * NB_INPUT_CHANNELS * BOARD_HEIGHT * BOARD_WIDTH * sizeof(float);
    size_t valSize = m_batchSize * sizeof(float);
    size_t polSize = m_batchSize * NB_POLICY_CHANNELS * BOARD_HEIGHT * BOARD_WIDTH * sizeof(float);

    // Step 1: Upload Input (Async)
    cudaMemcpyAsync(m_deviceObsBuffer, obs, inputSize, cudaMemcpyHostToDevice, m_cudaStream);

    // Step 2: GPU Compute via CUDA Graph
    if (!m_graphCreated) {
        // Warmup execution to initialize internal TRT states
        m_context->enqueueV3(m_cudaStream);
        
        // Capture the kernel sequence
        cudaStreamBeginCapture(m_cudaStream, cudaStreamCaptureModeGlobal);
        m_context->enqueueV3(m_cudaStream); 
        cudaStreamEndCapture(m_cudaStream, &m_graph);
        
        // Instantiate the executable graph
        cudaGraphInstantiate(&m_instance, m_graph, 0);
        m_graphCreated = true;
    }
    cudaGraphLaunch(m_instance, m_cudaStream);

    // Step 3: Download Outputs (Async)
    cudaMemcpyAsync(value, m_deviceValueBuffer, valSize, cudaMemcpyDeviceToHost, m_cudaStream);
    cudaMemcpyAsync(piA, m_devicePolicyABuffer, polSize, cudaMemcpyDeviceToHost, m_cudaStream);
    cudaMemcpyAsync(piB, m_devicePolicyBBuffer, polSize, cudaMemcpyDeviceToHost, m_cudaStream);

    // Step 4: Final Synchronize
    cudaStreamSynchronize(m_cudaStream);

    return true;
}