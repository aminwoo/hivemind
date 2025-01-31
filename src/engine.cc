#include "engine.h"

Engine::Engine() {
    // Initialize CUDA (if needed)
    cudaSetDevice(0);
    cudaStreamCreate(&m_cudaStream); // Create CUDA stream once
}

Engine::~Engine() {
    // Clean up CUDA resources
    if (m_cudaStream) {
        cudaStreamDestroy(m_cudaStream);
    }
    if (m_inputBuffer) {
        cudaFree(m_inputBuffer);
    }
    if (m_valueOutputBuffer) {
        cudaFree(m_valueOutputBuffer);
    }
    if (m_policyOutputBuffer) {
        cudaFree(m_policyOutputBuffer);
    }
    m_context.reset();
    m_engine.reset();
}

bool Engine::loadNetwork(const std::string& onnxFile, const std::string& engineFile) {
    // Check if engine file exists
    std::ifstream checkFile(engineFile, std::ios::binary);
    bool engineExists = checkFile.good();
    checkFile.close();

    if (engineExists) {
        std::cout << "Loading existing engine file: " << engineFile << std::endl;
        return loadEngineFromFile(engineFile);
    } else {
        std::cout << "Building new engine from ONNX file: " << onnxFile << std::endl;
        return buildEngineFromONNX(onnxFile) && saveEngineToFile(engineFile);
    }
}

bool Engine::buildEngineFromONNX(const std::string& onnxFile) {
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(m_logger));
    if (!builder) {
        std::cerr << "Failed to create builder" << std::endl;
        return false;
    }

    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0));
    if (!network) {
        std::cerr << "Failed to create network" << std::endl;
        return false;
    }

    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, m_logger));
    if (!parser->parseFromFile(onnxFile.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        std::cerr << "Failed to parse ONNX file" << std::endl;
        for (int i = 0; i < parser->getNbErrors(); ++i) {
            std::cerr << "Parser error: " << parser->getError(i)->desc() << std::endl;
        }
        return false;
    }

    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1 << 20);

    if (builder->platformHasFastFp16()) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }

    auto profile = builder->createOptimizationProfile();
    const char* inputTensorName = network->getInput(0)->getName();
    profile->setDimensions(inputTensorName, nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4{1, 32, 8, 16});
    profile->setDimensions(inputTensorName, nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4{1, 32, 8, 16});
    profile->setDimensions(inputTensorName, nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4{1, 32, 8, 16});
    config->addOptimizationProfile(profile);

    m_engine.reset(builder->buildEngineWithConfig(*network, *config));
    if (!m_engine) {
        std::cerr << "Failed to build engine" << std::endl;
        return false;
    }

    // Create execution context and allocate GPU memory
    return initializeResources();
}

bool Engine::loadEngineFromFile(const std::string& engineFile) {
    std::ifstream file(engineFile, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open engine file: " << engineFile << std::endl;
        return false;
    }

    std::vector<char> engineData((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();

    auto runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(m_logger));
    m_engine.reset(runtime->deserializeCudaEngine(engineData.data(), engineData.size()));
    if (!m_engine) {
        std::cerr << "Failed to deserialize engine" << std::endl;
        return false;
    }

    // Create execution context and allocate GPU memory
    return initializeResources();
}

bool Engine::saveEngineToFile(const std::string& engineFile) {
    auto serializedEngine = std::unique_ptr<nvinfer1::IHostMemory>(m_engine->serialize());
    if (!serializedEngine) {
        std::cerr << "Failed to serialize engine" << std::endl;
        return false;
    }

    std::ofstream file(engineFile, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to create engine file: " << engineFile << std::endl;
        return false;
    }

    file.write((char*)serializedEngine->data(), serializedEngine->size());
    file.close();
    return true;
}

bool Engine::initializeResources() {
    if (!m_engine) {
        std::cerr << "Engine not loaded" << std::endl;
        return false;
    }

    // Create execution context
    m_context.reset(m_engine->createExecutionContext());
    if (!m_context) {
        std::cerr << "Failed to create execution context" << std::endl;
        return false;
    }

    // Define input and output sizes
    const int batchSize = 1;
    const int inputChannels = 32;
    const int inputHeight = 8;
    const int inputWidth = 16;
    const int inputSize = batchSize * inputChannels * inputHeight * inputWidth;

    const int valueOutputSize = 1;
    const int policyOutputChannels = 73;
    const int policyOutputHeight = 8;
    const int policyOutputWidth = 16;
    const int policyOutputSize = batchSize * policyOutputChannels * policyOutputHeight * policyOutputWidth;

    // Allocate GPU memory
    cudaMalloc(&m_inputBuffer, inputSize * sizeof(float)); // Input
    cudaMalloc(&m_valueOutputBuffer, valueOutputSize * sizeof(float)); // Value Output
    cudaMalloc(&m_policyOutputBuffer, policyOutputSize * sizeof(float)); // Policy Output

    // Set tensor addresses
    const char* inputTensorName = m_engine->getIOTensorName(0);
    const char* valueOutputTensorName = m_engine->getIOTensorName(1);
    const char* policyOutputTensorName = m_engine->getIOTensorName(2);

    m_context->setTensorAddress(inputTensorName, m_inputBuffer);
    m_context->setTensorAddress(valueOutputTensorName, m_valueOutputBuffer);
    m_context->setTensorAddress(policyOutputTensorName, m_policyOutputBuffer);

    return true;
}

bool Engine::runInference(float* inputPlanes, float* valueOutput, float* policyOutput) {
    if (!m_engine || !m_context) {
        std::cerr << "Engine or context not loaded" << std::endl;
        return false;
    }

    // Define input and output sizes
    const int batchSize = 1;
    const int inputChannels = 32;
    const int inputHeight = 8;
    const int inputWidth = 16;
    const int inputSize = batchSize * inputChannels * inputHeight * inputWidth;

    const int valueOutputSize = 1;
    const int policyOutputChannels = 73;
    const int policyOutputHeight = 8;
    const int policyOutputWidth = 16;
    const int policyOutputSize = batchSize * policyOutputChannels * policyOutputHeight * policyOutputWidth;

    // Perform inference
    cudaMemcpyAsync(m_inputBuffer, inputPlanes, inputSize * sizeof(float), cudaMemcpyHostToDevice, m_cudaStream);
    m_context->enqueueV3(m_cudaStream);
    cudaMemcpyAsync(valueOutput, m_valueOutputBuffer, valueOutputSize * sizeof(float), cudaMemcpyDeviceToHost, m_cudaStream);
    cudaMemcpyAsync(policyOutput, m_policyOutputBuffer, policyOutputSize * sizeof(float), cudaMemcpyDeviceToHost, m_cudaStream);
    cudaStreamSynchronize(m_cudaStream);

    return true;
}
