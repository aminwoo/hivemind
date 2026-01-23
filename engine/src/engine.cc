#include "engine.h"

Engine::Engine(int deviceId) : m_deviceId(deviceId) {
    // Set the current CUDA device to the one passed in
    cudaSetDevice(m_deviceId);
    // Create a CUDA stream on this device
    cudaStreamCreate(&m_cudaStream);
}

Engine::~Engine() {
    // Clean up CUDA resources
    if (m_cudaStream) {
        cudaStreamDestroy(m_cudaStream);
    }
    if (m_obsBuffer) {
        cudaFree(m_obsBuffer);
    }
    if (m_valueBuffer) {
        cudaFree(m_valueBuffer);
    }
    if (m_policyABuffer) {
        cudaFree(m_policyABuffer);
    }
    if (m_policyBBuffer) {
        cudaFree(m_policyBBuffer);
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
    profile->setDimensions(inputTensorName, nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4{BATCH_SIZE, NB_INPUT_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH});
    profile->setDimensions(inputTensorName, nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4{BATCH_SIZE, NB_INPUT_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH});
    profile->setDimensions(inputTensorName, nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4{BATCH_SIZE, NB_INPUT_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH});
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
    const int batchSize = BATCH_SIZE;
    const int inputChannels = NB_INPUT_CHANNELS;
    const int inputHeight = BOARD_HEIGHT;
    const int inputWidth = BOARD_WIDTH;
    const int inputSize = batchSize * inputChannels * inputHeight * inputWidth;

    const int valueOutputSize = batchSize;
    const int policyOutputChannels = NB_POLICY_CHANNELS;
    const int policyOutputHeight = BOARD_HEIGHT;
    const int policyOutputWidth = BOARD_WIDTH;
    const int policyOutputSize = batchSize * policyOutputChannels * policyOutputHeight * policyOutputWidth;

    // Allocate GPU memory
    cudaMalloc(&m_obsBuffer, inputSize * sizeof(float));        // Input
    cudaMalloc(&m_valueBuffer, valueOutputSize * sizeof(float)); // Value Output
    cudaMalloc(&m_policyABuffer, policyOutputSize * sizeof(float)); // Policy Output A
    cudaMalloc(&m_policyBBuffer, policyOutputSize * sizeof(float)); // Policy Output B

    // Set tensor addresses
    const char* inputTensorName = m_engine->getIOTensorName(0);
    const char* valueOutputTensorName = m_engine->getIOTensorName(1);
    const char* policyOutputTensorName = m_engine->getIOTensorName(2);
    const char* policyBOutputTensorName = m_engine->getIOTensorName(3);

    m_context->setTensorAddress(inputTensorName, m_obsBuffer);
    m_context->setTensorAddress(valueOutputTensorName, m_valueBuffer);
    m_context->setTensorAddress(policyOutputTensorName, m_policyABuffer);
    m_context->setTensorAddress(policyBOutputTensorName, m_policyBBuffer);

    return true;
}

bool Engine::runInference(float* obs, float* value, float* piA, float* piB) {
    // Thread-safe: serialize inference calls
    std::lock_guard<std::mutex> lock(m_inferenceMutex);
    
    if (!m_engine || !m_context) {
        std::cerr << "Engine or context not loaded" << std::endl;
        return false;
    }

    // Ensure the correct device is set for this thread.
    cudaSetDevice(m_deviceId);

    // Define input and output sizes
    const int batchSize = BATCH_SIZE;
    const int inputChannels = NB_INPUT_CHANNELS;
    const int inputHeight = BOARD_HEIGHT;
    const int inputWidth = BOARD_WIDTH;
    const int inputSize = batchSize * inputChannels * inputHeight * inputWidth;

    const int valueOutputSize = batchSize;
    const int policyOutputChannels = NB_POLICY_CHANNELS;
    const int policyOutputHeight = BOARD_HEIGHT;
    const int policyOutputWidth = BOARD_WIDTH;
    const int policyOutputSize = batchSize * policyOutputChannels * policyOutputHeight * policyOutputWidth;

    // Perform inference
    cudaMemcpyAsync(m_obsBuffer, obs, inputSize * sizeof(float), cudaMemcpyHostToDevice, m_cudaStream);
    m_context->enqueueV3(m_cudaStream);
    cudaMemcpyAsync(value, m_valueBuffer, valueOutputSize * sizeof(float), cudaMemcpyDeviceToHost, m_cudaStream);
    cudaMemcpyAsync(piA, m_policyABuffer, policyOutputSize * sizeof(float), cudaMemcpyDeviceToHost, m_cudaStream);
    cudaMemcpyAsync(piB, m_policyBBuffer, policyOutputSize * sizeof(float), cudaMemcpyDeviceToHost, m_cudaStream);
    cudaStreamSynchronize(m_cudaStream);

    return true;
}
