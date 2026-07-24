#include "engine.h"
#include <algorithm>
#include <cctype>
#include <iostream>
#include <fstream>
#include <vector>

namespace {

bool checkCuda(cudaError_t result, const char* operation) {
    if (result == cudaSuccess) {
        return true;
    }
    std::cerr << operation << " failed: " << cudaGetErrorString(result) << std::endl;
    return false;
}

size_t elementsPerBatch(const nvinfer1::Dims& dims) {
    if (dims.nbDims == 0) {
        return 1;
    }

    size_t elements = 1;
    for (int i = 1; i < dims.nbDims; ++i) {
        if (dims.d[i] <= 0) {
            return 0;
        }
        elements *= static_cast<size_t>(dims.d[i]);
    }
    return elements;
}

}  // namespace

Engine::Engine(int deviceId, int batchSize)
    : m_deviceId(deviceId), m_batchSize(std::max(1, batchSize)) {
    if (!checkCuda(cudaSetDevice(m_deviceId), "cudaSetDevice") ||
        !checkCuda(cudaStreamCreate(&m_cudaStream), "cudaStreamCreate")) {
        m_cudaStream = nullptr;
    }
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
    nvinfer1::Dims4 dims{m_batchSize, NB_INPUT_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH};
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
    if (!m_engine || !m_cudaStream) return false;
    m_context.reset(m_engine->createExecutionContext());
    if (!m_context) return false;

    std::vector<std::string> policyNames;
    for (int i = 0; i < m_engine->getNbIOTensors(); ++i) {
        const char* tensorName = m_engine->getIOTensorName(i);
        if (!tensorName) return false;

        if (m_engine->getTensorDataType(tensorName) != nvinfer1::DataType::kFLOAT) {
            std::cerr << "TensorRT I/O tensor must use FP32: " << tensorName << std::endl;
            return false;
        }

        nvinfer1::Dims tensorDims = m_engine->getTensorShape(tensorName);
        if (m_engine->getTensorIOMode(tensorName) == nvinfer1::TensorIOMode::kINPUT) {
            if (!m_inputName.empty()) {
                std::cerr << "Expected exactly one TensorRT input tensor" << std::endl;
                return false;
            }
            m_inputName = tensorName;
        } else {
            size_t outputElements = elementsPerBatch(tensorDims);
            if (outputElements == 1 && m_valueName.empty()) {
                m_valueName = tensorName;
            } else if (outputElements == NB_POLICY_VALUES()) {
                policyNames.emplace_back(tensorName);
            } else {
                std::cerr << "Unexpected TensorRT output shape for tensor " << tensorName << std::endl;
                return false;
            }
        }
    }

    if (m_inputName.empty() || m_valueName.empty() || policyNames.size() != 2) {
        std::cerr << "TensorRT model must expose one input, one value output, and two policy outputs" << std::endl;
        return false;
    }

    auto normalizedName = [](std::string name) {
        std::transform(name.begin(), name.end(), name.begin(), [](unsigned char character) {
            return static_cast<char>(std::tolower(character));
        });
        return name;
    };
    for (const std::string& policyName : policyNames) {
        std::string lowerName = normalizedName(policyName);
        if (lowerName.find("policy_a") != std::string::npos ||
            lowerName.find("policya") != std::string::npos ||
            lowerName.find("pi_a") != std::string::npos) {
            m_policyAName = policyName;
        } else if (lowerName.find("policy_b") != std::string::npos ||
                   lowerName.find("policyb") != std::string::npos ||
                   lowerName.find("pi_b") != std::string::npos) {
            m_policyBName = policyName;
        }
    }
    if (m_policyAName.empty() || m_policyBName.empty()) {
        std::cerr << "TensorRT policy head names are ambiguous; using engine output order" << std::endl;
        m_policyAName = policyNames[0];
        m_policyBName = policyNames[1];
    }

    nvinfer1::Dims inputDims = m_engine->getTensorShape(m_inputName.c_str());
    if (inputDims.nbDims != 4 ||
        (inputDims.d[1] > 0 && inputDims.d[1] != NB_INPUT_CHANNELS) ||
        (inputDims.d[2] > 0 && inputDims.d[2] != BOARD_HEIGHT) ||
        (inputDims.d[3] > 0 && inputDims.d[3] != BOARD_WIDTH)) {
        std::cerr << "Unexpected TensorRT input shape" << std::endl;
        return false;
    }

    m_batchSize = inputDims.d[0] > 0 ? inputDims.d[0] : SearchParams::BATCH_SIZE;
    nvinfer1::Dims4 dims{m_batchSize, NB_INPUT_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH};
    if (!m_context->setInputShape(m_inputName.c_str(), dims)) {
        std::cerr << "Failed to set TensorRT input shape" << std::endl;
        return false;
    }

    size_t inputSize = m_batchSize * NB_INPUT_CHANNELS * BOARD_HEIGHT * BOARD_WIDTH * sizeof(float);
    size_t valSize = m_batchSize * sizeof(float);
    size_t polSize = m_batchSize * NB_POLICY_CHANNELS * BOARD_HEIGHT * BOARD_WIDTH * sizeof(float);

    // Allocate GPU Device Memory
    if (!checkCuda(cudaMalloc(&m_deviceObsBuffer, inputSize), "cudaMalloc(input)") ||
        !checkCuda(cudaMalloc(&m_deviceValueBuffer, valSize), "cudaMalloc(value)") ||
        !checkCuda(cudaMalloc(&m_devicePolicyABuffer, polSize), "cudaMalloc(policy A)") ||
        !checkCuda(cudaMalloc(&m_devicePolicyBBuffer, polSize), "cudaMalloc(policy B)")) {
        return false;
    }

    // Bind Tensor Addresses
    if (!m_context->setTensorAddress(m_inputName.c_str(), m_deviceObsBuffer) ||
        !m_context->setTensorAddress(m_valueName.c_str(), m_deviceValueBuffer) ||
        !m_context->setTensorAddress(m_policyAName.c_str(), m_devicePolicyABuffer) ||
        !m_context->setTensorAddress(m_policyBName.c_str(), m_devicePolicyBBuffer)) {
        std::cerr << "Failed to bind TensorRT tensor addresses" << std::endl;
        return false;
    }

    return true;
}

bool Engine::runInference(float* obs, float* value, float* piA, float* piB) {
    std::lock_guard<std::mutex> lock(m_inferenceMutex);
    if (!obs || !value || !piA || !piB || !m_context || !m_cudaStream ||
        !checkCuda(cudaSetDevice(m_deviceId), "cudaSetDevice")) {
        return false;
    }

    size_t inputSize = m_batchSize * NB_INPUT_CHANNELS * BOARD_HEIGHT * BOARD_WIDTH * sizeof(float);
    size_t valSize = m_batchSize * sizeof(float);
    size_t polSize = m_batchSize * NB_POLICY_CHANNELS * BOARD_HEIGHT * BOARD_WIDTH * sizeof(float);

    // Step 1: Upload Input (Async)
    if (!checkCuda(cudaMemcpyAsync(m_deviceObsBuffer, obs, inputSize, cudaMemcpyHostToDevice, m_cudaStream),
                   "cudaMemcpyAsync(input)")) {
        return false;
    }

    // Step 2: GPU Compute via CUDA Graph
    if (!m_graphCreated) {
        // Warmup execution to initialize internal TRT states
        if (!m_context->enqueueV3(m_cudaStream) ||
            !checkCuda(cudaStreamSynchronize(m_cudaStream), "TensorRT warmup")) {
            return false;
        }
        
        // Capture the kernel sequence
        if (!checkCuda(cudaStreamBeginCapture(m_cudaStream, cudaStreamCaptureModeGlobal), "cudaStreamBeginCapture") ||
            !m_context->enqueueV3(m_cudaStream) ||
            !checkCuda(cudaStreamEndCapture(m_cudaStream, &m_graph), "cudaStreamEndCapture")) {
            return false;
        }
        
        // Instantiate the executable graph
        if (!checkCuda(cudaGraphInstantiate(&m_instance, m_graph, 0), "cudaGraphInstantiate")) {
            return false;
        }
        m_graphCreated = true;
    }
    if (!checkCuda(cudaGraphLaunch(m_instance, m_cudaStream), "cudaGraphLaunch")) {
        return false;
    }

    // Step 3: Download Outputs (Async)
    if (!checkCuda(cudaMemcpyAsync(value, m_deviceValueBuffer, valSize, cudaMemcpyDeviceToHost, m_cudaStream),
                   "cudaMemcpyAsync(value)") ||
        !checkCuda(cudaMemcpyAsync(piA, m_devicePolicyABuffer, polSize, cudaMemcpyDeviceToHost, m_cudaStream),
                   "cudaMemcpyAsync(policy A)") ||
        !checkCuda(cudaMemcpyAsync(piB, m_devicePolicyBBuffer, polSize, cudaMemcpyDeviceToHost, m_cudaStream),
                   "cudaMemcpyAsync(policy B)")) {
        return false;
    }

    // Step 4: Final Synchronize
    if (!checkCuda(cudaStreamSynchronize(m_cudaStream), "cudaStreamSynchronize")) {
        return false;
    }

    return true;
}