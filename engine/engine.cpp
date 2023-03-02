#include <iostream>
#include <fstream>

#include "engine.h"
#include "NvOnnxParser.h"

void Logger::log(Severity severity, const char *msg) noexcept {
    // Would advise using a proper logging utility such as https://github.com/gabime/spdlog
    // For the sake of this tutorial, will just log to the console.

    // Only log Warnings or more important.
    if (severity <= Severity::kWARNING) {
        //std::cout << msg << std::endl;
    }
}

bool Engine::doesFileExist(const std::string &filepath) {
    std::ifstream f(filepath.c_str());
    return f.good();
}

bool Engine::build(std::string onnxModelPath, std::string engineName) {
    // Only regenerate the engine file if it has not already been generated for the specified options
    m_engineName = engineName;

    if (doesFileExist(m_engineName)) {
        return true;
    }

    // Was not able to find the engine file, generate...
    std::cout << "Engine not found, generating..." << std::endl;

    
    // Create our engine builder.
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(m_logger));
    builder->setMaxBatchSize(1);

    // Define an explicit batch size and then create the network.
    // More info here: https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#explicit-implicit-batch
    auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));

    // Create a parser for reading the onnx file.
    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, m_logger));

    // We are going to first read the onnx file into memory, then pass that buffer to the parser.
    // Had our onnx model file been encrypted, this approach would allow us to first decrypt the buffer.
    std::ifstream file(onnxModelPath, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        throw std::runtime_error("Unable to read engine file");
    }

    auto parsed = parser->parse(buffer.data(), buffer.size());
    if (!parsed) {
        return false;
    }

    // Save the input height, width, and channels.
    // Require this info for inference.
    const auto input = network->getInput(0);
    //const auto output = network->getOutput(0);
    const auto inputName = input->getName();
    const auto inputDims = input->getDimensions();
    int32_t inputC = inputDims.d[1];
    int32_t inputH = inputDims.d[2];
    int32_t inputW = inputDims.d[3];

    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());

    IOptimizationProfile* profile = builder->createOptimizationProfile();
    profile->setDimensions(inputName, OptProfileSelector::kMIN, Dims4(1, inputC, inputH, inputW));
    profile->setDimensions(inputName, OptProfileSelector::kOPT, Dims4(batchSize, inputC, inputH, inputW));
    profile->setDimensions(inputName, OptProfileSelector::kMAX, Dims4(batchSize, inputC, inputH, inputW));
    config->addOptimizationProfile(profile);
    config->setMaxWorkspaceSize(4000000000);

    // CUDA stream used for profiling by the builder.
    auto profileStream = samplesCommon::makeCudaStream();
    config->setProfileStream(*profileStream);

    // Build the engine
    std::unique_ptr<IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};

    // Write the engine to disk
    std::ofstream outfile(m_engineName, std::ofstream::binary);
    outfile.write(reinterpret_cast<const char*>(plan->data()), plan->size());

    std::cout << "Success, saved engine to " << m_engineName << std::endl;

    return true;
}

Engine::~Engine() {
    if (m_cudaStream) {
        cudaStreamDestroy(m_cudaStream);
    }
}

bool Engine::loadNetwork() {
    // Read the serialized model from disk
    std::ifstream file(m_engineName, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        throw std::runtime_error("Unable to read engine file");
    }

    std::unique_ptr<IRuntime> runtime{createInferRuntime(m_logger)};
    if (!runtime) {
        return false;
    }

    // Set the device index
    auto ret = cudaSetDevice(0);
    if (ret != 0) {
        int numGPUs;
        cudaGetDeviceCount(&numGPUs);
        auto errMsg = "Unable to set GPU device index to: " + std::to_string(0) +
                ". Note, your device has " + std::to_string(numGPUs) + " CUDA-capable GPU(s).";
        throw std::runtime_error(errMsg);
    }

    m_engine = std::unique_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(buffer.data(), buffer.size()));
    if (!m_engine) {
        return false;
    }

    m_context = std::unique_ptr<nvinfer1::IExecutionContext>(m_engine->createExecutionContext());
    if (!m_context) {
        return false;
    }

    auto cudaRet = cudaStreamCreate(&m_cudaStream);
    if (cudaRet != 0) {
        throw std::runtime_error("Unable to create cuda stream");
    }

    return true;
}

void Engine::bind_executor_input(int channels) {
    std::string inputLayerName("input_planes");
    idxInput = m_engine->getBindingIndex(inputLayerName.c_str());
    memorySizes[idxInput] = batchSize * channels * sizeof(float);
    cudaMalloc(&deviceMemory[idxInput], memorySizes[idxInput]);
}

void Engine::bind_executor_policy(int channels) {
    std::string policyOutputName("policy_output");
    idxPolicyOutput = m_engine->getBindingIndex(policyOutputName.c_str());
    memorySizes[idxPolicyOutput] = batchSize * channels * sizeof(float);
    cudaMalloc(&deviceMemory[idxPolicyOutput], memorySizes[idxPolicyOutput]);
}

void Engine::bind_executor_value(int channels) {
    std::string valueOutputName("value_output");
    idxValueOutput = m_engine->getBindingIndex(valueOutputName.c_str());
    memorySizes[idxValueOutput] = batchSize * channels * sizeof(float);
    cudaMalloc(&deviceMemory[idxValueOutput], memorySizes[idxValueOutput]);
}

void Engine::predict(float* inputPlanes, float* valueOutput, float* policyOutput) {
    auto dims = m_engine->getBindingDimensions(0);
    Dims4 inputDims = {batchSize, dims.d[1], dims.d[2], dims.d[3]};
    m_context->setBindingDimensions(0, inputDims);

    // copy input planes from host to device
    cudaMemcpyAsync(deviceMemory[idxInput], inputPlanes, memorySizes[idxInput],
                          cudaMemcpyHostToDevice, m_cudaStream);

    // run inference for given data
    m_context->enqueueV2(deviceMemory, m_cudaStream, nullptr);

    // copy output from device back to host
    cudaMemcpyAsync(valueOutput, deviceMemory[idxValueOutput],
                          memorySizes[idxValueOutput], cudaMemcpyDeviceToHost, m_cudaStream);
    cudaMemcpyAsync(policyOutput, deviceMemory[idxPolicyOutput],
                          memorySizes[idxPolicyOutput], cudaMemcpyDeviceToHost, m_cudaStream);

    cudaStreamSynchronize(m_cudaStream);
}

void Engine::getGPUUUIDs(std::vector<std::string>& gpuUUIDs) {
    int numGPUs;
    cudaGetDeviceCount(&numGPUs);

    for (int device=0; device<numGPUs; device++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);

        char uuid[33];
        for(int b=0; b<16; b++) {
            sprintf(&uuid[b*2], "%02x", (unsigned char)prop.uuid.bytes[b]);
        }

        gpuUUIDs.push_back(std::string(uuid));
        // by comparing uuid against a preset list of valid uuids given by the client (using: nvidia-smi -L) we decide which gpus can be used.
    }
}