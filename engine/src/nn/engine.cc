#include "nn/engine.h"
#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <cuda_fp16.h>
#include <dlfcn.h>
#include <iostream>
#include <fstream>
#include <mutex>
#include <sstream>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

#include "nn/onnx_utils.h"

namespace {

constexpr int BUILDER_OPTIMIZATION_LEVEL = 5;
constexpr std::string_view ENGINE_CACHE_SCHEMA = "hivemind-trt-cache-v4";
constexpr std::string_view FP16_CONVERTER_SCHEMA = "onnxruntime-fp16-v1";

std::string engineBuildDescriptor(int deviceId, int batchSize) {
    cudaDeviceProp deviceProperties{};
    const cudaError_t propertiesResult =
        cudaGetDeviceProperties(&deviceProperties, deviceId);

    std::ostringstream descriptor;
    descriptor << ENGINE_CACHE_SCHEMA
               << "|tensorrt=" << NV_TENSORRT_VERSION
               << "|cudart=" << CUDART_VERSION
               << "|device=" << deviceId
               << "|batch=" << batchSize
               << "|profiles=" << SearchParams::NUM_SEARCH_THREADS
               << "|optimization=" << BUILDER_OPTIMIZATION_LEVEL
               << "|workspace=default"
               << "|converter=" << FP16_CONVERTER_SCHEMA;
    if (propertiesResult == cudaSuccess) {
        descriptor << "|gpu=" << deviceProperties.name
                   << "|compute=" << deviceProperties.major
                   << '.' << deviceProperties.minor;
    }
    return descriptor.str();
}

std::filesystem::path cacheMetadataPath(const std::string& engineFile) {
    return std::filesystem::path(engineFile).concat(".meta");
}

bool cacheSignatureMatches(const std::string& engineFile,
                           const std::string& expectedSignature) {
    std::ifstream metadata(cacheMetadataPath(engineFile));
    std::string storedSignature;
    return metadata && std::getline(metadata, storedSignature)
        && storedSignature == expectedSignature;
}

bool writeCacheMetadata(const std::string& engineFile,
                        const std::string& signature,
                        const std::string& descriptor) {
    std::ofstream metadata(
        cacheMetadataPath(engineFile), std::ios::trunc);
    if (!metadata) {
        return false;
    }
    metadata << signature << '\n' << descriptor << '\n';
    return metadata.good();
}

void preloadTensorRTBuilderResources() {
    static std::once_flag preloadOnce;
    std::call_once(preloadOnce, []() {
        namespace fs = std::filesystem;
        std::error_code errorCode;
        const fs::path libraryDir(TENSORRT_LIBRARY_DIR);
        if (!fs::exists(libraryDir, errorCode)) {
            std::cerr << "TensorRT library directory not found: " << libraryDir << std::endl;
            return;
        }

        for (const auto& entry : fs::directory_iterator(libraryDir, errorCode)) {
            if (errorCode) {
                break;
            }
            if (!entry.is_regular_file()) {
                continue;
            }

            const std::string fileName = entry.path().filename().string();
            if (fileName.rfind("libnvinfer_builder_resource_", 0) != 0) {
                continue;
            }

            void* handle = dlopen(entry.path().c_str(), RTLD_NOW | RTLD_GLOBAL);
            if (!handle) {
                std::cerr << "Failed to preload TensorRT resource library "
                          << entry.path() << ": " << dlerror() << std::endl;
            }
        }
    });
}

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

bool runFp16Converter(const std::string& python,
                      const std::filesystem::path& input,
                      const std::filesystem::path& output) {
    const pid_t child = fork();
    if (child < 0) {
        return false;
    }
    if (child == 0) {
        execlp(python.c_str(), python.c_str(), HIVEMIND_FP16_CONVERTER_SCRIPT,
               input.c_str(), output.c_str(), static_cast<char*>(nullptr));
        _exit(127);
    }

    int status = 0;
    return waitpid(child, &status, 0) == child &&
           WIFEXITED(status) && WEXITSTATUS(status) == 0;
}

bool convertOnnxToFp16(const std::filesystem::path& input,
                       const std::filesystem::path& output) {
    std::vector<std::string> pythonCandidates;
    if (const char* configuredPython = std::getenv("HIVEMIND_PYTHON")) {
        pythonCandidates.emplace_back(configuredPython);
    }
    const std::filesystem::path workspacePython =
        std::filesystem::path(HIVEMIND_WORKSPACE_ROOT) / ".venv/bin/python";
    if (std::filesystem::is_regular_file(workspacePython)) {
        pythonCandidates.push_back(workspacePython.string());
    }
    pythonCandidates.emplace_back("python3");

    for (const std::string& python : pythonCandidates) {
        if (runFp16Converter(python, input, output)) {
            return true;
        }
    }
    std::cerr << "Failed to convert ONNX model to FP16. Set HIVEMIND_PYTHON to a "
                 "Python environment containing onnx and onnxruntime."
              << std::endl;
    return false;
}

void floatsToHalves(const float* source, void* destination, size_t count) {
    auto* halves = static_cast<__half*>(destination);
    for (size_t i = 0; i < count; ++i) {
        halves[i] = __float2half_rn(source[i]);
    }
}

void halvesToFloats(const void* source, float* destination, size_t count) {
    const auto* halves = static_cast<const __half*>(source);
    for (size_t i = 0; i < count; ++i) {
        destination[i] = __half2float(halves[i]);
    }
}

}  // namespace

Engine::Engine(int deviceId, int batchSize)
    : m_deviceId(deviceId), m_batchSize(std::max(1, batchSize)) {
    checkCuda(cudaSetDevice(m_deviceId), "cudaSetDevice");
}

Engine::ExecutionState::~ExecutionState() {
    if (stream) cudaStreamSynchronize(stream);
    if (graphInstance) cudaGraphExecDestroy(graphInstance);
    if (graph) cudaGraphDestroy(graph);
    context.reset();
    if (deviceObsBuffer) cudaFree(deviceObsBuffer);
    if (deviceValueBuffer) cudaFree(deviceValueBuffer);
    if (devicePolicyABuffer) cudaFree(devicePolicyABuffer);
    if (devicePolicyBBuffer) cudaFree(devicePolicyBBuffer);
    if (deviceWdlBuffer) cudaFree(deviceWdlBuffer);
    if (deviceMovesLeftBuffer) cudaFree(deviceMovesLeftBuffer);
    if (hostObsHalf) cudaFreeHost(hostObsHalf);
    if (hostValueHalf) cudaFreeHost(hostValueHalf);
    if (hostPolicyAHalf) cudaFreeHost(hostPolicyAHalf);
    if (hostPolicyBHalf) cudaFreeHost(hostPolicyBHalf);
    if (hostWdlHalf) cudaFreeHost(hostWdlHalf);
    if (hostMovesLeftHalf) cudaFreeHost(hostMovesLeftHalf);
    if (stream) cudaStreamDestroy(stream);
}

Engine::~Engine() {
    cudaSetDevice(m_deviceId);
    m_executionStates.clear();
    m_engine.reset();
}

bool Engine::loadNetwork(const std::string& onnxFile, const std::string& engineFile) {
    const std::string buildDescriptor = engineBuildDescriptor(m_deviceId, m_batchSize);
    const std::string cacheSignature =
        computeFileSignature(onnxFile, buildDescriptor);
    if (cacheSignature.empty()) {
        std::cerr << "Failed to fingerprint ONNX model: " << onnxFile << std::endl;
        return false;
    }

    std::ifstream checkFile(engineFile, std::ios::binary);
    if (checkFile.good()) {
        checkFile.close();
        if (cacheSignatureMatches(engineFile, cacheSignature) &&
            loadEngineFromFile(engineFile)) {
            std::cout << "Loaded TensorRT engine: " << engineFile << std::endl;
            return true;
        }
        std::cout << "TensorRT cache is missing, stale, or incompatible; "
                     "rebuilding from ONNX"
                  << std::endl;
    }
    if (!buildEngineFromONNX(onnxFile) || !saveEngineToFile(engineFile)) {
        return false;
    }
    if (!writeCacheMetadata(
            engineFile, cacheSignature, buildDescriptor)) {
        std::cerr << "Failed to write TensorRT cache metadata for "
                  << engineFile << std::endl;
    }
    std::cout << "Built and loaded TensorRT engine: " << engineFile << std::endl;
    return true;
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
    if (!runtime) {
        std::cerr << "Failed to create TensorRT runtime" << std::endl;
        return false;
    }
    m_engine.reset(runtime->deserializeCudaEngine(engineData.data(), engineData.size()));
    if (!m_engine) {
        std::cerr << "Failed to deserialize TensorRT engine: " << engineFile << std::endl;
        return false;
    }
    
    return initializeResources();
}

bool Engine::saveEngineToFile(const std::string& engineFile) {
    if (!m_engine) {
        std::cerr << "Cannot save an empty TensorRT engine" << std::endl;
        return false;
    }
    auto serializedEngine = std::unique_ptr<nvinfer1::IHostMemory>(m_engine->serialize());
    if (!serializedEngine) return false;

    std::ofstream file(engineFile, std::ios::binary);
    if (!file) return false;

    file.write((char*)serializedEngine->data(), serializedEngine->size());
    return true;
}

bool Engine::buildEngineFromONNX(const std::string& onnxFile) {
    std::cout << "Building TensorRT engine from ONNX: " << onnxFile << std::endl;
    static std::atomic_uint64_t conversionId{0};
    const std::filesystem::path convertedOnnx =
        std::filesystem::temp_directory_path() /
        ("hivemind-fp16-" + std::to_string(getpid()) + "-" +
         std::to_string(conversionId.fetch_add(1)) + ".onnx");
    struct ConvertedOnnxCleanup {
        std::filesystem::path path;
        ~ConvertedOnnxCleanup() {
            std::error_code error;
            std::filesystem::remove(path, error);
        }
    } cleanup{convertedOnnx};
    if (!convertOnnxToFp16(onnxFile, convertedOnnx)) {
        return false;
    }

    preloadTensorRTBuilderResources();
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(m_logger));
    if (!builder) {
        std::cerr << "Failed to create TensorRT builder. Run the generated hivemind launcher "
                  << "so TensorRT resource libraries are on LD_LIBRARY_PATH." << std::endl;
        return false;
    }
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(
        builder->createNetworkV2(0U));
    if (!network) {
        std::cerr << "Failed to create TensorRT network definition" << std::endl;
        return false;
    }
    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, m_logger));
    if (!parser) {
        std::cerr << "Failed to create TensorRT ONNX parser" << std::endl;
        return false;
    }

    if (!parser->parseFromFile(convertedOnnx.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        return false;
    }

    size_t fp16TensorCount = 0;
    for (int layerIndex = 0; layerIndex < network->getNbLayers(); ++layerIndex) {
        nvinfer1::ILayer* layer = network->getLayer(layerIndex);
        for (int outputIndex = 0; outputIndex < layer->getNbOutputs(); ++outputIndex) {
            nvinfer1::ITensor* output = layer->getOutput(outputIndex);
            if (!output) {
                continue;
            }
            if (output->getType() == nvinfer1::DataType::kHALF) {
                fp16TensorCount++;
            } else if (output->getType() == nvinfer1::DataType::kFLOAT) {
                std::cerr << "Refusing to build a non-FP16 TensorRT plan: internal tensor "
                          << output->getName() << " from layer " << layer->getName()
                          << " is FP32 after automatic conversion."
                          << std::endl;
                return false;
            }
        }
    }
    if (fp16TensorCount == 0) {
        std::cerr << "Refusing to build a TensorRT plan without FP16 internal tensors"
                  << std::endl;
        return false;
    }

    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config) {
        std::cerr << "Failed to create TensorRT builder configuration" << std::endl;
        return false;
    }
    
    config->setBuilderOptimizationLevel(BUILDER_OPTIMIZATION_LEVEL);
    
    const char* inputName = network->getInput(0)->getName();
    nvinfer1::Dims dims{};
    dims.nbDims = 4;
    dims.d[0] = m_batchSize;
    dims.d[1] = NB_INPUT_CHANNELS;
    dims.d[2] = BOARD_HEIGHT;
    dims.d[3] = BOARD_WIDTH;
    for (int workerIndex = 0; workerIndex < SearchParams::NUM_SEARCH_THREADS; ++workerIndex) {
        auto profile = builder->createOptimizationProfile();
        if (!profile) {
            std::cerr << "Failed to create TensorRT optimization profile" << std::endl;
            return false;
        }
        profile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMIN, dims);
        profile->setDimensions(inputName, nvinfer1::OptProfileSelector::kOPT, dims);
        profile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMAX, dims);
        if (config->addOptimizationProfile(profile) < 0) {
            std::cerr << "Failed to add TensorRT optimization profile" << std::endl;
            return false;
        }
    }

    // TensorRT 10: use buildSerializedNetwork instead of buildEngineWithConfig
    auto serializedEngine = std::unique_ptr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
    if (!serializedEngine) return false;
    
    auto runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(m_logger));
    if (!runtime) {
        std::cerr << "Failed to create TensorRT runtime" << std::endl;
        return false;
    }
    m_engine.reset(runtime->deserializeCudaEngine(serializedEngine->data(), serializedEngine->size()));
    if (!m_engine) {
        std::cerr << "Failed to deserialize newly built TensorRT engine" << std::endl;
        return false;
    }
    return initializeResources();
}

bool Engine::initializeResources() {
    if (!m_engine ||
        m_engine->getNbOptimizationProfiles() < SearchParams::NUM_SEARCH_THREADS) {
        std::cerr << "TensorRT engine does not contain enough worker profiles" << std::endl;
        return false;
    }

    m_executionStates.clear();
    m_inputName.clear();
    m_valueName.clear();
    m_policyAName.clear();
    m_policyBName.clear();
    m_wdlName.clear();
    m_movesLeftName.clear();

    auto normalizedName = [](std::string name) {
        std::transform(name.begin(), name.end(), name.begin(), [](unsigned char character) {
            return static_cast<char>(std::tolower(character));
        });
        return name;
    };

    for (int i = 0; i < m_engine->getNbIOTensors(); ++i) {
        const char* tensorName = m_engine->getIOTensorName(i);
        if (!tensorName) return false;

        if (m_engine->getTensorDataType(tensorName) != nvinfer1::DataType::kHALF) {
            std::cerr << "TensorRT I/O tensor must use FP16: " << tensorName << std::endl;
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
            const std::string lowerName = normalizedName(tensorName);
            if (lowerName == "value" && outputElements == 1) {
                m_valueName = tensorName;
            } else if ((lowerName == "pi_a" || lowerName == "policy_a") &&
                       outputElements == NB_POLICY_VALUES()) {
                m_policyAName = tensorName;
            } else if ((lowerName == "pi_b" || lowerName == "policy_b") &&
                       outputElements == NB_POLICY_VALUES()) {
                m_policyBName = tensorName;
            } else if ((lowerName == "wdl_out" || lowerName == "wdl") &&
                       outputElements == 3) {
                m_wdlName = tensorName;
            } else if ((lowerName == "moves_left" || lowerName == "plys_to_end_out") &&
                       outputElements == 1) {
                m_movesLeftName = tensorName;
            } else {
                std::cerr << "Unexpected TensorRT output tensor " << tensorName << std::endl;
                return false;
            }
        }
    }

    if (m_inputName.empty() || m_valueName.empty() || m_policyAName.empty() ||
        m_policyBName.empty() || m_wdlName.empty() || m_movesLeftName.empty()) {
        std::cerr << "TensorRT model must expose data, value, pi_a, pi_b, wdl_out, and moves_left" << std::endl;
        return false;
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
    nvinfer1::Dims dims{};
    dims.nbDims = 4;
    dims.d[0] = m_batchSize;
    dims.d[1] = NB_INPUT_CHANNELS;
    dims.d[2] = BOARD_HEIGHT;
    dims.d[3] = BOARD_WIDTH;
    size_t inputSize = m_batchSize * NB_INPUT_CHANNELS * BOARD_HEIGHT * BOARD_WIDTH * sizeof(__half);
    size_t valSize = m_batchSize * sizeof(__half);
    size_t polSize = m_batchSize * NB_POLICY_CHANNELS * BOARD_HEIGHT * BOARD_WIDTH * sizeof(__half);
    size_t wdlSize = m_batchSize * 3 * sizeof(__half);
    size_t movesLeftSize = m_batchSize * sizeof(__half);

    m_executionStates.reserve(SearchParams::NUM_SEARCH_THREADS);
    for (int workerIndex = 0; workerIndex < SearchParams::NUM_SEARCH_THREADS; ++workerIndex) {
        auto state = std::make_unique<ExecutionState>();
        if (!checkCuda(cudaStreamCreate(&state->stream), "cudaStreamCreate")) {
            return false;
        }
        state->context.reset(m_engine->createExecutionContext());
        if (!state->context ||
            !state->context->setOptimizationProfileAsync(workerIndex, state->stream) ||
            !state->context->setInputShape(m_inputName.c_str(), dims)) {
            std::cerr << "Failed to initialize TensorRT worker context "
                      << workerIndex << std::endl;
            return false;
        }
        if (!checkCuda(cudaMalloc(&state->deviceObsBuffer, inputSize), "cudaMalloc(input)") ||
            !checkCuda(cudaMalloc(&state->deviceValueBuffer, valSize), "cudaMalloc(value)") ||
            !checkCuda(cudaMalloc(&state->devicePolicyABuffer, polSize), "cudaMalloc(policy A)") ||
            !checkCuda(cudaMalloc(&state->devicePolicyBBuffer, polSize), "cudaMalloc(policy B)") ||
            !checkCuda(cudaMalloc(&state->deviceWdlBuffer, wdlSize), "cudaMalloc(WDL)") ||
            !checkCuda(cudaMalloc(&state->deviceMovesLeftBuffer, movesLeftSize), "cudaMalloc(moves left)") ||
            !checkCuda(cudaMallocHost(&state->hostObsHalf, inputSize), "cudaMallocHost(input half)") ||
            !checkCuda(cudaMallocHost(&state->hostValueHalf, valSize), "cudaMallocHost(value half)") ||
            !checkCuda(cudaMallocHost(&state->hostPolicyAHalf, polSize), "cudaMallocHost(policy A half)") ||
            !checkCuda(cudaMallocHost(&state->hostPolicyBHalf, polSize), "cudaMallocHost(policy B half)") ||
            !checkCuda(cudaMallocHost(&state->hostWdlHalf, wdlSize), "cudaMallocHost(WDL half)") ||
            !checkCuda(cudaMallocHost(&state->hostMovesLeftHalf, movesLeftSize), "cudaMallocHost(moves left half)")) {
            return false;
        }
        if (!state->context->setTensorAddress(m_inputName.c_str(), state->deviceObsBuffer) ||
            !state->context->setTensorAddress(m_valueName.c_str(), state->deviceValueBuffer) ||
            !state->context->setTensorAddress(m_policyAName.c_str(), state->devicePolicyABuffer) ||
            !state->context->setTensorAddress(m_policyBName.c_str(), state->devicePolicyBBuffer) ||
            !state->context->setTensorAddress(m_wdlName.c_str(), state->deviceWdlBuffer) ||
            !state->context->setTensorAddress(m_movesLeftName.c_str(), state->deviceMovesLeftBuffer) ||
            !checkCuda(cudaStreamSynchronize(state->stream), "TensorRT profile setup")) {
            std::cerr << "Failed to bind TensorRT worker context " << workerIndex << std::endl;
            return false;
        }
        m_executionStates.push_back(std::move(state));
    }

    return true;
}

bool Engine::runInference(float* obs, float* value, float* piA, float* piB,
                          float* wdl, float* movesLeft, size_t workerIndex) {
    if (!obs || !value || !piA || !piB || !wdl || !movesLeft ||
        workerIndex >= m_executionStates.size()) {
        return false;
    }
    ExecutionState& state = *m_executionStates[workerIndex];

    const size_t inputElements = m_batchSize * NB_INPUT_CHANNELS * BOARD_HEIGHT * BOARD_WIDTH;
    const size_t valueElements = m_batchSize;
    const size_t policyElements = m_batchSize * NB_POLICY_CHANNELS * BOARD_HEIGHT * BOARD_WIDTH;
    const size_t wdlElements = m_batchSize * 3;
    const size_t movesLeftElements = m_batchSize;
    floatsToHalves(obs, state.hostObsHalf, inputElements);

    HalfInferenceOutputs outputs;
    if (!runInferenceHalfImpl(
            static_cast<const __half*>(state.hostObsHalf), outputs,
            workerIndex, true)) {
        return false;
    }

    halvesToFloats(outputs.value, value, valueElements);
    halvesToFloats(outputs.policyA, piA, policyElements);
    halvesToFloats(outputs.policyB, piB, policyElements);
    halvesToFloats(state.hostWdlHalf, wdl, wdlElements);
    halvesToFloats(state.hostMovesLeftHalf, movesLeft, movesLeftElements);

    return true;
}

bool Engine::runInferenceHalf(const __half* obs, HalfInferenceOutputs& outputs,
                              size_t workerIndex) {
    return runInferenceHalfImpl(obs, outputs, workerIndex, true);
}

bool Engine::enqueueInferenceHalf(const __half* obs, size_t workerIndex) {
    return enqueueInferenceHalfImpl(obs, workerIndex, true);
}

bool Engine::enqueueInferenceHalfImpl(const __half* obs, size_t workerIndex,
                                      bool copyAuxiliaryOutputs) {
    if (!obs || workerIndex >= m_executionStates.size() ||
        !checkCuda(cudaSetDevice(m_deviceId), "cudaSetDevice")) {
        return false;
    }
    ExecutionState& state = *m_executionStates[workerIndex];
    if (state.inferencePending) {
        std::cerr << "TensorRT worker context " << workerIndex
                  << " already has an inference pending" << std::endl;
        return false;
    }

    const size_t inputSize = m_batchSize * NB_INPUT_CHANNELS * BOARD_HEIGHT
        * BOARD_WIDTH * sizeof(__half);
    const size_t valSize = m_batchSize * sizeof(__half);
    const size_t polSize = m_batchSize * NB_POLICY_CHANNELS * BOARD_HEIGHT
        * BOARD_WIDTH * sizeof(__half);
    const size_t wdlSize = m_batchSize * 3 * sizeof(__half);
    const size_t movesLeftSize = m_batchSize * sizeof(__half);

    if (!checkCuda(cudaMemcpyAsync(
            state.deviceObsBuffer, obs, inputSize, cudaMemcpyHostToDevice,
            state.stream), "cudaMemcpyAsync(input)")) {
        return false;
    }

    if (!state.graphCreated) {
        // Warmup execution to initialize internal TRT states
        if (!state.context->enqueueV3(state.stream) ||
            !checkCuda(cudaStreamSynchronize(state.stream), "TensorRT warmup")) {
            return false;
        }
        
        // Capture the kernel sequence
        if (!checkCuda(cudaStreamBeginCapture(state.stream, cudaStreamCaptureModeThreadLocal), "cudaStreamBeginCapture") ||
            !state.context->enqueueV3(state.stream) ||
            !checkCuda(cudaStreamEndCapture(state.stream, &state.graph), "cudaStreamEndCapture")) {
            return false;
        }
        
        // Instantiate the executable graph
        if (!checkCuda(cudaGraphInstantiate(&state.graphInstance, state.graph, 0), "cudaGraphInstantiate")) {
            return false;
        }
        state.graphCreated = true;
    }
    if (!checkCuda(cudaGraphLaunch(state.graphInstance, state.stream), "cudaGraphLaunch")) {
        return false;
    }
    state.inferencePending = true;

    if (!checkCuda(cudaMemcpyAsync(state.hostValueHalf, state.deviceValueBuffer, valSize, cudaMemcpyDeviceToHost, state.stream),
                   "cudaMemcpyAsync(value)") ||
        !checkCuda(cudaMemcpyAsync(state.hostPolicyAHalf, state.devicePolicyABuffer, polSize, cudaMemcpyDeviceToHost, state.stream),
                   "cudaMemcpyAsync(policy A)") ||
        !checkCuda(cudaMemcpyAsync(state.hostPolicyBHalf, state.devicePolicyBBuffer, polSize, cudaMemcpyDeviceToHost, state.stream),
                   "cudaMemcpyAsync(policy B)")) {
        cudaStreamSynchronize(state.stream);
        state.inferencePending = false;
        return false;
    }
    if (copyAuxiliaryOutputs &&
        (!checkCuda(cudaMemcpyAsync(state.hostWdlHalf, state.deviceWdlBuffer, wdlSize, cudaMemcpyDeviceToHost, state.stream),
                    "cudaMemcpyAsync(WDL)") ||
         !checkCuda(cudaMemcpyAsync(state.hostMovesLeftHalf, state.deviceMovesLeftBuffer, movesLeftSize, cudaMemcpyDeviceToHost, state.stream),
                    "cudaMemcpyAsync(moves left)"))) {
        cudaStreamSynchronize(state.stream);
        state.inferencePending = false;
        return false;
    }

    return true;
}

bool Engine::synchronizeInferenceHalf(HalfInferenceOutputs& outputs,
                                      size_t workerIndex) {
    outputs = {};
    if (workerIndex >= m_executionStates.size() ||
        !checkCuda(cudaSetDevice(m_deviceId), "cudaSetDevice")) {
        return false;
    }
    ExecutionState& state = *m_executionStates[workerIndex];
    if (!state.inferencePending) {
        std::cerr << "TensorRT worker context " << workerIndex
                  << " has no inference pending" << std::endl;
        return false;
    }

    if (!checkCuda(cudaStreamSynchronize(state.stream), "cudaStreamSynchronize")) {
        state.inferencePending = false;
        return false;
    }
    state.inferencePending = false;

    outputs.value = static_cast<const __half*>(state.hostValueHalf);
    outputs.policyA = static_cast<const __half*>(state.hostPolicyAHalf);
    outputs.policyB = static_cast<const __half*>(state.hostPolicyBHalf);
    outputs.wdl = static_cast<const __half*>(state.hostWdlHalf);
    outputs.movesLeft = static_cast<const __half*>(state.hostMovesLeftHalf);

    return true;
}

bool Engine::runInferenceHalfImpl(const __half* obs, HalfInferenceOutputs& outputs,
                                  size_t workerIndex, bool copyAuxiliaryOutputs) {
    outputs = {};
    return enqueueInferenceHalfImpl(obs, workerIndex, copyAuxiliaryOutputs)
        && synchronizeInferenceHalf(outputs, workerIndex);
}