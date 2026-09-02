#pragma once

#include "nn/backend_compat.h"

#if defined(HIVEMIND_BACKEND_TENSORRT)
#include <NvInfer.h>
#include <NvOnnxParser.h>
#endif

#include <cstdint>
#include <iostream>
#include <fstream>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "environment/constants.h"
#include "search/search_params.h"

#if defined(HIVEMIND_BACKEND_TENSORRT)
/**
 * @brief Simple Logger implementation for TensorRT.
 */
class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        const std::string_view message = msg ? msg : "";
        const bool isProfileZeroMetadataWarning =
            message.find("only returns results for profile 0") != std::string_view::npos &&
            (message.find("getTensorVectorizedDim") != std::string_view::npos ||
             message.find("getTensorComponentsPerElement") != std::string_view::npos);
        if (isProfileZeroMetadataWarning) {
            return;
        }
        // Only log warnings and errors to reduce console noise during high-speed inference
        if (severity <= Severity::kWARNING) {
            std::cerr << (severity == Severity::kERROR ? "[ERROR] " : "[WARNING] ") << msg << std::endl;
        }
    }
};
#endif

/**
 * @brief Neural network inference wrapper.
 *
 * Two backends implement this interface and the search code cannot tell them
 * apart:
 *   * TensorRT (engine.cc)      — FP16, CUDA graphs, pinned async streams.
 *   * ONNX Runtime (engine_ort.cc) — FP32, portable, CPU or any ORT provider.
 *
 * Both honour the same contract: `enqueueInferenceHalf` starts one batch for a
 * worker and returns immediately, `synchronizeInferenceHalf` waits for it and
 * hands back pointers into engine-owned buffers that stay valid until that
 * worker's next call.
 */
class Engine {
public:
    struct HalfInferenceOutputs {
        const __half* value = nullptr;
        const __half* policyA = nullptr;
        const __half* policyB = nullptr;
        const __half* wdl = nullptr;
        const __half* movesLeft = nullptr;
        const __half* jointFactorsA = nullptr;
        const __half* jointFactorsB = nullptr;
        size_t jointFactorRank = 0;
    };

    explicit Engine(int deviceId, int batchSize = SearchParams::BATCH_SIZE);
    ~Engine();

    // Prevent copying to avoid double-free of backend resources
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    bool loadNetwork(const std::string& onnxFile, const std::string& engineFile);

    /**
     * @brief Performs inference. For max QPS, ensure input/output pointers
     * are allocated via hm::alloc_pinned.
     */
    bool runInference(float* obs, float* value, float* piA, float* piB,
                      float* wdl, float* movesLeft, size_t workerIndex = 0);

    bool runInferenceHalf(const __half* obs, HalfInferenceOutputs& outputs,
                          size_t workerIndex = 0);

    // Each worker may have one request in flight. The pinned input must remain
    // unchanged until synchronizeInferenceHalf returns.
    bool enqueueInferenceHalf(const __half* obs, size_t workerIndex = 0);
    bool synchronizeInferenceHalf(HalfInferenceOutputs& outputs,
                                  size_t workerIndex = 0);

    /**
     * @brief Get the batch size this engine was built with.
     */
    int getBatchSize() const { return m_batchSize; }

    /// Human-readable backend name, e.g. "TensorRT" or "ONNX Runtime (CPU)".
    static const char* backendName();

private:
#if defined(HIVEMIND_BACKEND_TENSORRT)
    struct ExecutionState {
        ~ExecutionState();

        std::unique_ptr<nvinfer1::IExecutionContext> context;
        cudaStream_t stream = nullptr;
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t graphInstance = nullptr;
        bool graphCreated = false;
        bool inferencePending = false;
        void* deviceObsBuffer = nullptr;
        void* deviceValueBuffer = nullptr;
        void* devicePolicyABuffer = nullptr;
        void* devicePolicyBBuffer = nullptr;
        void* deviceWdlBuffer = nullptr;
        void* deviceMovesLeftBuffer = nullptr;
        void* deviceJointFactorsABuffer = nullptr;
        void* deviceJointFactorsBBuffer = nullptr;
        void* hostObsHalf = nullptr;
        void* hostValueHalf = nullptr;
        void* hostPolicyAHalf = nullptr;
        void* hostPolicyBHalf = nullptr;
        void* hostWdlHalf = nullptr;
        void* hostMovesLeftHalf = nullptr;
        void* hostJointFactorsAHalf = nullptr;
        void* hostJointFactorsBHalf = nullptr;
    };
#else
    // Opaque so onnxruntime headers stay out of every translation unit that
    // merely needs to hold an Engine*.
    struct OrtState;
#endif

    // Device ID and batch shape
    int m_deviceId;
    int m_batchSize = SearchParams::BATCH_SIZE;

#if defined(HIVEMIND_BACKEND_TENSORRT)
    Logger m_logger;

    // TensorRT Core Objects
    std::unique_ptr<nvinfer1::ICudaEngine> m_engine = nullptr;
    std::vector<std::unique_ptr<ExecutionState>> m_executionStates;
#else
    std::unique_ptr<OrtState> m_ort;
#endif

    std::string m_inputName;
    std::string m_valueName;
    std::string m_policyAName;
    std::string m_policyBName;
    std::string m_wdlName;
    std::string m_movesLeftName;
    std::string m_jointFactorsAName;
    std::string m_jointFactorsBName;
    size_t m_jointFactorRank = 0;

#if defined(HIVEMIND_BACKEND_TENSORRT)
    // Internal helper methods
    bool buildEngineFromONNX(const std::string& onnxFile);
    bool loadEngineFromFile(const std::string& engineFile);
    bool saveEngineToFile(const std::string& engineFile);
    bool initializeResources();
    bool enqueueInferenceHalfImpl(const __half* obs, size_t workerIndex,
                                  bool copyAuxiliaryOutputs);
    bool runInferenceHalfImpl(const __half* obs, HalfInferenceOutputs& outputs,
                              size_t workerIndex, bool copyAuxiliaryOutputs);
#endif
};
