// ONNX Runtime inference backend.
//
// A drop-in replacement for the TensorRT backend in engine.cc that runs the
// same exported graph on any ONNX Runtime execution provider. It exists so the
// engine can be built and shipped without CUDA: TensorRT plus its per-SM
// builder resources is ~2 GB and NVIDIA-only, while ORT's CPU build is ~28 MB
// and runs everywhere.
//
// `enqueue`/`synchronize` overlap is preserved with a worker thread per
//     search thread, which is what keeps batch N's inference running while the
//     search walks the tree for batch N+1.

#include "nn/engine.h"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <future>
#include <stdexcept>

#include "environment/constants.h"
#include "search/search_params.h"

namespace {

Ort::Env& ort_env() {
    static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "hivemind");
    return env;
}

std::string lowered(std::string name) {
    std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return name;
}

// The exported graph names its heads differently across training runs
// (`pi_a` / `policy_a`, `wdl_out` / `wdl`, ...), so match on a normalized
// substring rather than an exact string.
bool matches(const std::string& normalized,
             std::initializer_list<const char*> needles) {
    for (const char* needle : needles) {
        if (normalized.find(needle) != std::string::npos) return true;
    }
    return false;
}

}  // namespace

struct Engine::OrtState {
    // One in-flight request per search thread.
    struct Worker {
        std::vector<__half> inputHalf;
        std::vector<float> inputFloat;
        std::vector<Ort::Value> outputs;
        std::vector<std::vector<__half>> convertedOutputs;
        std::future<bool> pending;
        bool hasPending = false;
    };

    Ort::SessionOptions options;
    std::unique_ptr<Ort::Session> session;
    Ort::MemoryInfo memoryInfo =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::vector<Worker> workers;

    // Names owned here; the C API wants stable `const char*`.
    std::vector<std::string> outputNameStorage;
    std::vector<const char*> outputNames;
    std::array<const char*, 1> inputNames{nullptr};
    bool usesFp16 = true;
    std::vector<ONNXTensorElementDataType> outputTypes;

    // Index into `outputs` for each head, or -1 when the graph omits it.
    int valueIdx = -1;
    int policyAIdx = -1;
    int policyBIdx = -1;
    int wdlIdx = -1;
    int movesLeftIdx = -1;
    int jointFactorsAIdx = -1;
    int jointFactorsBIdx = -1;
};

const char* Engine::backendName() { return "ONNX Runtime (CPU)"; }

Engine::Engine(int deviceId, int batchSize)
    : m_deviceId(deviceId),
      m_batchSize(batchSize > 0 ? batchSize : SearchParams::BATCH_SIZE),
      m_ort(std::make_unique<OrtState>()) {}

Engine::~Engine() {
    if (!m_ort) return;
    // Drain anything still running so worker threads never outlive the session.
    for (auto& worker : m_ort->workers) {
        if (worker.hasPending && worker.pending.valid()) {
            try {
                worker.pending.wait();
            } catch (...) {
            }
            worker.hasPending = false;
        }
    }
}

bool Engine::loadNetwork(const std::string& onnxFile,
                         const std::string& /*engineFile*/) {
    try {
        auto& state = *m_ort;

        state.options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        // Search threads each drive their own batch, so give ORT the remaining
        // cores for intra-op parallelism rather than letting the two layers
        // oversubscribe each other.
        const int cores = static_cast<int>(std::thread::hardware_concurrency());
        const int searchThreads = std::max(1, SearchParams::NUM_SEARCH_THREADS);
        const int intraOp = std::max(1, cores / searchThreads);
        state.options.SetIntraOpNumThreads(intraOp);
        state.options.SetInterOpNumThreads(1);
        state.options.SetExecutionMode(ORT_SEQUENTIAL);

        const std::filesystem::path modelPath(onnxFile);
        state.session = std::make_unique<Ort::Session>(
            ort_env(), modelPath.c_str(), state.options);

        Ort::AllocatorWithDefaultOptions allocator;

        if (state.session->GetInputCount() != 1) {
            std::cerr << "Expected exactly one network input, found "
                      << state.session->GetInputCount() << std::endl;
            return false;
        }

        auto inputName = state.session->GetInputNameAllocated(0, allocator);
        m_inputName = inputName.get();

        const auto inputType = state.session->GetInputTypeInfo(0)
                                   .GetTensorTypeAndShapeInfo()
                                   .GetElementType();
        if (inputType != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 &&
            inputType != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            std::cerr << "Unsupported network input type: " << inputType << std::endl;
            return false;
        }
#if !defined(HIVEMIND_ORT_FP16)
        if (inputType == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
            std::cerr << "This is an FP16 network. Rebuild with "
                         "-DHIVEMIND_ORT_FP16=ON to use it directly."
                      << std::endl;
            return false;
        }
#endif
        state.usesFp16 = inputType == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;

        const size_t outputCount = state.session->GetOutputCount();
        state.outputNameStorage.reserve(outputCount);
        for (size_t i = 0; i < outputCount; ++i) {
            auto name = state.session->GetOutputNameAllocated(i, allocator);
            state.outputNameStorage.emplace_back(name.get());
            const auto type = state.session->GetOutputTypeInfo(i)
                                  .GetTensorTypeAndShapeInfo()
                                  .GetElementType();
            if (type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 &&
                type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
                std::cerr << "Unsupported network output type for " << name.get()
                          << ": " << type << std::endl;
                return false;
            }
            state.outputTypes.push_back(type);
        }
        for (const auto& name : state.outputNameStorage) {
            state.outputNames.push_back(name.c_str());
        }
        state.inputNames[0] = m_inputName.c_str();

        for (size_t i = 0; i < state.outputNameStorage.size(); ++i) {
            const std::string norm = lowered(state.outputNameStorage[i]);
            const int idx = static_cast<int>(i);
            // Order matters: "jointfactors_a" also contains "_a".
            if (matches(norm, {"jointfactor", "joint_factor"})) {
                if (matches(norm, {"_a", "a_"}) || norm.back() == 'a') {
                    state.jointFactorsAIdx = idx;
                    m_jointFactorsAName = state.outputNameStorage[i];
                } else {
                    state.jointFactorsBIdx = idx;
                    m_jointFactorsBName = state.outputNameStorage[i];
                }
            } else if (matches(norm, {"wdl"})) {
                state.wdlIdx = idx;
                m_wdlName = state.outputNameStorage[i];
            } else if (matches(norm, {"moves_left", "movesleft"})) {
                state.movesLeftIdx = idx;
                m_movesLeftName = state.outputNameStorage[i];
            } else if (matches(norm, {"pi_a", "policy_a", "policya"})) {
                state.policyAIdx = idx;
                m_policyAName = state.outputNameStorage[i];
            } else if (matches(norm, {"pi_b", "policy_b", "policyb"})) {
                state.policyBIdx = idx;
                m_policyBName = state.outputNameStorage[i];
            } else if (matches(norm, {"value"})) {
                state.valueIdx = idx;
                m_valueName = state.outputNameStorage[i];
            }
        }

        if (state.valueIdx < 0 || state.policyAIdx < 0 || state.policyBIdx < 0) {
            std::cerr << "Network is missing a required head (value/pi_a/pi_b)."
                      << std::endl;
            return false;
        }

        // Joint-factor rank comes from the head's own shape when present.
        m_jointFactorRank = 0;
        if (state.jointFactorsAIdx >= 0) {
            const auto shape =
                state.session->GetOutputTypeInfo(state.jointFactorsAIdx)
                    .GetTensorTypeAndShapeInfo()
                    .GetShape();
            // [batch, rank, vocabulary]
            if (shape.size() == 3 && shape[1] > 0) {
                m_jointFactorRank = static_cast<size_t>(shape[1]);
            }
        }

        const int workerCount = std::max(1, SearchParams::NUM_SEARCH_THREADS);
        state.workers.resize(workerCount);
        for (auto& worker : state.workers) {
            const size_t inputElements =
                static_cast<size_t>(m_batchSize) * NB_INPUT_VALUES();
            if (state.usesFp16) {
                worker.inputHalf.resize(inputElements);
            } else {
                worker.inputFloat.resize(inputElements);
            }
            worker.convertedOutputs.resize(outputCount);
        }

        std::cout << "info string backend " << backendName() << " model "
                  << onnxFile << " batch " << m_batchSize << " workers "
                  << workerCount << " precision "
                  << (state.usesFp16 ? "fp16" : "fp32")
                  << " intra-op threads " << intraOp
                  << std::endl;
        return true;
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime failed to load the network: " << e.what()
                  << std::endl;
        return false;
    } catch (const std::exception& e) {
        std::cerr << "Failed to load the network: " << e.what() << std::endl;
        return false;
    }
}

bool Engine::enqueueInferenceHalf(const __half* obs, size_t workerIndex) {
    if (!obs || !m_ort || !m_ort->session ||
        workerIndex >= m_ort->workers.size()) {
        return false;
    }
    auto& worker = m_ort->workers[workerIndex];
    if (worker.hasPending) {
        // One request in flight per worker, per the header contract.
        return false;
    }

    // Copy rather than alias the caller's buffer: the search reuses its
    // double-buffered observation slabs as soon as it has enqueued.
    if (m_ort->usesFp16) {
        std::memcpy(worker.inputHalf.data(), obs,
                    worker.inputHalf.size() * sizeof(__half));
    } else {
        std::transform(obs, obs + worker.inputFloat.size(),
                       worker.inputFloat.begin(), __half2float);
    }

    auto& state = *m_ort;
    const int64_t batch = m_batchSize;
    worker.pending = std::async(std::launch::async, [&state, &worker, batch]() {
        try {
            const std::array<int64_t, 4> shape{
                batch, NB_INPUT_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH};
            Ort::Value input{nullptr};
            if (state.usesFp16) {
                input = Ort::Value::CreateTensor(
                    state.memoryInfo, worker.inputHalf.data(),
                    worker.inputHalf.size() * sizeof(__half),
                    shape.data(), shape.size(),
                    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
            } else {
                input = Ort::Value::CreateTensor<float>(
                    state.memoryInfo, worker.inputFloat.data(),
                    worker.inputFloat.size(), shape.data(), shape.size());
            }

            worker.outputs = state.session->Run(
                Ort::RunOptions{nullptr}, state.inputNames.data(), &input, 1,
                state.outputNames.data(), state.outputNames.size());
            return true;
        } catch (const Ort::Exception& e) {
            std::cerr << "ONNX Runtime inference failed: " << e.what()
                      << std::endl;
            return false;
        }
    });
    worker.hasPending = true;
    return true;
}

bool Engine::synchronizeInferenceHalf(HalfInferenceOutputs& outputs,
                                      size_t workerIndex) {
    if (!m_ort || workerIndex >= m_ort->workers.size()) return false;
    auto& worker = m_ort->workers[workerIndex];
    if (!worker.hasPending) return false;

    worker.hasPending = false;
    bool ok = false;
    try {
        ok = worker.pending.get();
    } catch (const std::exception& e) {
        std::cerr << "Inference worker threw: " << e.what() << std::endl;
        return false;
    }
    if (!ok) return false;

    auto& state = *m_ort;
    auto at = [&](int idx) -> const __half* {
        if (idx < 0 || static_cast<size_t>(idx) >= worker.outputs.size()) {
            return nullptr;
        }
        const size_t outputIndex = static_cast<size_t>(idx);
        if (state.outputTypes[outputIndex] ==
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
            return static_cast<const __half*>(
                worker.outputs[outputIndex].GetTensorRawData());
        }
        const auto* values = worker.outputs[outputIndex].GetTensorData<float>();
        auto& converted = worker.convertedOutputs[outputIndex];
        const size_t count = worker.outputs[outputIndex]
                                 .GetTensorTypeAndShapeInfo()
                                 .GetElementCount();
        converted.resize(count);
        std::transform(values, values + count, converted.begin(), __float2half_rn);
        return converted.data();
    };

    outputs.value = at(state.valueIdx);
    outputs.policyA = at(state.policyAIdx);
    outputs.policyB = at(state.policyBIdx);
    outputs.wdl = at(state.wdlIdx);
    outputs.movesLeft = at(state.movesLeftIdx);
    outputs.jointFactorsA = at(state.jointFactorsAIdx);
    outputs.jointFactorsB = at(state.jointFactorsBIdx);
    outputs.jointFactorRank = outputs.jointFactorsA ? m_jointFactorRank : 0;
    return outputs.value && outputs.policyA && outputs.policyB;
}

bool Engine::runInferenceHalf(const __half* obs, HalfInferenceOutputs& outputs,
                              size_t workerIndex) {
    if (!enqueueInferenceHalf(obs, workerIndex)) return false;
    return synchronizeInferenceHalf(outputs, workerIndex);
}

bool Engine::runInference(float* obs, float* value, float* piA, float* piB,
                          float* wdl, float* movesLeft, size_t workerIndex) {
    const size_t inputElements =
        static_cast<size_t>(m_batchSize) * NB_INPUT_VALUES();
    std::vector<__half> halfInput(inputElements);
    std::transform(obs, obs + inputElements, halfInput.begin(), __float2half_rn);

    HalfInferenceOutputs outputs;
    if (!runInferenceHalf(halfInput.data(), outputs, workerIndex)) return false;

    const size_t batch = static_cast<size_t>(m_batchSize);
    auto copy = [](const __half* src, float* dst, size_t count) {
        if (src && dst) {
            std::transform(src, src + count, dst, __half2float);
        }
    };
    copy(outputs.value, value, batch);
    copy(outputs.policyA, piA, batch * NB_POLICY_VALUES());
    copy(outputs.policyB, piB, batch * NB_POLICY_VALUES());
    copy(outputs.wdl, wdl, batch * 3);
    copy(outputs.movesLeft, movesLeft, batch);
    return true;
}
