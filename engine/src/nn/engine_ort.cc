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
//
// On macOS the network runs on the GPU through ORT's Core ML provider when
// the runtime has it; HIVEMIND_COREML selects the compute units (gpu, all,
// ane) or turns it off. Networks should first go through
// engine/scripts/convert_onnx_coreml.py, or Core ML runs them in pieces.

#include "nn/engine.h"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <future>
#include <stdexcept>
#include <unordered_map>

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

// Core ML compute units for MLComputeUnits; `units` is null when Core ML is
// unavailable or disabled, which leaves the CPU provider.
struct CoreMLUnits {
    const char* units = nullptr;
    const char* label = "";
};

CoreMLUnits coreml_units() {
    const auto providers = Ort::GetAvailableProviders();
    if (std::find(providers.begin(), providers.end(),
                  "CoreMLExecutionProvider") == providers.end()) {
        return {};
    }
    const char* env = std::getenv("HIVEMIND_COREML");
    const std::string choice = lowered(env && *env ? env : "gpu");
    if (choice == "off" || choice == "cpu" || choice == "0") return {};
    if (choice == "all") return {"ALL", "GPU and Neural Engine"};
    if (choice == "ane") return {"CPUAndNeuralEngine", "Neural Engine"};
    if (choice != "gpu") {
        std::cout << "info string warning unknown HIVEMIND_COREML value '"
                  << choice << "'; using gpu" << std::endl;
    }
    return {"CPUAndGPU", "GPU"};
}

// Core ML compiles for fixed shapes, and the graph derives reshape targets
// from the batch dimension. Pinning that dimension lets ORT fold the shape
// arithmetic away so the whole network stays in one Core ML partition.
std::string batch_dimension_name(const std::filesystem::path& model) {
    Ort::SessionOptions options;
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
    const Ort::Session probe(ort_env(), model.c_str(), options);
    // The shape info is a view into the type info, which must outlive it.
    const Ort::TypeInfo type = probe.GetInputTypeInfo(0);
    const auto info = type.GetTensorTypeAndShapeInfo();
    const auto shape = info.GetShape();
    const auto symbols = info.GetSymbolicDimensions();
    if (shape.empty() || shape[0] >= 0 || symbols.empty() || !symbols[0]) {
        return {};
    }
    return symbols[0];
}

std::vector<std::unique_ptr<Ort::Session>> coreml_sessions(
    const Ort::SessionOptions& cpuOptions, const std::filesystem::path& model,
    const CoreMLUnits& coreml, int batchSize, int count) {
    Ort::SessionOptions options = cpuOptions.Clone();
    // The CPU only casts the input and outputs here. A one-thread pool keeps
    // idle spinning workers from taking thermal headroom from the GPU and
    // the search threads, which matters on a fanless MacBook Air.
    options.SetIntraOpNumThreads(1);
    const std::string batchName = batch_dimension_name(model);
    if (!batchName.empty()) {
        Ort::ThrowOnError(Ort::GetApi().AddFreeDimensionOverrideByName(
            options, batchName.c_str(), batchSize));
    }
    options.AppendExecutionProvider("CoreML", {
        {"ModelFormat", "MLProgram"},
        {"MLComputeUnits", coreml.units},
        {"RequireStaticInputShapes", "1"},
        {"SpecializationStrategy", "FastPrediction"},
    });

    // ORT serialises predictions on one Core ML model, so each search worker
    // gets its own. Their batches then overlap on the GPU the way TensorRT's
    // per-worker execution contexts do.
    std::vector<std::future<std::unique_ptr<Ort::Session>>> pending;
    for (int i = 0; i < count; ++i) {
        pending.push_back(std::async(std::launch::async, [&options, &model] {
            return std::make_unique<Ort::Session>(ort_env(), model.c_str(),
                                                  options);
        }));
    }
    std::vector<std::unique_ptr<Ort::Session>> sessions;
    for (auto& session : pending) sessions.push_back(session.get());
    return sessions;
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
        Ort::Session* session = nullptr;
    };

    Ort::SessionOptions options;
    // One shared session on the CPU, one per worker on Core ML.
    std::vector<std::unique_ptr<Ort::Session>> sessions;
    std::string provider = "CPU";
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

const char* Engine::backendName() { return "ONNX Runtime"; }

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
        // Concurrent Run calls share this session's intra-op pool. Let ORT
        // choose its topology-aware default instead of dividing the one shared
        // pool by the number of search workers and leaving cores idle.
        const int intraOp = 0;
        state.options.SetIntraOpNumThreads(intraOp);
        state.options.SetInterOpNumThreads(1);
        state.options.SetExecutionMode(ORT_SEQUENTIAL);

        const std::filesystem::path modelPath(onnxFile);
        const int workerCount = std::max(1, SearchParams::NUM_SEARCH_THREADS);
        const CoreMLUnits coreml = coreml_units();
        if (coreml.units) {
            try {
                state.sessions = coreml_sessions(state.options, modelPath, coreml,
                                                 m_batchSize, workerCount);
                state.provider = std::string("Core ML ") + coreml.label;
            } catch (const Ort::Exception& e) {
                std::cout << "info string warning Core ML failed to load the "
                             "network, using the CPU: "
                          << e.what() << std::endl;
                state.sessions.clear();
            }
        }
        if (state.sessions.empty()) {
            state.sessions.push_back(std::make_unique<Ort::Session>(
                ort_env(), modelPath.c_str(), state.options));
            state.provider = "CPU";
        }
        const bool onCoreML = state.provider != "CPU";
        const Ort::Session& session = *state.sessions.front();

        Ort::AllocatorWithDefaultOptions allocator;

        if (session.GetInputCount() != 1) {
            std::cerr << "Expected exactly one network input, found "
                      << session.GetInputCount() << std::endl;
            return false;
        }

        auto inputName = session.GetInputNameAllocated(0, allocator);
        m_inputName = inputName.get();

        if (onCoreML &&
            !session.GetModelMetadata().LookupCustomMetadataMapAllocated(
                "hivemind_coreml", allocator)) {
            std::cout << "info string warning this network has not been "
                         "prepared for Core ML and will bounce between the "
                         "GPU and CPU; convert it with "
                         "engine/scripts/convert_onnx_coreml.py"
                      << std::endl;
        }

        const auto inputType = session.GetInputTypeInfo(0)
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
        if (state.usesFp16 && !onCoreML) {
            std::cout
                << "info string warning FP16 inference is usually slow on CPU; "
                   "convert the model with engine/scripts/convert_onnx_fp32.py"
                << std::endl;
        }

        const size_t outputCount = session.GetOutputCount();
        state.outputNameStorage.reserve(outputCount);
        for (size_t i = 0; i < outputCount; ++i) {
            auto name = session.GetOutputNameAllocated(i, allocator);
            state.outputNameStorage.emplace_back(name.get());
            const auto type = session.GetOutputTypeInfo(i)
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
                session.GetOutputTypeInfo(state.jointFactorsAIdx)
                    .GetTensorTypeAndShapeInfo()
                    .GetShape();
            // [batch, rank, vocabulary]
            if (shape.size() == 3 && shape[1] > 0) {
                m_jointFactorRank = static_cast<size_t>(shape[1]);
            }
        }

        state.workers.resize(workerCount);
        for (size_t i = 0; i < state.workers.size(); ++i) {
            auto& worker = state.workers[i];
            const size_t inputElements =
                static_cast<size_t>(m_batchSize) * NB_INPUT_VALUES();
            if (state.usesFp16) {
                worker.inputHalf.resize(inputElements);
            } else {
                worker.inputFloat.resize(inputElements);
            }
            worker.convertedOutputs.resize(outputCount);
            worker.session = state.sessions[i % state.sessions.size()].get();
        }

        std::cout << "info string backend " << backendName() << " ("
                  << state.provider << ") model " << onnxFile << " batch "
                  << m_batchSize << " workers " << workerCount << " sessions "
                  << state.sessions.size() << " precision "
                  << (state.usesFp16 ? "fp16" : "fp32")
                  << " intra-op threads "
                  << (onCoreML ? std::string("1")
                      : intraOp == 0 ? std::string("auto")
                                     : std::to_string(intraOp))
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
    if (!obs || !m_ort || m_ort->sessions.empty() ||
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

            worker.outputs = worker.session->Run(
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
