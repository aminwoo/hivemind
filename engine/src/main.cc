#include "interface/uci.h"
#include "environment/constants.h"
#include "common/globals.h"
#include "nn/engine.h"
#include "nn/onnx_utils.h"
#include "tools/benchmark.h"
#include "tools/selfplay.h"
#include "tools/tournament.h"
#include "search/search_params.h"
#include "Fairy-Stockfish/src/bitboard.h"
#include "Fairy-Stockfish/src/misc.h"
#include "Fairy-Stockfish/src/position.h"
#include "Fairy-Stockfish/src/thread.h"
#include "Fairy-Stockfish/src/piece.h"
#include "Fairy-Stockfish/src/types.h"
#include <iostream>
#include "nn/backend_compat.h"
#include <cstring>
#include <filesystem>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

using namespace std; 

namespace {

/**
 * @brief Reads the optional numeric argument of `bench` / `perft`.
 *
 * Both subcommands may also be given the --model flag the other subcommands
 * accept, so scan for the count instead of assuming argv[2] is a number -
 * stoi() on a flag terminates the process.
 */
int optional_count_argument(int argc, char** argv, int defaultValue) {
    for (int index = 2; index < argc; ++index) {
        const string argument = argv[index];
        if (!argument.empty()
            && argument.find_first_not_of("0123456789") == string::npos) {
            return stoi(argument);
        }
    }
    return defaultValue;
}

/**
 * @brief Reads the --batch-size flag of a subcommand.
 * @return The requested size, the compiled default when absent, or -1 on error.
 */
int batch_size_argument(int argc, char** argv) {
    for (int index = 2; index + 1 < argc; ++index) {
        if (string(argv[index]) != "--batch-size") {
            continue;
        }
        try {
            const int requested = stoi(argv[index + 1]);
            if (requested < 1 || requested > 1024) {
                cerr << "--batch-size must be between 1 and 1024" << endl;
                return -1;
            }
            return requested;
        } catch (const exception&) {
            cerr << "--batch-size expects a number" << endl;
            return -1;
        }
    }
    return SearchParams::BATCH_SIZE;
}

/** Returns the --model / --network path of a subcommand, or "" when absent. */
string model_path_argument(int argc, char** argv) {
    for (int index = 2; index + 1 < argc; ++index) {
        const string argument = argv[index];
        if (argument == "--model" || argument == "--network") {
            return argv[index + 1];
        }
    }
    return {};
}

bool parse_bool_argument(const string& value) {
    if (value == "true" || value == "1" || value == "on") return true;
    if (value == "false" || value == "0" || value == "off") return false;
    throw invalid_argument("Expected true/false, got: " + value);
}

}  // namespace

void printUsage(const char* progName) {
    cout << "Usage: " << progName << " [options]" << endl;
    cout << "Options:" << endl;
    cout << "  --log <level>      Set log level: none, info, debug (default: none)" << endl;
    cout << "  --model <onnx>     Load this model in UCI mode (or --network, default: scans ./models)" << endl;
    cout << "  --batch-size <n>   Inference batch size in UCI mode (default: "
         << SearchParams::BATCH_SIZE << "; also settable with the BatchSize UCI option)" << endl;
    cout << "  bench [iters]      Run inference benchmark (accepts --model, --batch-size)" << endl;
    cout << "  perft [depth]      Run move generation benchmark" << endl;
    cout << "  selfplay [options] Generate HVM5 training chunks and bughouse PGN" << endl;
    cout << "    --model <onnx> --games <n> --nodes <n> --output <dir> --seed <n>" << endl;
    cout << "    --max-macro-plies <n> --raw-policy-mean-macro-plies <x>" << endl;
    cout << "    --raw-policy-max-macro-plies <n> --raw-policy-high-temp-probability <x>" << endl;
    cout << "    --mcts-temperature <x> --mcts-temperature-decay <x>" << endl;
    cout << "    --node-random-factor <x>" << endl;
    cout << "    --chunk-samples <n> --dirichlet-alpha <x> --dirichlet-epsilon <x>" << endl;
    cout << "  tournament [options] Run a paired model-vs-model tournament" << endl;
    cout << "    --contender <onnx> --baseline <onnx> --games <even-n>" << endl;
    cout << "    --nodes <n> or --movetime <ms>" << endl;
    cout << "    --contender-batch-size <n> --baseline-batch-size <n>" << endl;
    cout << "    --output <dir> --seed <n> --max-macro-plies <n>" << endl;
    cout << "    --dirichlet-alpha <x> --dirichlet-epsilon <x>" << endl;
    cout << "    --contender-pw-coefficient <x> --baseline-pw-coefficient <x>" << endl;
    cout << "    --contender-threads <n> --baseline-threads <n> --positions <tsv>" << endl;
    cout << "    --contender-{mcgs,transpositions,root-mate-search,wdl-eval} <bool>" << endl;
    cout << "    --baseline-{mcgs,transpositions,root-mate-search,wdl-eval} <bool>" << endl;
    cout << "    --{contender,baseline}-{root-pw-coefficient,wdl-weight,moves-left-discount,q-value-weight,q-veto-delta} <x>" << endl;
    cout << "    --sprt-elo0 <x> --sprt-elo1 <x> [--sprt-alpha <x> --sprt-beta <x>]" << endl;
}

int main(int argc, char* argv[]) {
    // Model discovery must be relative to the executable for release bundles;
    // chess GUIs commonly launch engines with an unrelated working directory.
    Stockfish::CommandLine::init(argc, argv);

    // Parse --log argument first (can appear anywhere)
    for (int i = 1; i < argc; i++) {
        if ((strcmp(argv[i], "--help") == 0) || (strcmp(argv[i], "-h") == 0)) {
            printUsage(argv[0]);
            return EXIT_SUCCESS;
        }
        if (strcmp(argv[i], "--log") == 0 && i + 1 < argc) {
            g_logLevel = parseLogLevel(argv[i + 1]);
            // Remove these args from consideration
            for (int j = i; j + 2 < argc; j++) {
                argv[j] = argv[j + 2];
            }
            argc -= 2;
            i--;  // Recheck this position
        }
    }

    // How many Engine instances to build: one per CUDA device, or a single
    // host engine on the portable backend. Handle --help first so packaged
    // GPU builds can be smoke-tested on CI hosts without an NVIDIA device.
    int deviceCount = 1;
#if defined(HIVEMIND_BACKEND_TENSORRT)
    deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
    if (error_id != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: "
                  << cudaGetErrorString(error_id) << std::endl;
        return EXIT_FAILURE;
    }
#endif

    init_fairy_stockfish();

    init_policy_index();

    // Check for benchmark flag
    if (argc > 1 && string(argv[1]) == "bench") {
        cout << "Running inference benchmark..." << endl;
        const int benchBatchSize = batch_size_argument(argc, argv);
        if (benchBatchSize <= 0) {
            return EXIT_FAILURE;
        }
        Engine engine(0, benchBatchSize);

        const std::string onnxFile = resolveModelPath(
            model_path_argument(argc, argv));
        if (onnxFile.empty()) {
            cerr << "No ONNX model found in ./models or ./engine/models" << endl;
            return EXIT_FAILURE;
        }
        const std::string engineFile = getEnginePath(onnxFile, "fp16", benchBatchSize, 0, "v3");
        
        if (!engine.loadNetwork(onnxFile, engineFile)) {
            cerr << "Failed to load engine" << endl;
            return EXIT_FAILURE;
        }
        
        int iterations = optional_count_argument(argc, argv, 1000);
        benchmark_inference(engine, iterations);
        return EXIT_SUCCESS;
    }

    // Check for perft benchmark flag
    if (argc > 1 && string(argv[1]) == "perft") {
        int depth = optional_count_argument(argc, argv, 5);
        benchmark_movegen(depth);
        return EXIT_SUCCESS;
    }

    if (argc > 1 && string(argv[1]) == "selfplay") {
        SelfPlayConfig config;
        filesystem::path modelPath;
        try {
            for (int i = 2; i < argc; ++i) {
                const string option = argv[i];
                if (i + 1 >= argc) {
                    throw invalid_argument("Missing value for " + option);
                }
                const string value = argv[++i];
                if (option == "--games") config.games = stoull(value);
                else if (option == "--nodes") config.nodes = stoull(value);
                else if (option == "--model" || option == "--network") modelPath = value;
                else if (option == "--output") config.outputDirectory = value;
                else if (option == "--seed") config.seed = stoull(value);
                else if (option == "--max-macro-plies") config.maxMacroPlies = stoull(value);
                else if (option == "--raw-policy-mean-macro-plies") config.rawPolicyMeanMacroPlies = stod(value);
                else if (option == "--raw-policy-max-macro-plies") config.rawPolicyMaxMacroPlies = stoull(value);
                else if (option == "--raw-policy-high-temp-probability") config.rawPolicyHighTemperatureProbability = stod(value);
                else if (option == "--mcts-temperature") config.mctsTemperature = stod(value);
                else if (option == "--mcts-temperature-decay") config.mctsTemperatureDecay = stod(value);
                else if (option == "--mcts-temperature-plies") config.mctsTemperaturePlies = stoull(value);
                else if (option == "--resign-threshold") config.resignThreshold = stof(value);
                else if (option == "--resign-consecutive-plies") config.resignConsecutivePlies = stoull(value);
                else if (option == "--resign-disable-fraction") config.resignDisableFraction = stod(value);
                else if (option == "--q-value-ratio") config.qValueRatio = stod(value);
                else if (option == "--node-random-factor") config.nodeRandomFactor = stod(value);
                else if (option == "--chunk-samples") config.chunkSamples = stoull(value);
                else if (option == "--dirichlet-alpha") config.dirichletAlpha = stof(value);
                else if (option == "--dirichlet-epsilon") config.dirichletEpsilon = stof(value);
                else if (option == "--batch-size") config.batchSize = stoi(value);
                else throw invalid_argument("Unknown selfplay option: " + option);
            }
        } catch (const exception& error) {
            cerr << "Invalid selfplay arguments: " << error.what() << endl;
            return EXIT_FAILURE;
        }

        const string onnxFile = resolveModelPath(modelPath.string());
        if (onnxFile.empty()) {
            cerr << "No ONNX model found; pass --model <onnx>" << endl;
            return EXIT_FAILURE;
        }
        vector<unique_ptr<Engine>> ownedEngines;
        vector<Engine*> engines;
        if (config.batchSize < 1 || config.batchSize > 1024) {
            cerr << "Self-play batch size must be between 1 and 1024" << endl;
            return EXIT_FAILURE;
        }
        for (int deviceId = 0; deviceId < deviceCount; ++deviceId) {
            auto engine = make_unique<Engine>(deviceId, config.batchSize);
            const string engineFile = getEnginePath(
                onnxFile, "fp16", config.batchSize, deviceId, "v3");
            if (!engine->loadNetwork(onnxFile, engineFile)) {
                cerr << "Failed to load engine on device " << deviceId << endl;
                continue;
            }
            engines.push_back(engine.get());
            ownedEngines.push_back(std::move(engine));
        }
        if (engines.empty()) {
            cerr << "Failed to load an engine on any CUDA device" << endl;
            return EXIT_FAILURE;
        }
        cout << "Self-play using " << engines.size() << " GPU engine(s)" << endl;
        try {
            return run_selfplay(engines, config);
        } catch (const exception& error) {
            cerr << "Self-play failed: " << error.what() << endl;
            return EXIT_FAILURE;
        }
    }

    if (argc > 1 && string(argv[1]) == "tournament") {
        TournamentConfig config;
        filesystem::path contenderPath;
        filesystem::path baselinePath;
        try {
            for (int i = 2; i < argc; ++i) {
                const string option = argv[i];
                if (i + 1 >= argc) {
                    throw invalid_argument("Missing value for " + option);
                }
                const string value = argv[++i];
                if (option == "--games") config.games = stoull(value);
                else if (option == "--nodes") {
                    config.nodes = stoull(value);
                    config.moveTimeMs = 0;
                }
                else if (option == "--movetime") {
                    config.moveTimeMs = stoi(value);
                    config.nodes = 0;
                }
                else if (option == "--contender-batch-size") config.contenderBatchSize = stoi(value);
                else if (option == "--baseline-batch-size") config.baselineBatchSize = stoi(value);
                else if (option == "--contender-threads") config.contenderThreads = stoi(value);
                else if (option == "--baseline-threads") config.baselineThreads = stoi(value);
                else if (option == "--contender") contenderPath = value;
                else if (option == "--baseline") baselinePath = value;
                else if (option == "--output") config.outputDirectory = value;
                else if (option == "--positions") config.positionsFile = value;
                else if (option == "--seed") config.seed = stoull(value);
                else if (option == "--max-macro-plies") config.maxMacroPlies = stoull(value);
                else if (option == "--dirichlet-alpha") config.dirichletAlpha = stof(value);
                else if (option == "--dirichlet-epsilon") config.dirichletEpsilon = stof(value);
                else if (option == "--contender-pw-coefficient") {
                    config.contenderPwCoefficient = stof(value);
                    config.contenderRootPwCoefficient = config.contenderPwCoefficient;
                }
                else if (option == "--baseline-pw-coefficient") {
                    config.baselinePwCoefficient = stof(value);
                    config.baselineRootPwCoefficient = config.baselinePwCoefficient;
                }
                else if (option == "--contender-root-pw-coefficient") config.contenderRootPwCoefficient = stof(value);
                else if (option == "--baseline-root-pw-coefficient") config.baselineRootPwCoefficient = stof(value);
                else if (option == "--contender-mcgs") config.contenderMcgs = parse_bool_argument(value);
                else if (option == "--baseline-mcgs") config.baselineMcgs = parse_bool_argument(value);
                else if (option == "--contender-transpositions") config.contenderTranspositions = parse_bool_argument(value);
                else if (option == "--baseline-transpositions") config.baselineTranspositions = parse_bool_argument(value);
                else if (option == "--contender-root-mate-search") config.contenderRootMateSearch = parse_bool_argument(value);
                else if (option == "--baseline-root-mate-search") config.baselineRootMateSearch = parse_bool_argument(value);
                else if (option == "--contender-wdl-eval") config.contenderWdlEval = parse_bool_argument(value);
                else if (option == "--baseline-wdl-eval") config.baselineWdlEval = parse_bool_argument(value);
                else if (option == "--contender-wdl-weight") config.contenderWdlWeight = stof(value);
                else if (option == "--baseline-wdl-weight") config.baselineWdlWeight = stof(value);
                else if (option == "--contender-moves-left-discount") config.contenderMovesLeftDiscount = stof(value);
                else if (option == "--baseline-moves-left-discount") config.baselineMovesLeftDiscount = stof(value);
                else if (option == "--contender-q-value-weight") config.contenderQValueWeight = stof(value);
                else if (option == "--baseline-q-value-weight") config.baselineQValueWeight = stof(value);
                else if (option == "--contender-q-veto-delta") config.contenderQVetoDelta = stof(value);
                else if (option == "--baseline-q-veto-delta") config.baselineQVetoDelta = stof(value);
                else if (option == "--sprt-elo0") config.sprtElo0 = stod(value);
                else if (option == "--sprt-elo1") config.sprtElo1 = stod(value);
                else if (option == "--sprt-alpha") config.sprtAlpha = stod(value);
                else if (option == "--sprt-beta") config.sprtBeta = stod(value);
                else throw invalid_argument("Unknown tournament option: " + option);
            }
        } catch (const exception& error) {
            cerr << "Invalid tournament arguments: " << error.what() << endl;
            return EXIT_FAILURE;
        }
        if (contenderPath.empty() || baselinePath.empty()) {
            cerr << "Tournament requires --contender <onnx> and --baseline <onnx>" << endl;
            return EXIT_FAILURE;
        }

        Engine contender(0, config.contenderBatchSize);
        Engine baseline(0, config.baselineBatchSize);
        const string contenderEngine = getEnginePath(
            contenderPath.string(), "fp16", config.contenderBatchSize, 0, "v3");
        const string baselineEngine = getEnginePath(
            baselinePath.string(), "fp16", config.baselineBatchSize, 0, "v3");
        if (!contender.loadNetwork(contenderPath.string(), contenderEngine)) {
            cerr << "Failed to load contender model" << endl;
            return EXIT_FAILURE;
        }
        if (!baseline.loadNetwork(baselinePath.string(), baselineEngine)) {
            cerr << "Failed to load baseline model" << endl;
            return EXIT_FAILURE;
        }
        config.contenderModelSignature = computeFileSignature(
            contenderPath.string(), "hivemind-tournament-model");
        config.baselineModelSignature = computeFileSignature(
            baselinePath.string(), "hivemind-tournament-model");
        try {
            return run_tournament(
                contender, baseline,
                contenderPath.stem().string() + "-b" + std::to_string(config.contenderBatchSize),
                baselinePath.stem().string() + "-b" + std::to_string(config.baselineBatchSize),
                config);
        } catch (const exception& error) {
            cerr << "Tournament failed: " << error.what() << endl;
            return EXIT_FAILURE;
        }
    }

    filesystem::path modelPath;
    int uciBatchSize = SearchParams::BATCH_SIZE;
    try {
        for (int i = 1; i < argc; ++i) {
            const string option = argv[i];
            if (option != "--model" && option != "--network"
                && option != "--batch-size") {
                throw invalid_argument("Unknown UCI option: " + option);
            }
            if (i + 1 >= argc) {
                throw invalid_argument("Missing value for " + option);
            }
            const string value = argv[++i];
            if (option == "--batch-size") {
                uciBatchSize = stoi(value);
                if (uciBatchSize < 1 || uciBatchSize > 1024) {
                    throw invalid_argument("--batch-size must be between 1 and 1024");
                }
            } else {
                modelPath = value;
            }
        }
    } catch (const exception& error) {
        cerr << "Invalid UCI arguments: " << error.what() << endl;
        return EXIT_FAILURE;
    }

    UCI uci;
    std::vector<int> deviceIds(deviceCount);
    iota(deviceIds.begin(), deviceIds.end(), 0);

    std::cout << "Hivemind " << HIVEMIND_VERSION << std::endl;

    if (!uci.initializeEngines(deviceIds, modelPath.string(), uciBatchSize)) {
        return EXIT_FAILURE;
    }
    uci.loop();
}
