#include "uci.h"
#include "constants.h"
#include "globals.h"
#include "engine.h"
#include "onnx_utils.h"
#include "benchmark.h"
#include "selfplay.h"
#include "tournament.h"
#include "Fairy-Stockfish/src/bitboard.h"
#include "Fairy-Stockfish/src/position.h"
#include "Fairy-Stockfish/src/thread.h"
#include "Fairy-Stockfish/src/piece.h"
#include "Fairy-Stockfish/src/types.h"
#include <iostream>
#include <cuda_runtime.h>
#include <cstring>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <vector>

using namespace std; 

void printUsage(const char* progName) {
    cout << "Usage: " << progName << " [options]" << endl;
    cout << "Options:" << endl;
    cout << "  --log <level>      Set log level: none, info, debug (default: none)" << endl;
    cout << "  --network <onnx>   Load this model in UCI mode instead of scanning ./networks" << endl;
    cout << "  bench [iters]      Run inference benchmark" << endl;
    cout << "  perft [depth]      Run move generation benchmark" << endl;
    cout << "  selfplay [options] Generate HVM3 training chunks and bughouse PGN" << endl;
    cout << "    --network <onnx> --games <n> --nodes <n> --output <dir> --seed <n>" << endl;
    cout << "    --max-macro-plies <n> --raw-policy-mean-macro-plies <x>" << endl;
    cout << "    --raw-policy-max-macro-plies <n> --raw-policy-high-temp-probability <x>" << endl;
    cout << "    --mcts-temperature <x> --mcts-temperature-decay <x>" << endl;
    cout << "    --node-random-factor <x>" << endl;
    cout << "    --chunk-samples <n> --dirichlet-alpha <x> --dirichlet-epsilon <x>" << endl;
    cout << "    --wait-pass-prior-floor <x> --coordination-pass-prior-floor <x>" << endl;
    cout << "  tournament [options] Run a paired network-vs-network tournament" << endl;
    cout << "    --contender <onnx> --baseline <onnx> --games <even-n> --nodes <n>" << endl;
    cout << "    --output <dir> --seed <n> --max-macro-plies <n>" << endl;
    cout << "    --dirichlet-alpha <x> --dirichlet-epsilon <x>" << endl;
    cout << "    --contender-pw-coefficient <x> --baseline-pw-coefficient <x>" << endl;
    cout << "    --contender-wait-pass-prior-floor <x>" << endl;
    cout << "    --contender-coordination-pass-prior-floor <x>" << endl;
    cout << "    --baseline-wait-pass-prior-floor <x>" << endl;
    cout << "    --baseline-coordination-pass-prior-floor <x>" << endl;
}

int main(int argc, char* argv[]) {
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
    if (error_id != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " 
                  << cudaGetErrorString(error_id) << std::endl;
        return EXIT_FAILURE;
    }

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

    Stockfish::pieceMap.init();
    Stockfish::variants.init();
    Stockfish::Bitboards::init();
    Stockfish::Position::init();
    Stockfish::Threads.set(1);

    init_policy_index();

    // Check for benchmark flag
    if (argc > 1 && string(argv[1]) == "bench") {
        cout << "Running inference benchmark..." << endl;
        Engine engine(0);
        
        const std::string onnxFile = findLatestOnnxFile("./networks");
        if (onnxFile.empty()) {
            cerr << "No ONNX file found in ./networks" << endl;
            return EXIT_FAILURE;
        }
        const std::string engineFile = getEnginePath(onnxFile, "fp16", SearchParams::BATCH_SIZE, 0, "v1");
        
        if (!engine.loadNetwork(onnxFile, engineFile)) {
            cerr << "Failed to load engine" << endl;
            return EXIT_FAILURE;
        }
        
        int iterations = (argc > 2) ? stoi(argv[2]) : 1000;
        benchmark_inference(engine, iterations);
        return EXIT_SUCCESS;
    }

    // Check for perft benchmark flag
    if (argc > 1 && string(argv[1]) == "perft") {
        int depth = (argc > 2) ? stoi(argv[2]) : 5;
        benchmark_movegen(depth);
        return EXIT_SUCCESS;
    }

    if (argc > 1 && string(argv[1]) == "selfplay") {
        SelfPlayConfig config;
        filesystem::path networkPath;
        try {
            for (int i = 2; i < argc; ++i) {
                const string option = argv[i];
                if (i + 1 >= argc) {
                    throw invalid_argument("Missing value for " + option);
                }
                const string value = argv[++i];
                if (option == "--games") config.games = stoull(value);
                else if (option == "--nodes") config.nodes = stoull(value);
                else if (option == "--network") networkPath = value;
                else if (option == "--output") config.outputDirectory = value;
                else if (option == "--seed") config.seed = stoull(value);
                else if (option == "--max-macro-plies") config.maxMacroPlies = stoull(value);
                else if (option == "--raw-policy-mean-macro-plies") config.rawPolicyMeanMacroPlies = stod(value);
                else if (option == "--raw-policy-max-macro-plies") config.rawPolicyMaxMacroPlies = stoull(value);
                else if (option == "--raw-policy-high-temp-probability") config.rawPolicyHighTemperatureProbability = stod(value);
                else if (option == "--mcts-temperature") config.mctsTemperature = stod(value);
                else if (option == "--mcts-temperature-decay") config.mctsTemperatureDecay = stod(value);
                else if (option == "--node-random-factor") config.nodeRandomFactor = stod(value);
                else if (option == "--chunk-samples") config.chunkSamples = stoull(value);
                else if (option == "--dirichlet-alpha") config.dirichletAlpha = stof(value);
                else if (option == "--dirichlet-epsilon") config.dirichletEpsilon = stof(value);
                else if (option == "--wait-pass-prior-floor") config.waitPassPriorFloor = stof(value);
                else if (option == "--coordination-pass-prior-floor") config.coordinationPassPriorFloor = stof(value);
                else throw invalid_argument("Unknown selfplay option: " + option);
            }
        } catch (const exception& error) {
            cerr << "Invalid selfplay arguments: " << error.what() << endl;
            return EXIT_FAILURE;
        }

        if (networkPath.empty()) {
            const filesystem::path networkDirectory = filesystem::exists("./networks")
                ? filesystem::path("./networks")
                : filesystem::path("./engine/networks");
            if (filesystem::exists(networkDirectory)) {
                networkPath = findLatestOnnxFile(networkDirectory.string());
            }
        }
        const string onnxFile = networkPath.string();
        if (onnxFile.empty()) {
            cerr << "No ONNX model found; pass --network <onnx>" << endl;
            return EXIT_FAILURE;
        }
        vector<unique_ptr<Engine>> ownedEngines;
        vector<Engine*> engines;
        for (int deviceId = 0; deviceId < deviceCount; ++deviceId) {
            auto engine = make_unique<Engine>(deviceId);
            const string engineFile = getEnginePath(
                onnxFile, "fp16", SearchParams::BATCH_SIZE, deviceId, "v1");
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
                else if (option == "--nodes") config.nodes = stoull(value);
                else if (option == "--contender") contenderPath = value;
                else if (option == "--baseline") baselinePath = value;
                else if (option == "--output") config.outputDirectory = value;
                else if (option == "--seed") config.seed = stoull(value);
                else if (option == "--max-macro-plies") config.maxMacroPlies = stoull(value);
                else if (option == "--dirichlet-alpha") config.dirichletAlpha = stof(value);
                else if (option == "--dirichlet-epsilon") config.dirichletEpsilon = stof(value);
                else if (option == "--contender-pw-coefficient") config.contenderPwCoefficient = stof(value);
                else if (option == "--baseline-pw-coefficient") config.baselinePwCoefficient = stof(value);
                else if (option == "--contender-wait-pass-prior-floor") config.contenderPassPriorFloors.wait = stof(value);
                else if (option == "--contender-coordination-pass-prior-floor") config.contenderPassPriorFloors.coordination = stof(value);
                else if (option == "--baseline-wait-pass-prior-floor") config.baselinePassPriorFloors.wait = stof(value);
                else if (option == "--baseline-coordination-pass-prior-floor") config.baselinePassPriorFloors.coordination = stof(value);
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

        Engine contender(0);
        Engine baseline(0);
        const string contenderEngine = getEnginePath(
            contenderPath.string(), "fp16", SearchParams::BATCH_SIZE, 0, "v1");
        const string baselineEngine = getEnginePath(
            baselinePath.string(), "fp16", SearchParams::BATCH_SIZE, 0, "v1");
        if (!contender.loadNetwork(contenderPath.string(), contenderEngine)) {
            cerr << "Failed to load contender network" << endl;
            return EXIT_FAILURE;
        }
        if (!baseline.loadNetwork(baselinePath.string(), baselineEngine)) {
            cerr << "Failed to load baseline network" << endl;
            return EXIT_FAILURE;
        }
        try {
            return run_tournament(
                contender, baseline,
                contenderPath.stem().string(), baselinePath.stem().string(),
                config);
        } catch (const exception& error) {
            cerr << "Tournament failed: " << error.what() << endl;
            return EXIT_FAILURE;
        }
    }

    filesystem::path networkPath;
    try {
        for (int i = 1; i < argc; ++i) {
            const string option = argv[i];
            if (option != "--network") {
                throw invalid_argument("Unknown UCI option: " + option);
            }
            if (i + 1 >= argc) {
                throw invalid_argument("Missing value for --network");
            }
            networkPath = argv[++i];
        }
    } catch (const exception& error) {
        cerr << "Invalid UCI arguments: " << error.what() << endl;
        return EXIT_FAILURE;
    }

    UCI uci;
    std::vector<int> deviceIds(deviceCount);
    iota(deviceIds.begin(), deviceIds.end(), 0);

    std::cout << "Hivemind 1.0" << std::endl;

    if (!uci.initializeEngines(deviceIds, networkPath.string())) {
        return EXIT_FAILURE;
    }
    uci.loop();
}
