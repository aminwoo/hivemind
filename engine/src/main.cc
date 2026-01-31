#include "uci.h"
#include "constants.h"
#include "globals.h"
#include "engine.h"
#include "onnx_utils.h"
#include "benchmark.h"
#include "rl/selfplay.h"
#include "rl/model_eval.h"
#include "Fairy-Stockfish/src/bitboard.h"
#include "Fairy-Stockfish/src/position.h"
#include "Fairy-Stockfish/src/thread.h"
#include "Fairy-Stockfish/src/piece.h"
#include "Fairy-Stockfish/src/types.h"
#include <iostream>
#include <cuda_runtime.h>
#include <cstring>

using namespace std; 

void printUsage(const char* progName) {
    cout << "Usage: " << progName << " [options]" << endl;
    cout << "Options:" << endl;
    cout << "  --log <level>      Set log level: none, info, debug (default: none)" << endl;
    cout << "  bench [iters]      Run inference benchmark" << endl;
    cout << "  perft [depth]      Run move generation benchmark" << endl;
    cout << "  selfplay [games]   Run RL self-play (default: 1000 games)" << endl;
    cout << "                     One team has time advantage, the other does not" << endl;
    cout << "  eval               Evaluate two models against each other" << endl;
    cout << "    --new <path>     Path to new model ONNX file" << endl;
    cout << "    --old <path>     Path to old model ONNX file" << endl;
    cout << "    --games <n>      Number of games to play (default: 100)" << endl;
    cout << "    --nodes <n>      MCTS nodes per move (default: 800)" << endl;
    cout << "    --time <ms>      Fixed time per move in ms (default: 0, use nodes)" << endl;
    cout << "    --temperature <f> Temperature for opening moves (default: 0.6)" << endl;
    cout << "    --temp-moves <n> Moves before temperature decays to 0 (default: 15)" << endl;
    cout << "    --verbose        Print each game result" << endl;
    cout << "    --pgn <path>     Save games to PGN file" << endl;
    cout << "    --gui            Enable web GUI for live viewing" << endl;
    cout << endl;
    cout << "  param-eval         Test same model with different search parameters" << endl;
    cout << "    --model <path>   Path to model ONNX file" << endl;
    cout << "    --games <n>      Number of games to play (default: 100)" << endl;
    cout << "    --verbose        Print each game result" << endl;
    cout << "    --pgn <path>     Save games to PGN file" << endl;
    cout << "    --gui            Enable web GUI for live viewing" << endl;
    cout << endl;
    cout << "  Player-specific parameters (use --p1-* or --p2-* prefix):" << endl;
    cout << "    --pX-name <s>    Player name (default: 'Player1/2')" << endl;
    cout << "    --pX-nodes <n>   Nodes per move (default: 800)" << endl;
    cout << "    --pX-time <ms>   Time per move in ms (default: 0, use nodes)" << endl;
    cout << "    --pX-batch <n>   Batch size (default: 8)" << endl;
    cout << "    --pX-cpuct <f>   CPUCT init value (default: 2.5)" << endl;
    cout << "    --pX-fpu <f>     FPU reduction (default: 0.4)" << endl;
    cout << "    --pX-contempt <f> Draw contempt (default: 0.12)" << endl;
    cout << "    --pX-pw-coef <f> Progressive widening coefficient (default: 1.0)" << endl;
    cout << "    --pX-pw-exp <f>  Progressive widening exponent (default: 0.3)" << endl;
    cout << "    --pX-mcgs <0|1>  Enable MCGS (default: 1)" << endl;
    cout << "    --pX-tt <0|1>    Enable transpositions (default: 1)" << endl;
    cout << "    --pX-qweight <f> Q-value weight (default: 1.0)" << endl;
    cout << "    --pX-qveto <f>   Q-value veto delta (default: 0.4)" << endl;
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

    // Check for selfplay flag
    if (argc > 1 && string(argv[1]) == "selfplay") {
        cout << "Starting RL self-play..." << endl;
        
        // Parse command line arguments
        RLSettings settings;
        size_t numberOfGames = settings.numberOfGames;
        
        for (int i = 2; i < argc; i++) {
            string arg = argv[i];
            if ((arg == "--games" || arg == "-g") && i + 1 < argc) {
                numberOfGames = stoul(argv[++i]);
            } else if ((arg == "--nodes" || arg == "-n") && i + 1 < argc) {
                settings.nodesPerMove = stoul(argv[++i]);
            }
        }
        
        // Initialize engines for all GPUs
        vector<Engine*> engines;
        for (int i = 0; i < deviceCount; i++) {
            Engine* engine = new Engine(i);
            const std::string onnxFile = findLatestOnnxFile("./networks");
            cout << "Loading model from: " << onnxFile << " on GPU " << i << endl;
            if (onnxFile.empty()) {
                cerr << "No ONNX file found in ./networks" << endl;
                return EXIT_FAILURE;
            }
            const std::string engineFile = getEnginePath(onnxFile, "fp16", SearchParams::BATCH_SIZE, i, "v1");
            cout << "Using TensorRT engine file: " << engineFile << endl;
            if (!engine->loadNetwork(onnxFile, engineFile)) {
                cerr << "Failed to load engine on GPU " << i << endl;
                return EXIT_FAILURE;
            }
            engines.push_back(engine);
        }
        
        run_selfplay(settings, engines, numberOfGames);
        
        // Cleanup
        for (auto* e : engines) {
            delete e;
        }
        
        return EXIT_SUCCESS;
    }

    // Check for eval flag
    if (argc > 1 && string(argv[1]) == "eval") {
        cout << "Starting model evaluation..." << endl;
        
        EvalSettings settings;
        string newModelPath = "";
        string oldModelPath = "";
        
        // Parse command line arguments
        for (int i = 2; i < argc; i++) {
            string arg = argv[i];
            if (arg == "--new" && i + 1 < argc) {
                newModelPath = argv[++i];
            } else if (arg == "--old" && i + 1 < argc) {
                oldModelPath = argv[++i];
            } else if ((arg == "--games" || arg == "-g") && i + 1 < argc) {
                settings.numGames = stoul(argv[++i]);
            } else if ((arg == "--nodes" || arg == "-n") && i + 1 < argc) {
                settings.nodesPerMove = stoul(argv[++i]);
            } else if (arg == "--time" && i + 1 < argc) {
                settings.moveTimeMs = stoi(argv[++i]);
            } else if ((arg == "--temperature" || arg == "--temp" || arg == "-t") && i + 1 < argc) {
                settings.temperature = stof(argv[++i]);
            } else if (arg == "--temp-moves" && i + 1 < argc) {
                settings.temperatureDecayMoves = stoul(argv[++i]);
            } else if (arg == "--verbose" || arg == "-v") {
                settings.verbose = true;
            } else if (arg == "--pgn" && i + 1 < argc) {
                settings.outputPgnPath = argv[++i];
            } else if (arg == "--gui") {
                settings.enableGui = true;
            } else if (arg == "--gui-path" && i + 1 < argc) {
                settings.guiStatePath = argv[++i];
                settings.enableGui = true;
            }
        }
        
        // Validate model paths
        if (newModelPath.empty() || oldModelPath.empty()) {
            cerr << "Error: Both --new and --old model paths are required" << endl;
            cerr << "Usage: " << argv[0] << " eval --new <path> --old <path> [options]" << endl;
            return EXIT_FAILURE;
        }
        
        run_model_eval(newModelPath, oldModelPath, settings);
        
        return EXIT_SUCCESS;
    }

    // Check for param-eval flag (same model, different parameters)
    if (argc > 1 && string(argv[1]) == "param-eval") {
        cout << "Starting parameter evaluation..." << endl;
        
        EvalSettings settings;
        settings.usePlayerConfigs = true;
        settings.player1.name = "Player1";
        settings.player2.name = "Player2";
        string modelPath = "";
        
        // Parse command line arguments
        for (int i = 2; i < argc; i++) {
            string arg = argv[i];
            if (arg == "--model" && i + 1 < argc) {
                modelPath = argv[++i];
            } else if ((arg == "--games" || arg == "-g") && i + 1 < argc) {
                settings.numGames = stoul(argv[++i]);
            } 
            // Player 1 settings
            else if (arg == "--p1-nodes" && i + 1 < argc) {
                settings.player1.nodesPerMove = stoul(argv[++i]);
            } else if (arg == "--p1-time" && i + 1 < argc) {
                settings.player1.moveTimeMs = stoi(argv[++i]);
            } else if (arg == "--p1-batch" && i + 1 < argc) {
                settings.player1.batchSize = stoi(argv[++i]);
            } else if (arg == "--p1-threads" && i + 1 < argc) {
                settings.player1.numSearchThreads = stoi(argv[++i]);
            } else if (arg == "--p1-cpuct" && i + 1 < argc) {
                settings.player1.cpuctInit = stof(argv[++i]);
            } else if (arg == "--p1-fpu" && i + 1 < argc) {
                settings.player1.fpuReduction = stof(argv[++i]);
            } else if (arg == "--p1-name" && i + 1 < argc) {
                settings.player1.name = argv[++i];
            } else if (arg == "--p1-contempt" && i + 1 < argc) {
                settings.player1.drawContempt = stof(argv[++i]);
            } else if (arg == "--p1-pw-coef" && i + 1 < argc) {
                settings.player1.pwCoefficient = stof(argv[++i]);
            } else if (arg == "--p1-pw-exp" && i + 1 < argc) {
                settings.player1.pwExponent = stof(argv[++i]);
            } else if (arg == "--p1-mcgs" && i + 1 < argc) {
                settings.player1.enableMCGS = (stoi(argv[++i]) != 0);
            } else if (arg == "--p1-tt" && i + 1 < argc) {
                settings.player1.enableTranspositions = (stoi(argv[++i]) != 0);
            } else if (arg == "--p1-qweight" && i + 1 < argc) {
                settings.player1.qValueWeight = stof(argv[++i]);
            } else if (arg == "--p1-qveto" && i + 1 < argc) {
                settings.player1.qVetoDelta = stof(argv[++i]);
            }
            // Player 2 settings
            else if (arg == "--p2-nodes" && i + 1 < argc) {
                settings.player2.nodesPerMove = stoul(argv[++i]);
            } else if (arg == "--p2-time" && i + 1 < argc) {
                settings.player2.moveTimeMs = stoi(argv[++i]);
            } else if (arg == "--p2-batch" && i + 1 < argc) {
                settings.player2.batchSize = stoi(argv[++i]);
            } else if (arg == "--p2-threads" && i + 1 < argc) {
                settings.player2.numSearchThreads = stoi(argv[++i]);
            } else if (arg == "--p2-cpuct" && i + 1 < argc) {
                settings.player2.cpuctInit = stof(argv[++i]);
            } else if (arg == "--p2-fpu" && i + 1 < argc) {
                settings.player2.fpuReduction = stof(argv[++i]);
            } else if (arg == "--p2-name" && i + 1 < argc) {
                settings.player2.name = argv[++i];
            } else if (arg == "--p2-contempt" && i + 1 < argc) {
                settings.player2.drawContempt = stof(argv[++i]);
            } else if (arg == "--p2-pw-coef" && i + 1 < argc) {
                settings.player2.pwCoefficient = stof(argv[++i]);
            } else if (arg == "--p2-pw-exp" && i + 1 < argc) {
                settings.player2.pwExponent = stof(argv[++i]);
            } else if (arg == "--p2-mcgs" && i + 1 < argc) {
                settings.player2.enableMCGS = (stoi(argv[++i]) != 0);
            } else if (arg == "--p2-tt" && i + 1 < argc) {
                settings.player2.enableTranspositions = (stoi(argv[++i]) != 0);
            } else if (arg == "--p2-qweight" && i + 1 < argc) {
                settings.player2.qValueWeight = stof(argv[++i]);
            } else if (arg == "--p2-qveto" && i + 1 < argc) {
                settings.player2.qVetoDelta = stof(argv[++i]);
            }
            // Common settings
            else if (arg == "--verbose" || arg == "-v") {
                settings.verbose = true;
            } else if (arg == "--pgn" && i + 1 < argc) {
                settings.outputPgnPath = argv[++i];
            } else if (arg == "--gui") {
                settings.enableGui = true;
            } else if (arg == "--gui-path" && i + 1 < argc) {
                settings.guiStatePath = argv[++i];
                settings.enableGui = true;
            }
        }
        
        // Validate model path
        if (modelPath.empty()) {
            // Try to find a model in ./networks
            modelPath = findLatestOnnxFile("./networks");
            if (modelPath.empty()) {
                cerr << "Error: --model path is required (or place a model in ./networks)" << endl;
                cerr << "Usage: " << argv[0] << " param-eval --model <path> [options]" << endl;
                return EXIT_FAILURE;
            }
            cout << "Using model: " << modelPath << endl;
        }
        
        run_param_eval(modelPath, settings);
        
        return EXIT_SUCCESS;
    }

    UCI uci;
    std::vector<int> deviceIds(deviceCount);
    iota(deviceIds.begin(), deviceIds.end(), 0);

    std::cout << "HiveMind 1.0" << std::endl;

    uci.initializeEngines(deviceIds);
    uci.loop();
}
