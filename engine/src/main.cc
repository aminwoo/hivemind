#include "uci.h"
#include "constants.h"
#include "globals.h"
#include "engine.h"
#include "onnx_utils.h"
#include "benchmark.h"
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
    cout << "  --log <level>   Set log level: none, info, debug (default: none)" << endl;
    cout << "  bench [iters]   Run inference benchmark" << endl;
    cout << "  perft [depth]   Run move generation benchmark" << endl;
}

int main(int argc, char* argv[]) {
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
    if (error_id != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " 
                  << cudaGetErrorString(error_id) << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Number of available GPUs: " << deviceCount << std::endl;

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

    UCI uci;
    std::vector<int> deviceIds(deviceCount);
    iota(deviceIds.begin(), deviceIds.end(), 0);

    uci.initializeEngines(deviceIds);
    uci.loop();
}
