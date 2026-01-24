#include "uci.h"
#include "constants.h"
#include "zobrist.h"
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

using namespace std; 

int main(int argc, char* argv[]) {
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
    if (error_id != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " 
                  << cudaGetErrorString(error_id) << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Number of available GPUs: " << deviceCount << std::endl;

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
