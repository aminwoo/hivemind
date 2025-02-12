#include "uci.h"
#include "constants.h"
#include "Fairy-Stockfish/src/bitboard.h"
#include "Fairy-Stockfish/src/position.h"
#include "Fairy-Stockfish/src/thread.h"
#include "Fairy-Stockfish/src/piece.h"
#include "Fairy-Stockfish/src/types.h"
#include <iostream>
#include <cuda_runtime.h>

std::unordered_map<std::string, int> POLICY_INDEX; 

using namespace std; 

int main() {
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

    for (int i = 0; i < NB_POLICY_VALUES(); i++) {
        if (POLICY_INDEX.find(UCI_MOVES[i]) == POLICY_INDEX.end()) {
            POLICY_INDEX[UCI_MOVES[i]] = i; 
        }
    }

    UCI uci;
    std::vector<int> deviceIds(deviceCount);
    iota(deviceIds.begin(), deviceIds.end(), 0);

    uci.initializeEngines(deviceIds);
    uci.loop();
}
