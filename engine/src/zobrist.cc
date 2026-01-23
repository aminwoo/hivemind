#include "zobrist.h"
#include <random>

namespace Stockfish {
namespace Zobrist {

// Definition of ply-based Zobrist keys (specific to this project)
// Other Zobrist keys are defined in Fairy-Stockfish
Key ply[MAX_PLY];

// Time advantage key - XOR'd into hash when team has time advantage
Key timeAdvantage;

// Static initialization
namespace {
    struct ZobristInit {
        ZobristInit() {
            std::mt19937_64 rng(1070372);  // Fixed seed for reproducibility
            
            // Initialize ply-based Zobrist keys
            for (int i = 0; i < MAX_PLY; ++i) {
                ply[i] = rng();
            }
            
            // Initialize time advantage key
            timeAdvantage = rng();
        }
    } static zobristInit;
}

} // namespace Zobrist
} // namespace Stockfish
