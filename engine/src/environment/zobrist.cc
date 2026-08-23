#include "environment/zobrist.h"
#include <random>

namespace Stockfish {
namespace Zobrist {

// Definition of ply-based Zobrist keys (specific to this project)
// Other Zobrist keys are defined in Fairy-Stockfish
Key ply[MAX_PLY];

// Time advantage key - XOR'd into hash when team has time advantage
Key timeAdvantage;

// Board-only repetition keys (see zobrist.h)
Key boardPsq[PIECE_NB][SQUARE_NB];
Key boardPromoted[SQUARE_NB];
Key boardCastling[16];
Key boardEnPassant[FILE_NB];
Key boardSide;

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

            // Initialize the board-only repetition keys
            for (int piece = 0; piece < PIECE_NB; ++piece) {
                for (int square = 0; square < SQUARE_NB; ++square) {
                    boardPsq[piece][square] = rng();
                }
            }
            for (int square = 0; square < SQUARE_NB; ++square) {
                boardPromoted[square] = rng();
            }
            for (int rights = 0; rights < 16; ++rights) {
                boardCastling[rights] = rng();
            }
            for (int file = 0; file < FILE_NB; ++file) {
                boardEnPassant[file] = rng();
            }
            boardSide = rng();
        }
    } static zobristInit;
}

} // namespace Zobrist
} // namespace Stockfish
